"""Verify upstream boundaries without loading weights or running model inference."""

from types import ModuleType, SimpleNamespace

import pytest

from s1mb.adapters.base import questions_for_api
from s1mb.adapters.minojev import MinojevAdapter
from s1mb.adapters.upstream import UpstreamAdapter, candidate_batches
from s1mb.data import Case


@pytest.mark.parametrize("masked", [False, True])
def test_fp32_sdpa_preserves_grouped_attention_and_padding(monkeypatch, masked):
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from s1mb.adapters.minojev import fp32_sdpa

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda backend: nullcontext())
    torch.manual_seed(11)
    query = torch.randn(2, 4, 7, 8)
    key, value = torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
    mask = None
    if masked:
        mask = torch.ones(2, 1, 7, 7, dtype=torch.bool).tril()
        mask[1, :, :, 5:] = False
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=not masked, enable_gqa=True
    )
    actual, _ = fp32_sdpa(SimpleNamespace(is_causal=True), query, key, value, mask)
    torch.testing.assert_close(actual, expected.transpose(1, 2))


def cases():
    return Case.model_validate(
        {
            "case_id": "boundary",
            "group_id": "g",
            "language": "en",
            "state": {"evidence": "text"},
            "questions": [
                {
                    "id": "choose",
                    "task": "choice",
                    "system_prompt": "System.",
                    "instructions": "Choose.",
                    "options": [
                        {"id": "x", "description": "Same"},
                        {"id": "y", "description": "Same"},
                    ],
                },
                {
                    "id": "judge",
                    "task": "noul",
                    "instructions": "Judge.",
                    "options": [
                        {"id": "true", "description": "Yes"},
                        {"id": "false", "description": "No"},
                    ],
                },
                {
                    "id": "rate",
                    "task": "score",
                    "instructions": "Rate.",
                    "options": [
                        {"id": "low", "description": "Low", "value": 10},
                        {"id": "high", "description": "High", "value": 30},
                    ],
                },
            ],
            "targets": {
                "choose": {"kind": "hard_label", "probabilities": {"x": 1, "y": 0}},
                "judge": {"kind": "hard_label", "probabilities": {"true": 0, "false": 1}},
                "rate": {"kind": "hard_label", "probabilities": {"low": 1, "high": 0}},
            },
            "provenance": {"secret": "gold-must-not-be-sent"},
        }
    ).inference()


def answer(definition):
    if definition["type"] == "choice":
        return {"type": "choice", "probabilities": {"x": 0.2, "y": 0.8}}
    if definition["type"] == "noul":
        return {"type": "noul", "noul": 0.7}
    return {"type": "score", "probabilities": {"0": 0.3, "1": 0.7}}


def test_native_api_preserves_instructions_and_maps_anonymous_choice_ids():
    from s1mb.adapters.base import decode_answers

    case = cases()
    request = questions_for_api(case.questions, anonymous_choice=True)
    assert "gold-must-not-be-sent" not in str(request)
    assert request["choose"]["criteria"] == {"option_0": "Same", "option_1": "Same"}
    assert request["choose"]["instructions"] == "System.\n\nChoose."
    assert request["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert request["rate"]["criteria"] == ["Low", "High"]
    answers = {key: answer(value) for key, value in request.items()}
    answers["choose"]["probabilities"] = {"option_0": 0.2, "option_1": 0.8}
    predictions = decode_answers(case, answers, anonymous_choice=True)
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities is not None
    assert predictions[1].probabilities["true"] == pytest.approx(0.7)
    assert predictions[2].probabilities == {"low": 0.3, "high": 0.7}
    answers["choose"]["probabilities"] = {"x": 0.2, "y": 0.8}
    with pytest.raises(ValueError, match="Anonymous choice keys"):
        decode_answers(case, answers, anonymous_choice=True)


def test_laya_full_sequence_retains_long_options_and_rejects_overflow():
    from s1mb.adapters.laya import full_sequence

    class Tokenizer:
        mask_token = "[MASK]"
        cls_token_id, sep_token_id, mask_token_id = 1, 2, 3

        def __call__(self, texts, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return {"input_ids": [[ord(c) for c in text] for text in texts]}

    common = SimpleNamespace(
        render_options=lambda q: q["options"], serialize_state=lambda s: s, QTYPES={"choice": 0}
    )
    q = {"t": "choice", "ins": "choose", "options": ["a" * 100, "b" * 100]}
    result = full_sequence(Tokenizer(), "evidence" * 100, q, common, 2048)
    assert result["ids"].count(ord("a")) >= 100
    assert result["ids"].count(ord("b")) == 100
    assert len(result["markers"]) == 2
    with pytest.raises(ValueError, match="exceeding"):
        full_sequence(Tokenizer(), "evidence" * 100, q, common, 512)
    with pytest.raises(ValueError, match="head_max_len"):
        full_sequence(Tokenizer(), "evidence", q, common, 2048, 32)


@pytest.mark.parametrize("subfolder", ["..", ".", "", "../weights", "nested/weights"])
def test_checkpoint_subfolder_is_validated_before_network_access(subfolder):
    from s1mb.adapters.upstream import checkpoint_path

    with pytest.raises(ValueError, match="single directory name"):
        checkpoint_path("unreachable/model", "main", subfolder)


def test_no_automatic_cpu_fallback():
    adapter = UpstreamAdapter()
    with pytest.raises(ValueError, match="no CPU fallback"):
        adapter.setup("test", "unreachable/model", "main", "missing-source", "cpu")


def test_candidate_memory_budget_keeps_every_path_in_order():
    lengths = [200, 200, 3000, 5000, 100, 100]
    batches = list(candidate_batches(list(range(6)), lengths))
    assert [item for batch in batches for item in batch] == list(range(6))
    for batch in batches:
        assert len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 4096
    assert [3] in batches  # An oversized path is not truncated or dropped.


def test_minojev_accepts_native_fp32_error_but_rejects_invalid_distribution(monkeypatch):
    case = cases()
    case = case.model_copy(update={"questions": case.questions[:1]})
    adapter = object.__new__(MinojevAdapter)
    adapter.types = ModuleType("mock_minojev_types")
    monkeypatch.setattr(adapter.types, "request_from_object", lambda value: value, raising=False)
    adapter.options = None
    row = {"candidate_ids": ["x", "y"], "probabilities": [0.99809164, 0.00190824]}
    adapter.engine = SimpleNamespace(score=lambda *_: [row])
    assert sum(adapter.predict(case)[0].probabilities.values()) == pytest.approx(1)
    row["probabilities"] = [0.9, 0.2]
    with pytest.raises(ValueError, match="rounding tolerance"):
        adapter.predict(case)


def test_attention_change_must_be_effective():
    adapter = UpstreamAdapter()
    adapter.name = "luce"
    adapter.settings = {}
    adapter.attention_model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="eager"),
        set_attn_implementation=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="did not activate"):
        adapter.set_attention("sdpa")
    assert "attention_implementation" not in adapter.settings


def test_von_rejects_backend_that_cannot_represent_candidate_isolation():
    adapter = UpstreamAdapter()
    adapter.name = "von"
    with pytest.raises(ValueError, match="4D masks"):
        adapter.set_attention("flash_attention_2")


def test_von_literal_markers_are_not_candidate_delimiters():
    import re

    from s1mb.adapters.von import candidate_marker_positions

    def pack(state, question, options):
        prefix = f"{question} {state}".strip()
        return prefix + " [SEP] " + " ".join("[MASK] " + o.strip() for o in options)

    def tokenize(text, **_):
        # ModernBERT's AddedToken includes preceding whitespace in its offset.
        matches = list(re.finditer(r"\s?\[MASK\]", text))
        return {"input_ids": [99] * len(matches), "offset_mapping": [m.span() for m in matches]}

    network = SimpleNamespace(pack_sequence=pack, tokenizer=tokenize, mask_token_id=99)
    ids, positions = candidate_marker_positions(
        network, "Paper discusses [MASK].", "Explain [MASK].", ["Literal [MASK]", "Other"]
    )
    assert len(ids) == 5
    assert positions == [2, 4]


def test_pinned_checkpoint_download_preserves_sha_without_revision_lookup(monkeypatch, tmp_path):
    from s1mb.adapters.upstream import checkpoint_path

    revision = "a" * 40
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        return str(tmp_path)

    hub = SimpleNamespace(snapshot_download=download)
    monkeypatch.setattr("s1mb.adapters.upstream.importlib.import_module", lambda name: hub)
    root, resolved = checkpoint_path("author/model", revision)
    assert root == tmp_path
    assert resolved == revision
    assert calls[0][0] == "author/model"
    assert calls[0][1]["revision"] == revision


def test_missing_pinned_cache_downloads_the_same_revision(monkeypatch, tmp_path):
    from s1mb.adapters.upstream import checkpoint_path

    class MissingCache(Exception):
        pass

    revision = "b" * 40
    calls = []

    def download(model, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise MissingCache
        return str(tmp_path)

    hub = SimpleNamespace(snapshot_download=download,
                          errors=SimpleNamespace(LocalEntryNotFoundError=MissingCache))
    monkeypatch.setattr("s1mb.adapters.upstream.importlib.import_module", lambda name: hub)
    assert checkpoint_path("author/model", revision) == (tmp_path, revision)
    assert len(calls) == 2
    assert all(call["revision"] == revision for call in calls)
    assert not calls[1].get("local_files_only")
