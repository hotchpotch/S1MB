"""Synthetic fidelity and overflow checks for additional typed model bridges."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.bosun import bosun_candidates
from s1mb.adapters.lumma import LummaAdapter
from s1mb.adapters.nimble import NimbleAdapter, nimble_field
from s1mb.adapters.tinyjev import TinyJevAdapter


def test_nimble_preserves_authored_options_and_numeric_levels():
    case = cases()
    keys, schema = nimble_field(case.questions[0])
    assert keys == ["option_0", "option_1"]
    assert schema["decision"]["description"] == "System.\n\nChoose."
    assert schema["decision"]["choice_descriptions"] == {"option_0": "Same", "option_1": "Same"}
    keys, schema = nimble_field(case.questions[1])
    assert keys == ["true", "false"]
    assert schema["decision"]["choice_descriptions"] == {"true": "Yes", "false": "No"}
    keys, schema = nimble_field(case.questions[2])
    assert keys == ["level_0: 10.0", "level_1: 30.0"]
    assert "gold-must-not-be-sent" not in str(schema)


def test_nimble_maps_probabilities_and_rejects_wrong_candidates():
    adapter = NimbleAdapter.__new__(NimbleAdapter)

    def score(state, schema):
        assert state == '{"evidence": "text"}'
        keys = schema["decision"]["choices"]
        return {"fields": {"decision": {"probabilities": dict(zip(keys, [0.2, 0.8]))}}}

    adapter.engine = SimpleNamespace(score=score)
    result = adapter.predict(cases())
    assert result[1].probabilities == {"true": 0.2, "false": 0.8}
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}
    adapter.engine.score = lambda *a: {"fields": {"decision": {"probabilities": {"bad": 1}}}}
    with pytest.raises(ValueError, match="unexpected"):
        adapter.predict(cases())


def test_lumma_rejects_state_before_native_truncation():
    adapter = LummaAdapter.__new__(LummaAdapter)
    adapter.native = SimpleNamespace(render=str)
    adapter.engine = SimpleNamespace(
        tokenizer=None,
        text_ids=lambda *a: [1, 2, 3],
        limits=lambda: (3, 10),
        decide=lambda *a: pytest.fail("overflow must be rejected before inference"),
    )
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_tinyjev_keeps_independent_questions_and_authored_boolean_meanings():
    adapter = TinyJevAdapter.__new__(TinyJevAdapter)

    def systemone(request):
        assert request["state"] == {"evidence": "text"}
        assert len(request["questions"]) == 1
        key, q = next(iter(request["questions"].items()))
        if q["type"] == "noul":
            assert q["criteria"] == {"true": "Yes", "false": "No"}
            answer = {"type": "noul", "noul": 0.7}
        else:
            keys = list(q["criteria"]) if q["type"] == "choice" else ["0", "1"]
            answer = {"type": q["type"], "probabilities": dict(zip(keys, [0.2, 0.8]))}
        return {"answers": {key: answer}}

    adapter.engine = SimpleNamespace(systemone=systemone)
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


def test_bosun_uses_anonymous_ids_but_original_values():
    assert bosun_candidates(cases().questions[2]) == [
        {"id": "option_0", "label": "10.0", "description": "Low"},
        {"id": "option_1", "label": "30.0", "description": "High"},
    ]


def test_manchego_native_extended_menu_preserves_every_candidate():
    from s1mb.adapters.manchego import manchego_prompt
    from s1mb.data import Option

    q = (
        cases()
        .questions[0]
        .model_copy(
            update={
                "options": [Option(id=f"secret{i}", description=f"Item {i}") for i in range(27)]
            }
        )
    )
    seen = {}

    def render(tokenizer, state, task, instruction, options, codes):
        seen.update(options=options, state=state, instruction=instruction)
        return "native extended prompt"

    native = SimpleNamespace(codebook=lambda n: list(range(n)), render=render)
    prompt, codes = manchego_prompt(native, None, {"text": "evidence"}, q)
    assert len(codes) == 27
    assert seen["options"][-1] == ("option_26", "Item 26")
    assert "secret" not in str(seen)
    assert prompt == "native extended prompt"


def test_verdict_rejects_candidate_overflow_before_native_truncation():
    from s1mb.adapters.verdict_encoder import VerdictEncoderAdapter

    case = cases()
    q = case.questions[0].model_copy(update={"options": case.questions[0].options * 13})
    adapter = VerdictEncoderAdapter.__new__(VerdictEncoderAdapter)
    with pytest.raises(ValueError, match="24 candidates"):
        adapter.predict(case.model_copy(update={"questions": [q]}))


def test_kotoba_rejects_state_before_collator_slices():
    from s1mb.adapters.kotoba import KotobaAdapter

    adapter = KotobaAdapter.__new__(KotobaAdapter)
    adapter.engine = SimpleNamespace(collator=SimpleNamespace(_ids=lambda x: [1, 2], max_state=1))
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_winnow_rejects_public_binding_before_loading_weights():
    from s1mb.adapters.winnow import WinnowAdapter

    with pytest.raises(ValueError, match="localhost or a Tailscale"):
        WinnowAdapter("unused", "unused", "unused", "cuda:0", server_host="0.0.0.0")


def test_winnow_keeps_structured_state_and_authored_questions(monkeypatch):
    from s1mb.adapters.winnow import WinnowAdapter

    adapter = WinnowAdapter.__new__(WinnowAdapter)
    requests = []

    def request(path, body=None):
        assert body is not None
        requests.append(body)
        question = body["questions"]["decision"]
        assert body["state"] == {"evidence": "text"}
        if question["type"] == "noul":
            assert question["criteria"] == {"true": "Yes", "false": "No"}
            answer = {"type": "noul", "noul": 0.8}
        else:
            keys = list(question["criteria"]) if question["type"] == "choice" else ["0", "1"]
            answer = {"type": question["type"], "probabilities": dict(zip(keys, [0.3, 0.7]))}
        return {"answers": {"decision": answer}}

    monkeypatch.setattr(adapter, "request", request)
    result = adapter.predict(cases())
    assert len(requests) == 3
    assert result[0].probabilities == {"x": 0.3, "y": 0.7}
    assert result[2].probabilities == {"low": 0.3, "high": 0.7}
    assert "gold-must-not-be-sent" not in str(requests)


def test_apus_uses_choice_for_authored_noul_and_numeric_score():
    from s1mb.adapters.apus import ApusAdapter

    seen = []
    adapter = ApusAdapter.__new__(ApusAdapter)

    def decide(request, effort):
        seen.append(request)
        assert effort == "high" and request["primitive"] == "choice"
        return {"probabilities": {c["id"]: p for c, p in zip(request["criteria"], [0.25, 0.75])}}

    adapter.engine = SimpleNamespace(decide=decide)
    result = adapter.predict(cases())
    assert seen[1]["criteria"][0]["description"] == "true: Yes"
    assert seen[2]["criteria"][1]["description"] == "30.0: High"
    assert result[2].probabilities == {"low": 0.25, "high": 0.75}
    assert "gold-must-not-be-sent" not in str(seen)


def test_tinyjev_parameter_metadata_includes_native_numpy_head(monkeypatch):
    from s1mb.adapters.upstream import UpstreamAdapter
    from s1mb.data import ModelInfo

    monkeypatch.setattr(
        UpstreamAdapter,
        "metadata",
        lambda self: ModelInfo(
            id="test",
            adapter="tinyjev",
            revision="test",
            settings={},
            total_params=100,
            active_params=80,
            parameter_count_method="non_lookup_parameters_v1",
        ),
    )
    adapter = TinyJevAdapter.__new__(TinyJevAdapter)
    adapter.engine = SimpleNamespace(
        family=SimpleNamespace(
            w={"weight": SimpleNamespace(size=8), "bias": SimpleNamespace(size=2)}
        )
    )
    info = adapter.metadata()
    assert info.total_params == 110
    assert info.active_params == 90
    assert info.settings["numpy_head_params"] == 10


def test_jeff_codebook_excludes_split_and_colliding_label_tokens():
    from s1mb.adapters.jeff import anonymous_choice_codes

    def tokenizer(labels, **kwargs):
        return {"input_ids": [{" A": [1], " B": [1], " C": [2]}.get(x, [3, 4]) for x in labels]}

    assert anonymous_choice_codes(tokenizer, 2) == ["A", "C"]
    with pytest.raises(ValueError, match="too few"):
        anonymous_choice_codes(tokenizer, 3)


def test_winnow_port_check_allows_closed_connections_but_rejects_listener():
    import socket

    import pytest

    from s1mb.adapters.winnow import check_port_available

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()
        with pytest.raises(OSError):
            check_port_available(host, port)
        with socket.create_connection((host, port), timeout=2) as client:
            connection, _ = listener.accept()
            connection.close()
            assert client.recv(1) == b""
    check_port_available(host, port)


def test_decoder_placement_keeps_head_on_gpu_and_checks_capacity(monkeypatch):
    from contextlib import nullcontext

    from s1mb.adapters import upstream

    body = SimpleNamespace(
        layers=[None] * 4,
        named_children=lambda: [
            (n, None) for n in ["embed_tokens", "layers", "norm", "rotary_emb"]
        ],
    )
    template = SimpleNamespace(
        model=body, named_children=lambda: [("model", body), ("lm_head", None)]
    )
    sizes = {
        "model.embed_tokens": 20,
        "model.norm": 0,
        "model.rotary_emb": 0,
        "lm_head": 20,
        **{f"model.layers.{i}": 25 for i in range(4)},
    }
    accelerate = SimpleNamespace(
        init_empty_weights=nullcontext,
        utils=SimpleNamespace(compute_module_sizes=lambda *a, **kw: sizes),
    )
    monkeypatch.setattr(upstream.importlib, "import_module", lambda name: accelerate)
    torch = SimpleNamespace(
        bfloat16="bf16",
        cuda=SimpleNamespace(device_count=lambda: 2, mem_get_info=lambda gpu: (100, 100)),
    )
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda path: object()),
        AutoModelForCausalLM=SimpleNamespace(from_config=lambda *a, **kw: template),
    )
    placement = upstream.decoder_device_map("model", "cuda:0", torch, transformers)
    assert placement["model.embed_tokens"] == 0
    assert [placement[f"model.layers.{i}"] for i in range(4)] == [0, 0, 1, 1]
    assert placement["lm_head"] == 1
    assert set(placement.values()) == {0, 1}
    torch.cuda.mem_get_info = lambda gpu: (70, 100)
    with pytest.raises(ValueError, match="memory budget"):
        upstream.decoder_device_map("model", "cuda:0", torch, transformers)
