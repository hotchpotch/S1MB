"""Jalebi-E rendering, overflow policy and prediction shape with a tiny random model; no download, no GPU."""

import pytest
from test_upstream_adapters import cases

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")

from s1mb.adapters import jalebi_e_model as model_module
from s1mb.adapters.jalebi_e import JalebiEAdapter, record

WORDS = ["evidence", "text", "choose", "judge", "rate", "same", "yes", "no", "low", "high", "a", "b", "c", "d", "e", "f"]
ID = {word: 6 + i for i, word in enumerate(WORDS)}              # ids 0..5 are PAD, UNK, CLS, SEP, MASK, OPTION


def tiny_engine():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[OPTION]": 5, **ID}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    torch.manual_seed(0)
    config = model_module.Config(vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
    engine = model_module.JalebiE(config).eval()
    engine.tokenizer = tokenizer
    return engine


def adapter(**options):
    return JalebiEAdapter("tiny", "0" * 40, "cpu", engine=tiny_engine(), **options)


def test_layout_special_tokens_and_option_positions():
    ids, positions, dropped = tiny_engine().encode("a b", "c", ["d", "e f"])
    assert ids == [2, ID["c"], 3, ID["a"], ID["b"], 3, 5, ID["d"], 5, ID["e"], ID["f"], 3]
    assert positions == [6, 8] and dropped == 0


def test_overflow_is_rejected_by_default_and_state_trimmed_only_on_opt_in(monkeypatch):
    monkeypatch.setattr(model_module, "MIN_STATE", 4)
    engine = tiny_engine()
    state = " ".join(["a"] * 80)
    with pytest.raises(ValueError, match="refusing truncation"):
        engine.encode(state, "c", ["d", "e"], window=64)
    ids, positions, dropped = engine.encode(state, "c", ["d", "e"], trim=True, window=64)
    assert len(ids) == 64 and dropped == 80 - (64 - 5 - 4)
    assert ids[:3] == [2, ID["c"], 3] and ids[-1] == 3 and len(positions) == 2   # question and every option intact


def test_options_are_never_split_and_a_tiny_state_budget_is_refused(monkeypatch):
    monkeypatch.setattr(model_module, "MIN_STATE", 4)
    engine = tiny_engine()
    with pytest.raises(ValueError, match="fewer than 4 state tokens"):
        engine.encode(" ".join(["a"] * 20), "c", [" ".join(["d"] * 28), " ".join(["e"] * 28)], trim=True, window=64)
    with pytest.raises(ValueError, match="refusing truncation"):
        engine.encode("", "c", [" ".join(["d"] * 40), " ".join(["e"] * 40)], trim=True, window=64)


def test_caps_apply_only_with_trim(monkeypatch):
    monkeypatch.setattr(model_module, "QUESTION_CAP", 3)
    monkeypatch.setattr(model_module, "OPTION_CAP", 2)
    engine = tiny_engine()
    args = ("a", "a b c d e", ["a b c d", "e f"])
    assert len(engine.encode(*args, trim=True)[0]) == (3 + 4) + 1 + (1 + 2) + (1 + 2)
    assert len(engine.encode(*args)[0]) == (5 + 4) + 1 + (1 + 4) + (1 + 2)


def test_record_uses_authored_text_without_targets_or_ids():
    case = cases()
    state, instructions, options = record(case, case.questions[0])
    assert instructions == "Choose." and options == ["Same", "Same"]
    assert "gold-must-not-be-sent" not in state and "boundary" not in state and "choose" not in state
    assert record(case, case.questions[1])[2] == ["Yes", "No"]       # declared order is kept


def test_predictions_follow_the_declared_option_ids():
    case = cases()
    out = adapter().predict(case)
    assert [p.question_id for p in out] == [q.id for q in case.questions]
    for prediction, question in zip(out, case.questions, strict=True):
        assert list(prediction.probabilities) == [o.id for o in question.options]
        assert sum(prediction.probabilities.values()) == pytest.approx(1)


def test_overflowing_question_is_an_error_not_a_guess():
    case = cases().model_copy(update={"state": " ".join(["a"] * 2100)})
    strict = adapter()
    out = strict.predict(case)
    assert all(p.probabilities is None and "refusing truncation" in p.error for p in out)
    assert strict.metadata().settings["run_counts"]["rejected"] == len(case.questions)


def test_metadata_records_policy_and_parameters():
    reject = adapter().metadata()
    assert reject.adapter == "jalebi-e" and reject.revision == "0" * 40
    assert reject.settings["input_length_policy"] == "reject-overflow"
    assert reject.settings["trim_question_cap"] is None
    trim = adapter(native_adaptive_budget=True).metadata()
    assert trim.settings["input_length_policy"] == "trim-state"
    assert trim.settings["trim_min_state_tokens"] == model_module.MIN_STATE
    assert reject.total_params and trim.total_params == reject.total_params


def test_batching_preserves_case_alignment():
    first, second = cases(), cases().model_copy(update={"case_id": "second"})
    batch = adapter().predict_batch([first, second])
    assert [p.case_id for p in batch[0]] == [first.case_id] * 3
    assert [p.case_id for p in batch[1]] == ["second"] * 3


def test_cuda_without_a_device_refuses_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        JalebiEAdapter("tiny", "0" * 40, "cuda:0", engine=tiny_engine())
