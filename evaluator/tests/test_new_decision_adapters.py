"""Synthetic boundary checks for independently published decision checkpoints."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.certo import CertoAdapter, certo_inputs
from s1mb.adapters.jev_omni import JevOmniAdapter, omni_inputs
from s1mb.adapters.jevlite import JevLiteAdapter, extended_labels
from s1mb.adapters.jevone import JevOneAdapter
from s1mb.adapters.needle import NeedleAdapter
from s1mb.adapters.rune import rune_messages


def test_certo_preserves_authored_order_structures_and_numeric_levels():
    case = cases()
    state, options = certo_inputs(case.state, case.questions[0])
    assert state == '{"system": "System.", "instruction": "Choose."}\n\n{"evidence": "text"}'
    assert options == ["Same", "Same"]
    _, options = certo_inputs(case.state, case.questions[1])
    assert options == ["Yes", "No"]
    q = case.questions[2]
    q.options.reverse()
    _, options = certo_inputs(case.state, q)
    assert options == [
        '{"value": 30.0, "description": "High"}',
        '{"value": 10.0, "description": "Low"}',
    ]
    assert "gold-must-not-be-sent" not in state + str(options)


def test_rune_native_order_and_structured_score_rubric():
    from s1mb.adapters.firelex_jeff import decision_row

    case = cases()
    noul = decision_row(case.state, case.questions[1])
    messages = rune_messages(noul["state"], noul["question"], ["A", "B"])
    assert messages[1]["content"].endswith("A: No\nB: Yes\nAnswer with one option letter only.")
    question = case.questions[2]
    question.options.reverse()
    score = decision_row(case.state, question)
    rendered = rune_messages(score["state"], score["question"], ["A", "B"])[1]["content"]
    assert 'A: {"value": 30.0, "description": "High"}' in rendered
    assert 'B: {"value": 10.0, "description": "Low"}' in rendered
    assert "gold-must-not-be-sent" not in rendered
    assert '"evidence": "text"' in rendered


def test_jevone_checks_both_option_orders_before_inference():
    adapter = JevOneAdapter.__new__(JevOneAdapter)
    adapter.context_limit = 2
    adapter.tokenizer = cast(Any, SimpleNamespace(encode=lambda text, **kw: list(text)))
    adapter.engine = lambda **kw: pytest.fail("Must validate every prompt before forwarding")
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.forward("/generate", {"text": ["ab", "abc"], "token_ids_logprob": [[1, 2], [1, 2]]})


def test_needle_checks_complete_tool_call_without_dropping_candidates():
    adapter = NeedleAdapter.__new__(NeedleAdapter)
    adapter.context_limit = 4
    adapter.bos, adapter.eos = 2, 1
    adapter.tokenizer = cast(Any, SimpleNamespace(encode=lambda text: list(range(len(text)))))
    adapter.renderer = cast(Any, SimpleNamespace(render_example=lambda row: ("abc", "def")))
    adapter.likelihood = lambda *args: pytest.fail("Must reject intact candidate before forwarding")
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_omni_duplicate_descriptions_stay_distinct_without_private_ids():
    case = cases()
    state, instruction, options = omni_inputs(case.state, case.questions[0])
    assert state == '{"evidence": "text"}'
    assert instruction == '{"system": "System.", "instruction": "Choose."}'
    assert options == ["1. option_0: Same", "2. option_1: Same"]
    assert omni_inputs(case.state, case.questions[1])[2] == ["1. true: Yes", "2. false: No"]
    assert omni_inputs(case.state, case.questions[2])[2] == ["1. 10.0: Low", "2. 30.0: High"]


def test_extended_labels_preserve_native_prefix_and_reject_ambiguous_tokens():
    vocab = {" A": [0], " B": [1], " AA": [0], " AB": [2, 3], " AC": [4]}
    tok = SimpleNamespace(encode=lambda text, **kwargs: vocab.get(text, [7, 8]))
    assert extended_labels(tok, [" A", " B"], 3) == [" A", " B", " AC"]
    with pytest.raises(ValueError, match="insufficient"):
        extended_labels(tok, [" A", " B"], 4)
    with pytest.raises(ValueError, match="distinct single-token"):
        extended_labels(tok, [" A", " AA"], 3)


def test_jevlite_maps_native_yes_no_order_and_preserves_failures():
    adapter = JevLiteAdapter.__new__(JevLiteAdapter)
    adapter.context_limit = 5
    adapter.prompt = cast(
        Any, SimpleNamespace(labels_for=lambda q: None, render=lambda *a: "prompt")
    )
    adapter.torch = cast(Any, SimpleNamespace(inference_mode=nullcontext))
    adapter.engine = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1, 2]),
        distributions=lambda rows: ([[0.2, 0.8]], 2),
    )
    result = adapter.predict(cases())
    assert result[1].probabilities == {"true": 0.2, "false": 0.8}
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}
    adapter.context_limit = 1
    adapter.engine.distributions = lambda *a: pytest.fail("Must reject before inference")
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_certo_rejects_overflow_before_tensor_creation():
    adapter = CertoAdapter.__new__(CertoAdapter)
    adapter.context_limit = 1
    adapter.tokenizer = lambda *a, **kw: {"input_ids": [[1, 2]]}
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_omni_maps_unique_output_keys_and_rejects_overflow():
    adapter = JevOmniAdapter.__new__(JevOmniAdapter)
    adapter.context_limit = 10
    adapter.native = cast(Any, SimpleNamespace(_prompt=lambda *a: "prompt"))
    adapter.engine = SimpleNamespace(
        processor=SimpleNamespace(
            apply_chat_template=lambda *a, **kw: {
                "input_ids": SimpleNamespace(shape=(1, 3)),
            }
        ),
        predict=lambda **kw: {"probabilities": dict(zip(kw["options"], [0.2, 0.8]))},
    )
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities == {"true": 0.2, "false": 0.8}
    adapter.engine.predict = lambda **kw: {"probabilities": {"unexpected": 1}}
    with pytest.raises(ValueError, match="unexpected option keys"):
        adapter.predict(cases())
    adapter.context_limit = 1
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_gliner_preserves_labels_and_checks_complete_distribution():
    from s1mb.adapters.gliner2 import Gliner2Adapter, gliner_inputs

    case = cases()
    text, labels = gliner_inputs(case.state, case.questions[0])
    assert labels == {"option_0": "Same", "option_1": "Same"}
    assert "System." in text and "gold-must-not-be-sent" not in text
    assert list(gliner_inputs(case.state, case.questions[2])[1]) == [
        "level_0: 10.0",
        "level_1: 30.0",
    ]
    adapter = Gliner2Adapter.__new__(Gliner2Adapter)
    adapter.context_limit = 10
    adapter.torch = cast(Any, SimpleNamespace(inference_mode=nullcontext))
    adapter.engine = SimpleNamespace(
        create_schema=lambda: SimpleNamespace(classification=lambda name, labels, **kw: labels),
        processor=SimpleNamespace(
            transform_record=lambda *a, **kw: SimpleNamespace(input_ids=[1, 2])
        ),
        extract=lambda text, labels, **kw: {
            "decision": [
                {"label": key, "confidence": value} for key, value in zip(labels, [0.2, 0.8])
            ]
        },
    )
    assert adapter.predict(case)[1].probabilities == {"true": 0.2, "false": 0.8}
    adapter.engine.extract = lambda *a, **kw: {"decision": [{"label": "true", "confidence": 1.0}]}
    with pytest.raises(ValueError):
        adapter.predict(case)
    adapter.context_limit = 1
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(case)


def test_flymy_maps_anonymous_keys_and_rejects_missing_probabilities():
    from s1mb.adapters.flymy import FlymyAdapter

    adapter = FlymyAdapter.__new__(FlymyAdapter)

    def decide(state, question):
        keys = (
            ["false", "true"]
            if question["type"] == "noul"
            else ["0", "1"]
            if question["type"] == "score"
            else list(question["criteria"])
        )
        return {"probabilities": dict(zip(keys, [0.3, 0.7], strict=True))}

    adapter.engine = SimpleNamespace(decide=decide)
    results = adapter.predict(cases())
    assert results[1].probabilities == {"false": 0.3, "true": 0.7}
    assert results[2].probabilities == {"low": 0.3, "high": 0.7}
    adapter.engine.decide = lambda *a: {"probabilities": {"option_0": 1.0}}
    with pytest.raises(ValueError):
        adapter.predict(cases())


def test_flymy_request_bound_accepts_long_text_but_preserves_validation():
    from s1mb.adapters.flymy import bounded_request_json

    assert bounded_request_json({"state": "a" * 70000}) > 65536
    with pytest.raises(ValueError, match="1 MiB"):
        bounded_request_json({"state": "日" * 400000})
    with pytest.raises(ValueError, match="structure"):
        bounded_request_json([0] * 4097)
    deep = []
    for _ in range(25):
        deep = [deep]
    with pytest.raises(ValueError, match="structure"):
        bounded_request_json(deep)
    with pytest.raises(TypeError, match="keys"):
        bounded_request_json({1: "invalid"})
    with pytest.raises(TypeError, match="JSON values"):
        bounded_request_json({"value": object()})
    with pytest.raises(ValueError):
        bounded_request_json({"value": float("nan")})


def test_lev_rejects_complete_input_overflow_before_forward():
    from s1mb.adapters.lev import LevAdapter

    adapter = LevAdapter.__new__(LevAdapter)
    adapter.context_limit = 1
    adapter.tokenizer = None
    adapter.api = cast(
        Any,
        SimpleNamespace(
            SystemOneRequest=lambda **kw: kw,
            to_record=lambda req: (req, None),
        ),
    )
    adapter.engine = SimpleNamespace(encode=lambda *a, **kw: {"ids": [1, 2]})
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_opendecision_never_honors_native_truncation_requests():
    from s1mb.adapters.opendecision import StrictTokenizer

    calls = []

    def tokenize(*args, **kwargs):
        calls.append(kwargs)
        return {"input_ids": [[1, 2, 3], [1, 2]]}

    tokenizer = StrictTokenizer(tokenize, 3)
    assert tokenizer("premise", "hypothesis", truncation="only_first", max_length=1)["input_ids"][
        0
    ] == [1, 2, 3]
    assert calls == [{"truncation": False}]
    tokenizer.limit = 2
    with pytest.raises(ValueError, match="refusing truncation"):
        tokenizer("premise", truncation=True)


def test_reranker_full_rubric_keeps_numeric_levels_and_authored_definitions():
    from s1mb.adapters.reranker import reranker_inputs

    case = cases()
    instruction, query, documents = reranker_inputs(case.state, case.questions[0])
    assert "System." in instruction and "Choose." in instruction
    assert documents == ["option_0: Same", "option_1: Same"]
    assert all(document in query for document in documents)
    assert "gold-must-not-be-sent" not in query
    assert reranker_inputs(case.state, case.questions[1])[2] == ["true: Yes", "false: No"]
    case.questions[2].options.reverse()
    assert reranker_inputs(case.state, case.questions[2])[2] == ["30.0: High", "10.0: Low"]


def test_reranker_rejects_late_candidate_overflow_before_any_forward():
    from s1mb.adapters.reranker import RerankerAdapter

    adapter = RerankerAdapter.__new__(RerankerAdapter)
    adapter.causal = False
    adapter.context_limit = 4
    calls = []

    def tokenize(query, document, **kwargs):
        calls.append((document, kwargs))
        return {"input_ids": [1] * (2 if len(calls) == 1 else 5)}

    adapter.tokenizer = tokenize
    adapter.torch = cast(
        Any,
        SimpleNamespace(
            inference_mode=lambda: pytest.fail("All candidates must be checked before inference")
        ),
    )
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())
    assert len(calls) == 2
    assert all(kwargs == {"truncation": False} for _, kwargs in calls)


def test_opendecision_retains_instructions_when_native_noul_ignores_them():
    from s1mb.adapters.opendecision import OpenDecisionAdapter

    seen = []
    adapter = OpenDecisionAdapter.__new__(OpenDecisionAdapter)

    def noul(**kwargs):
        seen.append(kwargs)
        return {"noul": 0.3}

    adapter.engine = SimpleNamespace(noul=noul)
    case = cases()
    case.questions = [case.questions[1]]
    prediction = adapter.predict(case)[0]
    assert prediction.probabilities == {"true": 0.3, "false": 0.7}
    assert seen[0]["instructions"] in seen[0]["criteria"]["true"]
    assert seen[0]["criteria"]["true"].endswith("true: Yes")
    assert seen[0]["criteria"]["false"].endswith("false: No")
