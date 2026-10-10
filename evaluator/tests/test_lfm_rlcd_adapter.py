"""Native schema preservation and complete branch limits without downloaded models."""

import json
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.lfm_rlcd import LFMRLCDAdapter, rlcd_branches, rlcd_inputs


def test_structured_fields_numeric_levels_declared_order_and_no_ids():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    rendered = [rlcd_inputs(case.state, q) for q in case.questions]
    assert rendered[0][2] == ['option 0: {"nested": true}', "option 1: Same"]
    assert rendered[1][2] == ["option 0: Yes", "option 1: No"]
    assert rendered[2][2] == ["level 0: 30.0: High", "level 1: 10.0: Low"]
    assert json.loads(rendered[0][1]["properties"]["decision"]["description"]) == {
        "system": "System.",
        "instruction": "Choose.",
    }
    assert "gold-must-not-be-sent" not in str(rendered)
    assert rendered[0][1]["additionalProperties"] is False


def test_complete_candidate_path_checked_not_just_prefill():
    engine = SimpleNamespace(prompt=lambda context, schema: "abcd", encode=lambda text: list(text))
    schema = {"properties": {"decision": {"enum": ["x", "longer"]}}}
    prefix, branches = rlcd_branches(engine, "", schema, 100)
    assert prefix == list("abcd")
    total = len(prefix) + max(len(branch[0]) for branch in branches)
    rlcd_branches(engine, "", schema, total)
    with pytest.raises(ValueError, match="refusing truncation"):
        rlcd_branches(engine, "", schema, total - 1)


def test_malformed_branch_count_rejected_before_distribution(monkeypatch):
    adapter = LFMRLCDAdapter.__new__(LFMRLCDAdapter)
    monkeypatch.setattr(adapter, "branch_scores", lambda context, schema: [0.0])
    with pytest.raises(ValueError, match="candidate count"):
        adapter.predict(cases())


def test_invalid_context_before_download():
    with pytest.raises(ValueError, match="positive"):
        LFMRLCDAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
