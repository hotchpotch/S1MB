"""Selected-token probabilities must be complete and aligned."""

import pytest

from s1mb.adapters.blink import blink_probabilities


def test_omitted_candidate_is_rejected():
    with pytest.raises(ValueError, match="omitted requested candidate"):
        blink_probabilities([{"token": "token_id:1", "logprob": -0.1}], [1, 2], 1.3)


def test_selected_logprobs_preserve_order_and_calibration():
    rows = [{"token": "token_id:9", "logprob": -1.3},
            {"token": "token_id:4", "logprob": -2.6},
            {"token": "token_id:2", "logprob": -0.1}]
    assert blink_probabilities(rows, [4, 9], 1.3) == pytest.approx(
        [0.2689414213699951, 0.7310585786300049],
    )


@pytest.mark.parametrize("omit_last", [False, True])
def test_wide_candidates_share_one_normalization(monkeypatch, omit_last):
    import io
    import json
    import string
    import urllib.request

    from s1mb.adapters.blink import BlinkAdapter
    from s1mb.data import InferenceCase

    adapter = object.__new__(BlinkAdapter)
    adapter.codes = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
    ]
    tokens = {code: i + 1 for i, code in enumerate(adapter.codes)}

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt:"

        def encode(self, text, **kwargs):
            return [0] if text == "prompt:" else [0, tokens[text.removeprefix("prompt:")]]

    adapter.tokenizer = Tokenizer()
    adapter.limit = 32768
    adapter.temperature = 1.3
    adapter.url = "http://localhost:18092"
    requested = []

    def respond(request, **kwargs):
        body = json.loads(request.data)
        ids = body["logprob_token_ids"]
        assert len(ids) <= 128
        requested.extend(ids)
        rows = [{"token": f"token_id:{token}", "logprob": -token / 100}
                for token in reversed(ids) if not (omit_last and token == 151)]
        # The sampled token may appear in every chunk, even when not requested.
        if 1 not in ids:
            rows.append({"token": "token_id:1", "logprob": -0.01})
        return io.BytesIO(json.dumps({"usage": {"prompt_tokens": 1}, "choices": [
            {"logprobs": {"content": [{"top_logprobs": rows}]}}
        ]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    case = InferenceCase.model_validate({"case_id": "wide", "state": "Choose route 150.",
        "questions": [{"id": "routes", "task": "choice", "instructions": "Choose a route.",
                       "options": [{"id": str(i), "description": f"Route {i}"}
                                   for i in range(151)]}]})
    if omit_last:
        with pytest.raises(ValueError, match="omitted requested candidate"):
            adapter.predict(case)
    else:
        actual = adapter.predict(case)[0].probabilities
        expected = blink_probabilities(
            [{"token": f"token_id:{i}", "logprob": -i / 100} for i in range(1, 152)],
            list(range(1, 152)), 1.3,
        )
        assert list(actual.values()) == pytest.approx(expected)
    assert requested == list(range(1, 152))
