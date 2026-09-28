"""Von's native calibrated typed decisions, with explicit context admission."""

import importlib
import json

from s1mb.data import InferenceCase, Prediction

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def candidate_marker_positions(network, state, instructions, options):
    """Locate inserted markers, allowing literal [MASK] tokens in user content."""
    text = network.pack_sequence(state, instructions, options)
    prefix = network.pack_sequence(state, instructions, [])
    empty_prefix = network.pack_sequence("", "", [])
    cursor = len(prefix)
    starts = set()
    for option in options:
        starts.add(cursor)
        fragment = network.pack_sequence("", "", [option])[len(empty_prefix) :]
        cursor += len(fragment) + 1
    encoded = network.tokenizer(text, return_offsets_mapping=True)
    positions = [
        i
        for i, (token, offset) in enumerate(
            zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
        )
        if token == network.mask_token_id
        and any(offset[0] <= start < offset[1] for start in starts)
    ]
    if len(positions) != len(options):
        raise ValueError("Cannot locate every inserted candidate marker")
    return encoded["input_ids"], positions


class VonAdapter(UpstreamAdapter):
    def __init__(self, model: str, revision: str, source: str, device: str, context_limit=None):
        self.setup("von", model, revision, source, device)
        module = importlib.import_module("von.backends.option_marker_backend")
        # Load weights on the host, then cast before transferring. No CPU inference.
        self.engine = module.OptionMarkerBackend(checkpoint_dir=str(self.path), device="cpu")
        network = self.engine._get_model()
        network.to(device=device, dtype=self.torch.float32)
        self.attention_model = network.encoder
        self.context_limit = context_limit or 8192
        network.tokenizer.model_max_length = self.context_limit
        native_forward = network.forward
        self._marker_positions = None

        def forward_with_candidate_markers(*args, **kwargs):
            if self._marker_positions is None:
                raise RuntimeError("Candidate marker positions were not prepared")
            kwargs["mask_positions"] = [self._marker_positions]
            return native_forward(*args, **kwargs)

        network.forward = forward_with_candidate_markers
        self.engine.device = self.torch.device(device)
        self.format_state = module._format_state
        self.settings = {
            "dtype": "float32",
            "max_input_tokens": self.context_limit,
            "checkpoint_context_length": 8192,
            "extended_context": self.context_limit > 8192,
            "candidate_markers": "inserted delimiters identified by tokenizer offsets",
            "context_overflow": "fail; never silently truncate",
            "calibration": json.loads((self.path / "marker_calibration.json").read_text()),
        }

    def predict(self, case: InferenceCase) -> list[Prediction]:
        answers = {}
        for q in case.questions:
            request = questions_for_api([q])
            definition = request[q.id]
            options = [o.description for o in q.options]
            if q.task == "noul":
                lookup = {o.id: o.description for o in q.options}
                options = [lookup["true"], lookup["false"]]
            network = self.engine._get_model()
            ids, positions = candidate_marker_positions(
                network, self.format_state(case.state), definition["instructions"], options
            )
            if len(ids) > self.context_limit:
                raise ValueError(
                    f"Von input exceeds {self.context_limit} tokens: {len(ids)}; no truncation applied"
                )
            self._marker_positions = positions
            try:
                with self.torch.inference_mode():
                    response = self.engine.evaluate(case.state, request)
            finally:
                self._marker_positions = None
            answers.update({k: v.model_dump() for k, v in response.answers.items()})
        return decode_answers(case, answers, rounded=True)
