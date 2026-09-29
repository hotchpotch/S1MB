"""Von's native calibrated typed decisions, with explicit context admission."""

import importlib
import json

from s1mb.data import InferenceCase, Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


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
    case_batch_size = 16

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
        self.engine.device = self.torch.device(device)
        self.format_state = module._format_state
        self.settings = {
            "dtype": "float32",
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "renderer": "native-independent-option-masks-batched-v2",
            "max_input_tokens": self.context_limit,
            "checkpoint_context_length": 8192,
            "extended_context": self.context_limit > 8192,
            "candidate_markers": "inserted delimiters identified by tokenizer offsets",
            "context_overflow": "fail; never silently truncate",
            "calibration": json.loads((self.path / "marker_calibration.json").read_text()),
        }

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        network = self.engine._get_model()
        rows = []
        for index, case in enumerate(cases):
            state = self.format_state(case.state)
            for q in case.questions:
                definition = questions_for_api([q])[q.id]
                options = q.options
                if q.task == "noul":
                    options = sorted(options, key=lambda option: option.id != "true")
                ids, positions = candidate_marker_positions(
                    network, state, definition["instructions"], [o.description for o in options]
                )
                if len(ids) > self.context_limit:
                    raise ValueError("Von input exceeds context limit; refusing truncation")
                rows.append((index, case.case_id, q, state, options, ids, positions))
        rows.sort(key=lambda row: len(row[5]))
        outputs = [{} for _ in cases]
        with self.torch.inference_mode():
            for group in candidate_batches(rows, [len(row[5]) for row in rows], 8, 4096):
                batch = network.tokenizer.pad(
                    {"input_ids": [row[5] for row in group]}, padding=True, return_tensors="pt"
                ).to(self.device)
                logits = network(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    mask_positions=[row[6] for row in group],
                    independent_options=self.engine._independent_options,
                )
                for row, values in zip(group, logits, strict=True):
                    index, case_id, q, state, options, _, _ = row
                    temperature = self.engine._effective_temperature(
                        values, state, len(options), network.tokenizer, None
                    )
                    p = (values.float() / max(temperature, 1e-4)).softmax(-1).cpu().tolist()
                    outputs[index][q.id] = Prediction(
                        case_id=case_id,
                        question_id=q.id,
                        probabilities=dict(zip([o.id for o in options], p, strict=True)),
                    )
        return [
            [out[q.id] for q in case.questions] for out, case in zip(outputs, cases, strict=True)
        ]
