"""JEVision's released visual sidecar and native pointer readout."""

import importlib
import json
import sys

from s1mb.data import Prediction

from .upstream import UpstreamAdapter
from .vision_native import choice_transport, ordered_images, typed_questions


def pointer_positions(token_ids, delimiters, option_count):
    """Validate one complete native pointer record before selecting hidden states."""
    if not 1 <= option_count <= 255:
        raise ValueError("JEVision requires 1..255 options")
    counts = [token_ids.count(token) for token in delimiters]
    if counts != [1, 1, option_count, option_count, 1]:
        raise ValueError("JEVision text delimiter counts differ from the request")
    state, question, opening, closing, decision = delimiters
    ends = [i for i, token in enumerate(token_ids) if token == closing]
    starts = [i for i, token in enumerate(token_ids) if token == opening]
    qpos, dpos = token_ids.index(question), token_ids.index(decision)
    if not token_ids.index(state) < qpos < starts[0] or not all(
        start < end < (starts[i + 1] if i + 1 < option_count else dpos)
        for i, (start, end) in enumerate(zip(starts, ends, strict=True))
    ):
        raise ValueError("JEVision text pointer boundaries differ from the request")
    return dpos, ends


class JEVisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        self.setup("jevvision", model, revision, source, device)
        sys.path.insert(0, str(self.path / "runtime"))
        native = importlib.import_module("kev.visual_model")
        self.engine = native.VisualDecisionModel.from_run(
            self.path, device=device, max_input_tokens=context_limit
        )
        self.prompt = importlib.import_module("kev.visual_prompt")
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.enable_kernels()
        self.image_receipts = {}
        report = json.loads((self.path / "report.json").read_text())
        self.settings.update({
            "dtype": "float16-float32-head", "max_input_tokens": context_limit,
            "input_length_policy": "reject-overflow-native", "media": "native-ordered-pil-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "renderer": "native-pointer-authored-noul-numeric-score-v1",
            "base_model": report["model"], "base_revision": report["revision"],
            "route": "released-visual-sidecar-with-text-only-pointer-v1", "max_images": 4,
            "text_transport": "native-chat-delimiters-without-image-tokens-v1",
            "calibration_policy": "native-pointer-softmax",
        })

    def text_probabilities(self, record):
        """Use the released visual weights and pointer head without fabricated media."""
        tok = self.engine.processor.tokenizer
        special = self.prompt.SPECIAL
        delimiters = [tok.convert_tokens_to_ids(token) for token in special]
        results = []
        for question in record["questions"]:
            options = question["options"]
            if not 1 <= len(options) <= 255:
                raise ValueError("JEVision requires 1..255 options")
            escape = self.prompt.escape_user_text
            s, q, o, c, d = special
            body = s + escape(record["state"]) + q + escape(question["instr"])
            body += "".join(o + escape(option) + c for option in options) + d
            text = tok.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": body}]}],
                tokenize=False, add_generation_prompt=False,
            )
            batch = self.engine.processor(text=[text], return_tensors="pt", padding=True)
            if batch["input_ids"].shape[1] > self.engine.max_input_tokens:
                raise ValueError("JEVision text input exceeds context limit")
            decision, ends = pointer_positions(batch["input_ids"][0].tolist(), delimiters, len(options))
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.engine.lock, self.torch.inference_mode(), self.engine._autocast():
                backbone = self.engine._backbone()
                backbone.rope_deltas = None
                h = backbone(**batch, use_cache=False).last_hidden_state[0].float()
                logits = self.engine.head(h[decision], h[ends])
                if not self.torch.isfinite(logits).all():
                    raise ValueError("JEVision text pointer logits are not finite")
                results.append(logits.float().softmax(-1).cpu())
        return results

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        predictions = []
        for key, question in typed_questions(case).items():
            record = {"state": json.dumps(state if images else case.state, ensure_ascii=False), "questions": [{
                "instr": question["instructions"], "options": choice_transport(question),
            }]}
            values = self.engine.probs(record, images) if images else self.text_probabilities(record)
            if len(values) != 1:
                raise ValueError("JEVision returned incorrect answer count")
            q = next(q for q in case.questions if q.id == key)
            keys = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
            predictions.append(Prediction(
                case_id=case.case_id, question_id=key,
                probabilities=dict(zip(keys, values[0].float().cpu().tolist(), strict=True)),
            ))
        return predictions
