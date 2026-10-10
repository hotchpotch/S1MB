"""Current Lev release API with pinned base and strict native input budgets."""

import importlib
import json
import sys

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, checkpoint_path


class LevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, base_revision=None):
        if not base_revision:
            raise ValueError("Lev requires an explicit base checkpoint revision")
        if context_limit is not None and context_limit < 1:
            raise ValueError("Lev context_limit must be positive")
        if sys.version_info < (3, 12):
            raise RuntimeError("The pinned Lev runtime requires Python 3.12 or newer")
        self.setup("lev", model, revision, source, device)
        self.native = importlib.import_module("lev.model")
        manifest = json.loads((self.path / "lev_release.json").read_text())
        base, resolved = checkpoint_path(manifest["base_model"], base_revision)
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        backbone = transformers.AutoModelForCausalLM.from_pretrained(
            base, dtype=self.torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        peft = importlib.import_module("peft")
        backbone = peft.PeftModel.from_pretrained(backbone, str(self.path)).merge_and_unload().eval()
        self.context_limit = context_limit or 32768
        if self.context_limit > backbone.config.get_text_config().max_position_embeddings:
            raise ValueError("Lev context limit exceeds checkpoint capacity")
        head = self.native._load_head(self.path / "mode_b_head.pt", backbone)
        if head is None:
            raise ValueError("Lev release is missing its native Mode B head")
        calibration = self.native.CalibrationProfile.load(self.path / "calibration.json")
        config = self.native.EngineConfig(
            model_id=manifest["base_model"], prompt_style=manifest["prompt_style"],
            noul_readout=manifest["noul_readout"], compile=False, skip_multi_token_codes=True,
            score_order_average="off", max_batch_tokens=16384,
        )
        self.engine = self.native.DecisionEngine(backbone, tokenizer, config, calibration, head)
        self.engine.checkpoint = self.path
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "base_model": manifest["base_model"], "base_revision": resolved,
            "max_input_tokens": self.context_limit, "candidate_text_limit": 64,
            "input_length_policy": "reject-overflow-before-native-truncation",
            "runtime": "native-label-or-candidate-path-calibrated-merged-lora",
            "renderer": "native-choice-authored-noul-numeric-score",
            "questions_per_call": 1,
        })

    def predict(self, case):
        predictions = []
        questions = sifr_questions(case)
        for question in case.questions:
            self.engine._candidate_cache.clear()
            prepared = self.engine.prepare(case.state, {"decision": questions[question.id]})
            if prepared.width > self.context_limit:
                raise ValueError("Lev input exceeds context limit; refusing truncation")
            native_question = prepared.questions["decision"]
            if prepared.routes["decision"].mode is self.native.Mode.CANDIDATE_PATH:
                texts = self.native.candidate_texts(native_question)
                if any(len(self.engine.tokenizer.encode(text, add_special_tokens=False)) > 64
                       for text in texts):
                    raise ValueError("Lev candidate text exceeds native 64-token budget")
            response = self.engine.answer([prepared])[0]
            if set(response.answers) != {"decision"}:
                raise ValueError("Lev returned incorrect answer count")
            probabilities = response.answers["decision"].probabilities
            keys = ([o.id for o in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={o.id: probabilities[key] for o, key in
                               zip(question.options, keys, strict=True)},
            ))
        self.engine._candidate_cache.clear()
        return predictions

    def close(self):
        if hasattr(self, "engine"):
            self.engine._candidate_cache.clear()
        super().close()
