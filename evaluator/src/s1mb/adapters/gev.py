"""GEV's native BF16/FP32 bare-v1 head and all-candidate tournament."""

import ast
import asyncio
import importlib
import json
import math
import zlib
from pathlib import Path
from types import SimpleNamespace

from s1mb.data import Prediction, check_probabilities

from .autotrust_jev import autotrust_choice_text
from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text


def gev_tournament(source, reader):
    """Load the released pure orchestration functions without launching vLLM."""
    tree = ast.parse((Path(source) / "serve_decide.py").read_text())
    nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in {"_groups", "s1_dist"}]
    if len(nodes) != 2:
        raise ValueError("GEV source lacks its native tournament functions")
    scope = {"asyncio": asyncio, "math": math, "zlib": zlib, "s1_pass": reader}
    # The caller explicitly selects the upstream source; retain its definitions verbatim.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native-gev-tournament", "exec"), scope)  # noqa: S102
    return SimpleNamespace(groups=scope["_groups"], distribution=scope["s1_dist"])


class GEVAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("gev", model, revision, source, device)
        self.limit = 32768 if context_limit is None else context_limit
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        config = json.loads((self.path / "judge_config.json").read_text())
        if (config["slots"]["template_version"] != "bare-v1"
                or config["slots"]["ranges"]["choice"] != [8, 24]
                or config["softcap"] != 30.0):
            raise ValueError("Unsupported GEV released head configuration")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        if self.tokenizer is None:
            raise ValueError("Checkpoint did not provide a supported tokenizer")
        if self.tokenizer.bos_token_id != config["readout"]["prefix_token"]:
            raise ValueError("GEV tokenizer BOS differs from its trained readout")
        base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            str(self.path), dtype=self.torch.bfloat16, device_map=device,
            attn_implementation="sdpa",
        )
        self.engine = peft.PeftModel.from_pretrained(
            base, str(self.path / config["adapter_subfolder"]),
        ).merge_and_unload().eval()
        self.attention_model = self.engine
        capacity = self.engine.config.get_text_config().max_position_embeddings
        if not 1 <= self.limit <= capacity:
            raise ValueError("GEV context limit exceeds checkpoint capacity")
        loader = importlib.import_module("safetensors.torch")
        head = loader.load_file(str(self.path / "head.safetensors"))
        self.weight = head["proj.weight"].to(device=device, dtype=self.torch.float32)
        self.bias = head["proj.bias"].to(device=device, dtype=self.torch.float32)
        if self.weight.shape != (24, config["hidden_size"]) or self.bias.shape != (24,):
            raise ValueError("GEV released head dimensions are inconsistent")
        self.engine.register_parameter("s1mb_slot_weight", self.torch.nn.Parameter(self.weight, requires_grad=False))
        self.engine.register_parameter("s1mb_slot_bias", self.torch.nn.Parameter(self.bias, requires_grad=False))
        self.temperature = json.loads((self.path / "calibration.json").read_text())["per_kind"]["choice"]
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid GEV native calibration temperature")
        self.native = gev_tournament(self.source, self._read)
        self.settings.update({
            "dtype": "bfloat16", "head_dtype": "float32", "attention_implementation": "sdpa",
            "max_input_tokens": self.limit, "checkpoint_position_limit": capacity,
            "input_length_policy": "reject-overflow-every-native-branch",
            "temperature": self.temperature, "head_softcap": 30.0,
            "choice_strategy": "native-tournament", "max_candidates": 256,
            "thinking": False, "runtime": "released-transformers-bf16-lora-merge-fp32-head",
            "renderer": "bare-v1-bos-choice-authored-noul-numeric-score",
        })

    async def _read(self, ctx, kind, state, has_image, instructions, options):
        if kind != "choice" or has_image:
            raise ValueError("GEV benchmark bridge requires text-only explicit Choice criteria")
        text = autotrust_choice_text(state, instructions, dict(enumerate(options)))
        ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.limit:
            raise ValueError("Complete GEV branch exceeds context limit")
        with self.torch.inference_mode():
            hidden = self.engine.model(
                input_ids=self.torch.tensor([ids], device=self.device), use_cache=False,
            ).last_hidden_state[0, -1].float()
            logits = 30.0 * self.torch.tanh((self.weight @ hidden + self.bias) / 30.0)
            return self.torch.softmax(logits[8:8 + len(options)] / self.temperature, 0).tolist()

    def predict(self, case):
        predictions = []
        wire = sifr_questions(case)
        for question in case.questions:
            spec = wire[question.id]
            options = [state_text(value) for value in spec["criteria"].values()]
            if not 2 <= len(options) <= 256:
                raise ValueError("GEV native tournament requires 2..256 candidates")
            values = asyncio.run(self.native.distribution(
                None, "choice", state_text(case.state), False,
                state_text(spec["instructions"]), options, "tournament",
            ))
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions

    def close(self):
        del self.weight, self.bias
        super().close()
