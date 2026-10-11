"""Generate local target-free reasoning contexts for the MetaEncoder-think adapter.

The output contains model generations and is an evaluation intermediate, not a
source- or results-repository artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

from s1mb.adapters.meta_encoder import render_query
from s1mb.adapters.meta_encoder_think import (
    REASONER_ID,
    REASONER_REVISION,
    parse_reasoning_generation,
    reasoner_prompt,
)
from s1mb.data import load_benchmark, load_cases, load_category
from s1mb.parameters import parameter_metadata

MAX_NEW_TOKENS = 1024
CONTEXT_LIMIT = 32768
SEED = 4242


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="english-v1")
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reasoner", default=REASONER_ID)
    parser.add_argument("--reasoner-revision", default=REASONER_REVISION)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    modeling = importlib.import_module(
        "transformers.models.muse_glimmer.modeling_muse_glimmer"
    )

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("reasoning generation requires explicit CUDA")
    torch.cuda.set_device(args.device)
    torch.manual_seed(SEED)
    processor = transformers.AutoProcessor.from_pretrained(
        args.reasoner, revision=args.reasoner_revision
    )
    processor.tokenizer.padding_side = "left"
    model = modeling.MuseGlimmerForConditionalGeneration.from_pretrained(
        args.reasoner,
        revision=args.reasoner_revision,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()

    category = load_category(args.data_dir, args.category)
    selected = args.benchmark or category.benchmarks
    if set(selected) - set(category.benchmarks):
        parser.error("selected benchmark is outside the category")
    records = []
    by_query: dict[str, dict] = {}
    for benchmark_id in selected:
        benchmark = load_benchmark(args.data_dir, benchmark_id)
        for case in load_cases(args.data_dir, benchmark):
            inference = case.inference()
            for question in inference.questions:
                query = render_query(inference, question)
                query_sha = hashlib.sha256(query.encode()).hexdigest()
                if query_sha in by_query:
                    source = by_query[query_sha]
                    records.append(
                        {
                            **source,
                            "benchmark_id": benchmark_id,
                            "case_id": inference.case_id,
                            "question_id": question.id,
                            "reused": True,
                        }
                    )
                    continue
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": reasoner_prompt(query)}],
                    }
                ]
                chat = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = processor(
                    text=[chat],
                    images=None,
                    videos=None,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                input_tokens = int(inputs["input_ids"].shape[1])
                if input_tokens + MAX_NEW_TOKENS > CONTEXT_LIMIT:
                    raise ValueError(f"reasoner input exceeds context limit: {benchmark_id}")
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                raw = processor.tokenizer.decode(
                    output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                if not raw:
                    raise ValueError(f"empty reasoner generation: {benchmark_id}")
                record = {
                    "benchmark_id": benchmark_id,
                    "case_id": inference.case_id,
                    "question_id": question.id,
                    "query_sha256": query_sha,
                    "raw_generation": raw,
                    **parse_reasoning_generation(raw),
                    "input_tokens": input_tokens,
                    "output_tokens": int(output.shape[1] - inputs["input_ids"].shape[1]),
                    "reused": False,
                }
                records.append(record)
                by_query[query_sha] = record

    payload = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "reasoner": {
            "id": args.reasoner,
            "revision": args.reasoner_revision,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "generation": {
                "do_sample": False,
                "seed": SEED,
                "max_new_tokens": MAX_NEW_TOKENS,
                "context_limit": CONTEXT_LIMIT,
            },
            "prompt_policy": "target-free-official-query-reasoning-then-final-answer-v1",
            "truncation_policy": "earliest-final-answer-marker-or-answer-like-conclusion-v1",
            "parameters": parameter_metadata(model),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} contexts ({len(by_query)} distinct queries) to {args.output}")


if __name__ == "__main__":
    main()
