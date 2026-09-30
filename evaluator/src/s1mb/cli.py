"""Command-line evaluation and offline validation."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .data import (
    DATA_DIR,
    Result,
    load_benchmark,
    load_cases,
    load_category,
    read_json,
    safe_id,
    validate_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="S1MB fixed-sample System One benchmark")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List benchmarks in a category")
    listing.add_argument("--category", default="english-v1")
    check_data = commands.add_parser("check-data", help="Validate all category inputs")
    check_data.add_argument("--category", default="english-v1")
    validate = commands.add_parser("validate", help="Recompute and validate saved results")
    validate.add_argument("paths", nargs="+", type=Path)
    export = commands.add_parser("export-results", help="Prepare model folders for a results PR")
    export.add_argument("paths", nargs="+", type=Path)
    export.add_argument("--metadata", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    sync = commands.add_parser("sync-results", help="Validate and install a Hub results snapshot")
    sync.add_argument("--repo-id", required=True)
    sync.add_argument("--revision", default="main")
    sync.add_argument("--output", type=Path)
    published = commands.add_parser(
        "validate-results", help="Validate a published model repository"
    )
    published.add_argument("repository", type=Path)
    published.add_argument("--download-datasets", action="store_true")
    run = commands.add_parser("run", help="Evaluate and save one run")
    run.add_argument(
        "--adapter",
        choices=[
            "dummy",
            "laya",
            "typesafe",
            "system-ichi",
            "bekko-v0",
            "von",
            "jevforge",
            "kev",
            "decider",
            "jevk5",
            "minojev",
            "openvons",
            "verdict2",
            "luce",
            "open-jev",
            "alex-openjev",
            "openjev-org",
            "clm",
            "tev",
            "lumma",
            "tinyjev",
            "nimble",
            "bosun",
            "manchego",
            "neohorse",
            "jebadiah",
            "openthai",
            "kotoba",
            "jeff",
            "pngwn",
            "eikos",
            "mini-jev",
            "apus",
            "winnow",
            "openjev-shim",
            "verdict-encoder",
        ],
        required=True,
    )
    run.add_argument("--model")
    run.add_argument("--revision", default="main")
    run.add_argument("--subfolder", help="Alex Openjev checkpoint subfolder")
    run.add_argument("--device", default="cpu")
    run.add_argument(
        "--server-host", default="127.0.0.1", help="Winnow bind IPv4: localhost or Tailscale"
    )
    run.add_argument("--server-port", type=int, default=8091, help="Winnow server port")
    run.add_argument("--dtype", choices=["float32", "bfloat16"], help="Minojev backbone precision")
    run.add_argument(
        "--attention",
        choices=["sdpa", "flash_attention_2"],
        help="Explicit attention backend for measured upstream adapters",
    )
    run.add_argument(
        "--context-limit",
        type=int,
        help="Input limit for supported adapters; Bekko v0 uses adaptive truncation",
    )
    run.add_argument(
        "--query-length",
        type=int,
        help="Ichi/Bekko query token cap including special tokens; v0 truncates adaptively",
    )
    run.add_argument(
        "--document-length",
        type=int,
        help="Ichi/Bekko candidate token cap including special tokens; v0 truncates adaptively",
    )
    run.add_argument("--max-len", type=int, help="Laya total token limit")
    run.add_argument("--head-max-len", type=int, help="Laya question/candidate token budget")
    run.add_argument("--questions-per-call", type=int, help="Laya/System Ichi question batch limit")
    run.add_argument("--disable-autocast-cache", action="store_true", help="Reduce Laya GPU memory")
    run.add_argument(
        "--case-batch-size",
        type=int,
        help="Case window size for Bekko v0, Kev, Open-Jev, CLM or Tev",
    )
    run.add_argument("--compile", action="store_true", help="Compile Bekko v0 tensor execution")
    run.add_argument("--microbatch-tokens", type=int, help="Bekko complete-question work budget")
    run.add_argument(
        "--source", help="Upstream source checkout for System Ichi or open-model adapters"
    )
    run.add_argument("--category", default="english-v1")
    run.add_argument("--benchmark", action="append", help="Select specific category benchmarks")
    run.add_argument(
        "--generalization-only",
        action="store_true",
        help="Select only generalization benchmarks within the category",
    )
    run.add_argument("--task", choices=["choice", "noul", "score"], help="Select one task")
    run.add_argument("--limit", type=int, help="First N cases per benchmark; remains partial")
    run.add_argument("--output", type=Path)
    run.add_argument("--run-id")
    run.add_argument(
        "--offline-dataset",
        action="store_true",
        help="Use installed dataset without checking Hugging Face",
    )
    args = parser.parse_args()
    if args.command == "run":
        from .dataset_source import dataset_session

        with dataset_session(args.data_dir, offline=args.offline_dataset):
            execute(args, parser)
    else:
        execute(args, parser)


def execute(args, parser):
    if args.command in {"export-results", "sync-results", "validate-results"}:
        from .result_repository import export_results, sync_results, validate_repository

        if args.command == "export-results":
            print(export_results(args.data_dir, args.paths, args.metadata, args.output))
        elif args.command == "sync-results":
            sync_results(
                args.data_dir,
                args.repo_id,
                args.revision,
                args.output or args.data_dir / "hub-results",
            )
        else:
            count = validate_repository(
                args.data_dir, args.repository, download=args.download_datasets
            )
            print(f"Validated {count} published results")
        return
    if args.command == "validate":
        from .result_repository import result_files

        files = result_files(args.paths)
        if not files:
            parser.error("No result files found")
        identities = {}
        models = {}
        for path in files:
            result = Result.model_validate(read_json(path))
            validate_result(args.data_dir, result)
            key = (result.run_id, result.benchmark.id)
            payload = result.model_dump()
            if key in identities and identities[key] != payload:
                parser.error(f"Conflicting results: {key}")
            identities[key] = payload
            model_identity = (result.model, result.provenance, result.evaluator_version)
            if result.run_id in models and models[result.run_id] != model_identity:
                parser.error(f"Inconsistent model settings within run: {result.run_id}")
            models[result.run_id] = model_identity
        print(f"Validated {len(files)} files; {len(identities)} unique results")
        return
    category = load_category(args.data_dir, args.category)
    benchmarks = [load_benchmark(args.data_dir, b) for b in category.benchmarks]
    if args.command in {"list", "check-data"}:
        for b in benchmarks:
            if args.command == "check-data":
                load_cases(args.data_dir, b)
            print(f"{b.id}\t{b.task}\t{b.case_count} cases\t{b.decision_count} decisions")
        return
    if args.query_length is not None or args.document_length is not None:
        if args.adapter not in {"system-ichi", "bekko-v0"}:
            parser.error("--query-length/--document-length apply only to Ichi/Bekko v0")
        if args.query_length is not None and args.query_length < 3:
            parser.error("--query-length must be at least 3")
        if args.document_length is not None and args.document_length < 2:
            parser.error("--document-length must be at least 2")
    if args.microbatch_tokens is not None and (
        args.adapter != "bekko-v0" or args.microbatch_tokens < 1
    ):
        parser.error("--microbatch-tokens must be positive and applies only to Bekko")
    if args.case_batch_size is not None and (
        args.adapter not in {"bekko-v0", "kev", "open-jev", "clm", "tev"}
        or args.case_batch_size < 1
    ):
        parser.error(
            "--case-batch-size must be positive and applies only to Bekko v0, Kev, Open-Jev, CLM or Tev"
        )
    if args.compile and args.adapter != "bekko-v0":
        parser.error("--compile applies only to Bekko v0")
    if (
        args.adapter == "bekko-v0"
        and args.device == "cpu"
        and (args.category != "smoke-v1" or args.limit is None or not 1 <= args.limit <= 2)
    ):
        parser.error("Bekko v0 CPU requires --category smoke-v1 and --limit 1 or 2")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(value is not None and value <= 0 for value in [args.max_len, args.head_max_len]):
        parser.error("Token limits must be positive")
    if args.questions_per_call is not None and args.questions_per_call < 1:
        parser.error("--questions-per-call must be positive")
    if args.adapter not in {"laya", "system-ichi"} and args.questions_per_call is not None:
        parser.error("--questions-per-call applies only to Laya/System Ichi")
    if args.adapter != "laya" and args.disable_autocast_cache:
        parser.error("--disable-autocast-cache applies only to Laya")
    if args.adapter != "laya" and (args.max_len is not None or args.head_max_len is not None):
        parser.error("Token limit options apply only to the Laya adapter")
    if args.subfolder is not None and args.adapter != "alex-openjev":
        parser.error("--subfolder applies only to alex-openjev")
    if args.dtype is not None and args.adapter != "minojev":
        parser.error("--dtype applies only to the Minojev adapter")
    if args.context_limit is not None and (
        args.context_limit < 1
        or args.adapter
        not in {
            "von",
            "jevforge",
            "kev",
            "open-jev",
            "alex-openjev",
            "openjev-org",
            "clm",
            "tev",
            "bekko-v0",
        }
    ):
        parser.error(
            "--context-limit must be positive and applies only to supported upstream adapters"
        )
    if args.attention and args.adapter not in {
        "von",
        "jevforge",
        "kev",
        "decider",
        "jevk5",
        "minojev",
        "luce",
        "open-jev",
        "alex-openjev",
        "openjev-org",
        "clm",
        "tev",
    }:
        parser.error("--attention applies only to measured upstream adapters")
    if args.benchmark:
        if set(args.benchmark) - set(category.benchmarks):
            parser.error("Selected benchmark is not in the category")
        benchmarks = [b for b in benchmarks if b.id in args.benchmark]
    if args.generalization_only:
        benchmarks = [
            b for b in benchmarks if b.dataset.startswith("datasets/s1mb-generalization-")
        ]
    if args.task:
        benchmarks = [b for b in benchmarks if b.task == args.task]
    if not benchmarks:
        parser.error("No benchmarks match the selection")
    run_id = safe_id(args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    output = args.output or args.data_dir / "results"
    if (output / run_id).exists():
        parser.error("Run directory already exists; choose a new run ID")
    if args.adapter != "dummy" and not args.model:
        parser.error("--model is required for real adapters")
    from .adapters.base import ModelAdapter

    adapter: ModelAdapter
    if args.adapter == "dummy":
        from .adapters.dummy import DummyAdapter

        adapter = DummyAdapter()
    elif args.adapter == "laya":
        from .adapters.laya import LayaAdapter

        adapter = LayaAdapter(
            args.model,
            args.revision,
            args.device,
            args.max_len,
            args.head_max_len,
            args.questions_per_call,
            not args.disable_autocast_cache,
        )
    elif args.adapter == "bekko-v0":
        from .adapters.bekko_v0 import BekkoV0Adapter

        adapter = BekkoV0Adapter(
            args.model,
            args.device,
            args.query_length,
            args.document_length,
            token_budget=args.microbatch_tokens if args.microbatch_tokens is not None else 64_000,
            case_batch_size=args.case_batch_size if args.case_batch_size is not None else 128,
            context_length=args.context_limit,
            revision=args.revision,
            compile_model=args.compile,
            cpu_smoke=args.device == "cpu",
        )
    elif args.adapter == "typesafe":
        from .adapters.typesafe import TypeSafeAdapter

        adapter = TypeSafeAdapter(args.model)
    elif args.adapter in {
        "lumma",
        "tinyjev",
        "nimble",
        "bosun",
        "manchego",
        "neohorse",
        "jebadiah",
        "openthai",
        "kotoba",
        "jeff",
        "pngwn",
        "eikos",
        "mini-jev",
        "apus",
        "winnow",
        "openjev-shim",
        "verdict-encoder",
        "von",
        "jevforge",
        "kev",
        "decider",
        "jevk5",
        "minojev",
        "openvons",
        "verdict2",
        "luce",
        "open-jev",
        "alex-openjev",
        "openjev-org",
        "clm",
        "tev",
    }:
        if not args.source:
            parser.error("--source is required for upstream model adapters")
        import importlib

        classes = {
            "lumma": "LummaAdapter",
            "tinyjev": "TinyJevAdapter",
            "nimble": "NimbleAdapter",
            "bosun": "BosunAdapter",
            "manchego": "ManchegoAdapter",
            "neohorse": "NeoHorseAdapter",
            "jebadiah": "JebadiahAdapter",
            "openthai": "OpenThaiAdapter",
            "kotoba": "KotobaAdapter",
            "jeff": "JeffAdapter",
            "pngwn": "PngwnAdapter",
            "eikos": "EikosAdapter",
            "mini-jev": "MiniJevAdapter",
            "apus": "ApusAdapter",
            "winnow": "WinnowAdapter",
            "openjev-shim": "OpenjevShimAdapter",
            "verdict-encoder": "VerdictEncoderAdapter",
            "von": "VonAdapter",
            "jevforge": "JevForgeAdapter",
            "kev": "KevAdapter",
            "decider": "DeciderAdapter",
            "jevk5": "JevK5Adapter",
            "minojev": "MinojevAdapter",
            "openvons": "OpenvonsAdapter",
            "verdict2": "Verdict2Adapter",
            "luce": "LuceAdapter",
            "open-jev": "OpenJevAdapter",
            "alex-openjev": "AlexOpenJevAdapter",
            "openjev-org": "OpenJevOrgAdapter",
            "clm": "CLMAdapter",
            "tev": "TevAdapter",
        }
        module_name = {
            "open-jev": "open_jev",
            "alex-openjev": "alex_openjev",
            "mini-jev": "mini_jev",
            "openjev-shim": "openjev_shim",
            "verdict-encoder": "verdict_encoder",
            "openjev-org": "openjev_org",
        }.get(args.adapter, args.adapter)
        module = importlib.import_module(f"s1mb.adapters.{module_name}")
        options = {"dtype": args.dtype} if args.dtype is not None else {}
        if args.context_limit is not None:
            options["context_limit"] = args.context_limit
        if args.subfolder is not None:
            options["subfolder"] = args.subfolder
        if args.case_batch_size is not None:
            options["case_batch_size"] = args.case_batch_size
        if args.adapter == "winnow":
            options.update(server_host=args.server_host, server_port=args.server_port)
        adapter = getattr(module, classes[args.adapter])(
            args.model, args.revision, args.source, args.device, **options
        )
        if args.attention:
            adapter.set_attention(args.attention)
    else:
        if not args.source:
            parser.error("--source is required for system-ichi")
        from .adapters.system_ichi import SystemIchiAdapter

        adapter = SystemIchiAdapter(
            args.model,
            args.source,
            args.device,
            args.questions_per_call,
            args.query_length,
            args.document_length,
        )
    from .runner import evaluate

    failures = 0
    try:
        for i, benchmark in enumerate(benchmarks):
            print(f"[{i + 1}/{len(benchmarks)}] {benchmark.id}", flush=True)
            result = evaluate(adapter, args.data_dir, benchmark, output, run_id, args.limit)
            failures += result.counts.failed
            print(
                f"  {result.status}: {result.metrics} ({result.elapsed_seconds:.1f}s)", flush=True
            )
            if "api_usage" in result.environment:
                print(f"  API usage: {result.environment['api_usage']}", flush=True)
    finally:
        adapter.close()
    print(f"Results: {output / run_id}", flush=True)
    if failures:
        raise SystemExit(1)
