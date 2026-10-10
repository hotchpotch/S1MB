"""Command-line evaluation and offline validation."""

import argparse
import json
import math
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
            "llama-cpp",
            "laya",
            "typesafe",
            "knowline",
            "llamacpp",
            "liquid",
            "unee",
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
            "torchcast",
            "solomon",
            "openthai",
            "kotoba",
            "jeff",
            "firelex-jeff",
            "clef",
            "julia",
            "dinah",
            "jevk5-lite",
            "lavoir",
            "lfm-rlcd",
            "decision2",
            "onejev",
            "jev-style",
            "sifr",
            "jiwo",
            "llm2jev",
            "intern-decision",
            "thisthat",
            "exaone-jev",
            "jev-style-2b",
            "sieve",
            "sieve-9b",
            "ajev",
            "xor",
            "blink",
            "jev27",
            "jade",
            "eikos-fp8",
            "lfm-pcd",
            "kodiak",
            "lev1",
            "kas",
            "intelif",
            "jet",
            "rsi",
            "gevva",
            "gev",
            "autotrust-jev",
            "jevstral",
            "nimble-v3",
            "dm-jepa",
            "jevlite",
            "certo",
            "jev-omni",
            "autojev",
            "flymy",
            "lev",
            "gliner2",
            "reranker",
            "metask",
            "nimble-lora",
            "spark",
            "evalengine",
            "deem",
            "hopper",
            "reflex",
            "verdict-small",
            "opendecision",
            "imajev",
            "smalljev",
            "plumb",
            "rune",
            "standardone",
            "jevone",
            "needle",
            "gliformer-jeff",
            "pngwn",
            "eikos",
            "mini-jev",
            "apus",
            "winnow",
            "openjev-shim",
            "verdict-encoder",
            "meta-encoder",
        ],
        required=True,
    )
    run.add_argument("--adapter-kwargs", help="JSON object of llama-cpp constructor keyword arguments")
    run.add_argument("--model")
    run.add_argument("--revision", default="main")
    run.add_argument("--subfolder", help="Alex Openjev checkpoint subfolder")
    run.add_argument("--device", default="cpu")
    run.add_argument(
        "--temperature",
        type=float,
        help="Explicit Meta Encoder cosine-softmax or llm2jev calibration temperature",
    )
    run.add_argument(
        "--server-host", default="127.0.0.1", help="Winnow/Xor/Blink server IPv4: localhost or Tailscale"
    )
    run.add_argument("--server-port", type=int, default=8091, help="Winnow/Xor/Blink server port")
    run.add_argument(
        "--max-candidates",
        type=int,
        help="Explicit candidate capacity for adapters supporting codebook extension",
    )
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
        help="Case window size for Bekko v0, Kev, Open-Jev, CLM, Tev or Liquid",
    )
    run.add_argument("--compile", action="store_true", help="Compile Bekko v0 tensor execution")
    run.add_argument("--microbatch-tokens", type=int, help="Bekko complete-question work budget")
    run.add_argument(
        "--source", help="Upstream source checkout for System Ichi or open-model adapters"
    )
    run.add_argument("--base-revision", help="Explicit dependency checkpoint revision")
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
        validate_run_arguments(args, parser)
        from .dataset_source import dataset_session

        with dataset_session(args.data_dir, offline=args.offline_dataset, category=args.category):
            execute(args, parser)
    else:
        execute(args, parser)


def parse_adapter_kwargs(args, parser):
    """Validate caller-managed llama.cpp constructor options."""
    adapter_kwargs = {}
    if args.adapter_kwargs is not None:
        if args.adapter != "llama-cpp":
            parser.error("--adapter-kwargs applies only to llama-cpp")
        try:
            adapter_kwargs = json.loads(args.adapter_kwargs)
        except json.JSONDecodeError:
            parser.error("--adapter-kwargs must be a JSON object")
        if not isinstance(adapter_kwargs, dict):
            parser.error("--adapter-kwargs must be a JSON object")
        allowed = {"base_url", "served_model", "timeout", "case_batch_size", "max_questions",
                   "max_candidates", "max_request_bytes", "retries", "runtime_settings"}
        if set(adapter_kwargs) - allowed:
            parser.error("Unsupported llama-cpp adapter kwargs")
    return adapter_kwargs


def validate_run_arguments(args, parser):
    """Reject invalid runtime options before acquiring the installed dataset lock."""
    parse_adapter_kwargs(args, parser)
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
        args.adapter not in {"bekko-v0", "kev", "open-jev", "clm", "tev", "liquid", "llamacpp"}
        or args.case_batch_size < 1
    ):
        parser.error(
            "--case-batch-size must be positive and applies only to Bekko v0, Kev, Open-Jev, CLM, Tev, Liquid or llama.cpp"
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
    if args.max_candidates is not None and (
        args.adapter
        not in {
            "apus",
            "verdict-encoder",
            "winnow",
            "firelex-jeff",
            "jevlite",
            "flymy",
            "metask",
            "nimble-lora",
            "smalljev",
            "plumb",
            "standardone",
            "spark",
            "deem",
            "hopper",
            "reflex",
        }
        or not 2 <= args.max_candidates <= 255
    ):
        parser.error("--max-candidates requires a supported adapter and a value in 2..255")
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
            "tinyjev",
            "nimble",
            "lumma",
            "mini-jev",
            "apus",
            "verdict-encoder",
            "kotoba",
            "firelex-jeff",
            "jevlite",
            "certo",
            "jev-omni",
            "autojev",
            "flymy",
            "lev",
            "gliner2",
            "reranker",
            "metask",
            "nimble-lora",
            "spark",
            "evalengine",
            "deem",
            "hopper",
            "reflex",
            "verdict-small",
            "opendecision",
            "imajev",
            "smalljev",
            "plumb",
            "rune",
            "standardone",
            "jevone",
            "needle",
            "gliformer-jeff",
            "meta-encoder",
            "clef",
            "julia",
            "dinah",
            "jevk5-lite",
            "lavoir",
            "lfm-rlcd",
            "decision2",
            "onejev",
            "jev-style",
            "sifr",
            "jiwo",
            "llm2jev",
            "intern-decision",
            "thisthat",
            "exaone-jev",
            "jev-style-2b",
            "sieve",
            "sieve-9b",
            "ajev",
            "xor",
            "blink",
            "jev27",
            "eikos-fp8",
            "lfm-pcd",
            "lev1",
            "kas",
            "intelif",
            "jet",
            "rsi",
            "gevva",
            "gev",
            "autotrust-jev",
            "jevstral",
            "nimble-v3",
            "dm-jepa",
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
        "meta-encoder",
    }:
        parser.error("--attention applies only to measured upstream adapters")
    if args.base_revision is not None and args.adapter not in {"sieve", "lev", "pngwn", "ajev"}:
        parser.error("--base-revision applies only to Sieve, Lev, Pngwn or AJev")
    if args.adapter in {"sieve", "lev", "pngwn", "ajev"} and not args.base_revision:
        parser.error("--base-revision is required for Sieve, Lev, Pngwn or AJev")
    if args.temperature is not None and args.adapter not in {"meta-encoder", "llm2jev"}:
        parser.error("--temperature applies only to Meta Encoder or llm2jev")
    if args.temperature is not None and (
        not math.isfinite(args.temperature) or args.temperature <= 0
    ):
        parser.error("--temperature must be finite and positive for Meta Encoder or llm2jev")
    if args.adapter == "meta-encoder" and args.temperature is None:
        parser.error("--temperature is required for Meta Encoder")
    if args.adapter == "llm2jev" and args.temperature is None:
        parser.error("--temperature is required for llm2jev")


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
    elif args.adapter == "llama-cpp":
        from .adapters.llama_cpp import LlamaCppAdapter

        try:
            adapter = LlamaCppAdapter(args.model, args.revision, **parse_adapter_kwargs(args, parser))
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
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
    elif args.adapter == "liquid":
        from .adapters.liquid import LiquidAdapter

        adapter = LiquidAdapter(
            args.model,
            case_batch_size=args.case_batch_size if args.case_batch_size is not None else 1,
        )
    elif args.adapter == "typesafe":
        from .adapters.typesafe import TypeSafeAdapter

        adapter = TypeSafeAdapter(args.model)
    elif args.adapter == "dm-jepa":
        from .adapters.dm_jepa import DMJEPAAdapter

        adapter = DMJEPAAdapter(
            args.model, args.revision, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "nimble-v3":
        if not args.source:
            parser.error("--source is required for NimbleV3")
        from .adapters.nimble_v3 import NimbleV3Adapter

        adapter = NimbleV3Adapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "jevstral":
        if not args.source:
            parser.error("--source is required for Jevstral")
        from .adapters.jevstral import JevstralAdapter

        adapter = JevstralAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "jev27":
        if not args.source:
            parser.error("--source is required for JEV 27B")
        from .adapters.jev27 import JEV27Adapter

        adapter = JEV27Adapter(
            args.model, args.revision, args.source, args.device,
            args.server_host, args.server_port, context_limit=args.context_limit
        )
    elif args.adapter == "eikos-fp8":
        if not args.source:
            parser.error("--source is required for Eikos FP8")
        from .adapters.eikos_fp8 import EikosFP8Adapter

        adapter = EikosFP8Adapter(
            args.model, args.revision, args.source, args.device,
            args.server_host, args.server_port, context_limit=args.context_limit
        )
    elif args.adapter == "jade":
        if not args.source:
            parser.error("--source is required for JADE")
        from .adapters.jade import JadeAdapter

        adapter = JadeAdapter(
            args.model, args.revision, args.source, args.device,
            args.server_host, args.server_port
        )
    elif args.adapter == "blink":
        if not args.source:
            parser.error("--source is required for Blink")
        from .adapters.blink import BlinkAdapter

        adapter = BlinkAdapter(
            args.model, args.revision, args.source, args.device,
            args.server_host, args.server_port, context_limit=args.context_limit
        )
    elif args.adapter == "xor":
        if not args.source:
            parser.error("--source is required for Xor")
        from .adapters.xor import XorAdapter

        adapter = XorAdapter(
            args.model, args.revision, args.source, args.device,
            args.server_host, args.server_port, context_limit=args.context_limit
        )
    elif args.adapter == "ajev":
        if not args.source or not args.base_revision:
            parser.error("--source and --base-revision are required for AJev")
        from .adapters.ajev import AJevAdapter

        adapter = AJevAdapter(
            args.model, args.revision, args.source, args.device,
            base_revision=args.base_revision, context_limit=args.context_limit
        )
    elif args.adapter == "sieve-9b":
        if not args.source:
            parser.error("--source is required for Sieve-9B")
        from .adapters.sieve_9b import Sieve9BAdapter

        adapter = Sieve9BAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "autotrust-jev":
        if not args.source:
            parser.error("--source is required for AutoTrust JEV")
        from .adapters.autotrust_jev import AutoTrustJevAdapter

        adapter = AutoTrustJevAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "gev":
        if not args.source:
            parser.error("--source is required for GEV")
        from .adapters.gev import GEVAdapter

        adapter = GEVAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "gevva":
        if not args.source:
            parser.error("--source is required for Gevva")
        from .adapters.gevva import GevvaAdapter

        adapter = GevvaAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "rsi":
        if not args.source:
            parser.error("--source is required for RSI")
        from .adapters.rsi import RSIAdapter

        adapter = RSIAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "jet":
        if not args.source:
            parser.error("--source is required for Jet")
        from .adapters.jet import JetAdapter

        adapter = JetAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "intelif":
        if not args.source:
            parser.error("--source is required for Intelif")
        from .adapters.intelif import IntelifAdapter

        adapter = IntelifAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "kas":
        if not args.source:
            parser.error("--source is required for Kas")
        from .adapters.kas import KasAdapter

        adapter = KasAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "lev1":
        if not args.source:
            parser.error("--source is required for Lev1")
        from .adapters.lev1 import Lev1Adapter

        adapter = Lev1Adapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "kodiak":
        if not args.source:
            parser.error("--source is required for Kodiak")
        from .adapters.kodiak import KodiakAdapter

        adapter = KodiakAdapter(args.model, args.revision, args.source, args.device)
    elif args.adapter == "lfm-pcd":
        if not args.source:
            parser.error("--source is required for LFM PCD")
        from .adapters.lfm_pcd import LFMPCDAdapter

        adapter = LFMPCDAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "sieve":
        if not args.source:
            parser.error("--source is required for Sieve")
        from .adapters.sieve import SieveAdapter

        adapter = SieveAdapter(
            args.model, args.revision, args.source, args.device,
            base_revision=args.base_revision, context_limit=args.context_limit,
        )
    elif args.adapter == "jev-style-2b":
        if not args.source:
            parser.error("--source is required for Jev-Style 2B")
        from .adapters.jev_style_2b import JevStyle2BAdapter

        adapter = JevStyle2BAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "exaone-jev":
        if not args.source:
            parser.error("--source is required for EXAONE-JEV")
        from .adapters.exaone_jev import ExaoneJevAdapter

        adapter = ExaoneJevAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "thisthat":
        if not args.source:
            parser.error("--source is required for this-that")
        from .adapters.thisthat import ThisThatAdapter

        adapter = ThisThatAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "intern-decision":
        if not args.source:
            parser.error("--source is required for Intern-Decision")
        from .adapters.intern_decision import InternDecisionAdapter

        adapter = InternDecisionAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "llm2jev":
        if not args.source:
            parser.error("--source is required for llm2jev")
        from .adapters.llm2jev import LLM2JevAdapter

        adapter = LLM2JevAdapter(
            args.model, args.revision, args.source, args.device, temperature=args.temperature,
            context_limit=args.context_limit,
        )
    elif args.adapter == "jiwo":
        if not args.source:
            parser.error("--source is required for jiwo")
        from .adapters.jiwo import JiwoAdapter

        adapter = JiwoAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "sifr":
        if not args.source:
            parser.error("--source is required for Sifr's Decision Index engine dependency")
        from .adapters.sifr import SifrAdapter

        adapter = SifrAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "jev-style":
        if not args.source:
            parser.error("--source is required for Jev-Style")
        from .adapters.jev_style import JevStyleAdapter

        adapter = JevStyleAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "onejev":
        if not args.source:
            parser.error("--source is required for OneJev")
        from .adapters.onejev import OneJevAdapter

        adapter = OneJevAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "decision2":
        if not args.source:
            parser.error("--source is required for Decision 2.0")
        from .adapters.decision2 import Decision2Adapter

        adapter = Decision2Adapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "lfm-rlcd":
        if not args.source:
            parser.error("--source is required for LFM RLCD")
        from .adapters.lfm_rlcd import LFMRLCDAdapter

        adapter = LFMRLCDAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "lavoir":
        if not args.source:
            parser.error("--source is required for Lavoir")
        from .adapters.lavoir import LavoirAdapter

        adapter = LavoirAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "jevk5-lite":
        if not args.source:
            parser.error("--source is required for JevK5-Lite")
        from .adapters.jevk5_lite import JevK5LiteAdapter

        adapter = JevK5LiteAdapter(
            args.model, args.revision, args.source, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "dinah":
        from .adapters.dinah import DinahAdapter

        adapter = DinahAdapter(
            args.model, args.revision, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "julia":
        from .adapters.julia import JuliaAdapter

        adapter = JuliaAdapter(
            args.model, args.revision, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "knowline":
        from .adapters.knowline import KnowLineAdapter

        adapter = KnowLineAdapter(
            args.model,
            args.revision,
            case_batch_size=args.case_batch_size if args.case_batch_size is not None else 16,
        )
    elif args.adapter == "llamacpp":
        from .adapters.llamacpp import LlamaCppAdapter

        adapter = LlamaCppAdapter(
            args.model,
            args.revision,
            case_batch_size=args.case_batch_size if args.case_batch_size is not None else 16,
        )
    elif args.adapter == "clef":
        from .adapters.clef import ClefAdapter

        adapter = ClefAdapter(
            args.model, args.revision, args.device, context_limit=args.context_limit
        )
    elif args.adapter == "meta-encoder":
        from .adapters.meta_encoder import MetaEncoderAdapter

        adapter = MetaEncoderAdapter(
            args.model,
            args.revision,
            args.device,
            temperature=args.temperature,
            context_limit=args.context_limit,
            attention=args.attention or "sdpa",
        )
    elif args.adapter == "unee":
        from .adapters.unee import UneeAdapter

        adapter = UneeAdapter(args.model)
    elif args.adapter in {
        "lumma",
        "tinyjev",
        "nimble",
        "bosun",
        "manchego",
        "neohorse",
        "jebadiah",
        "torchcast",
        "solomon",
        "openthai",
        "kotoba",
        "jeff",
        "firelex-jeff",
        "jevlite",
        "certo",
        "jev-omni",
        "autojev",
        "flymy",
        "lev",
        "gliner2",
        "reranker",
        "metask",
        "nimble-lora",
        "spark",
        "evalengine",
        "deem",
        "hopper",
        "reflex",
        "verdict-small",
        "opendecision",
        "imajev",
        "smalljev",
        "plumb",
        "rune",
        "standardone",
        "jevone",
        "needle",
        "gliformer-jeff",
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
            "torchcast": "TorchcastAdapter",
            "solomon": "SolomonAdapter",
            "openthai": "OpenThaiAdapter",
            "kotoba": "KotobaAdapter",
            "jeff": "JeffAdapter",
            "firelex-jeff": "FirelexJeffAdapter",
            "jevlite": "JevLiteAdapter",
            "certo": "CertoAdapter",
            "jev-omni": "JevOmniAdapter",
            "autojev": "AutoJevAdapter",
            "flymy": "FlymyAdapter",
            "lev": "LevAdapter",
            "gliner2": "Gliner2Adapter",
            "reranker": "RerankerAdapter",
            "metask": "MetaskAdapter",
            "nimble-lora": "NimbleLoraAdapter",
            "spark": "SparkAdapter",
            "evalengine": "EvalEngineAdapter",
            "deem": "DeemAdapter",
            "hopper": "HopperAdapter",
            "reflex": "ReflexAdapter",
            "verdict-small": "VerdictSmallAdapter",
            "opendecision": "OpenDecisionAdapter",
            "imajev": "ImajevAdapter",
            "smalljev": "SmallJevAdapter",
            "plumb": "PlumbAdapter",
            "rune": "RuneAdapter",
            "standardone": "StandardOneAdapter",
            "jevone": "JevOneAdapter",
            "needle": "NeedleAdapter",
            "gliformer-jeff": "GliformerJeffAdapter",
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
            "firelex-jeff": "firelex_jeff",
            "jev-omni": "jev_omni",
            "alex-openjev": "alex_openjev",
            "mini-jev": "mini_jev",
            "openjev-shim": "openjev_shim",
            "verdict-encoder": "verdict_encoder",
            "openjev-org": "openjev_org",
        }.get(args.adapter, args.adapter.replace("-", "_"))
        module = importlib.import_module(f"s1mb.adapters.{module_name}")
        options = {"dtype": args.dtype} if args.dtype is not None else {}
        if args.context_limit is not None:
            options["context_limit"] = args.context_limit
        if args.max_candidates is not None:
            options["max_candidates"] = args.max_candidates
        if args.subfolder is not None:
            options["subfolder"] = args.subfolder
        if args.case_batch_size is not None:
            options["case_batch_size"] = args.case_batch_size
        if args.adapter in {"lev", "pngwn"}:
            options["base_revision"] = args.base_revision
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
