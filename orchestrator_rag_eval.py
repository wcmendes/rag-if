#!/usr/bin/env python3
"""
Orchestrator for RAG evaluation pipeline.

Pipeline:
  collect   — read perguntas.json → run RAG per question → save respostas_rag.json
  evaluate  — read respostas_rag.json → run RAGAS → save resultados_ragas/
  all       — collect + evaluate in sequence

CLI:
  python orchestrator_rag_eval.py --mode collect  --input perguntas.json --output respostas_rag.json
  python orchestrator_rag_eval.py --mode evaluate --input respostas_rag.json --output-dir ./resultados_ragas
  python orchestrator_rag_eval.py --mode all      --input perguntas.json  --output respostas_rag.json --output-dir ./resultados_ragas

Checkpoint / resume:
  collect saves after every question. If interrupted, re-running with the same
  --output file will skip already-processed questions automatically.

RAGAS version:
  Targets ragas>=0.2.0. Falls back to the 0.1.x API if an older version is detected.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("eval_pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Third-party libraries (httpx, sentence_transformers, huggingface_hub) inherit
# the root INFO level and flood stdout with HTTP request logs. Silence them.
for _lib in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


# ── 1. Input handling ──────────────────────────────────────────────────────────

def load_questions(path: str) -> list[dict[str, Any]]:
    """Load and validate perguntas.json. Skips items without question/ground_truth."""
    p = Path(path)
    if not p.exists():
        logger.error("Input file not found: %s", path)
        sys.exit(1)

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        sys.exit(1)

    if not isinstance(data, list):
        logger.error("Expected a JSON array in %s", path)
        sys.exit(1)

    valid: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Item %d is not an object — skipping", i)
            continue
        if not item.get("question"):
            logger.warning("Item %d missing 'question' — skipping: %s", i, item)
            continue
        if not item.get("ground_truth"):
            logger.warning("Item %d missing 'ground_truth' — skipping: %s", i, item)
            continue
        valid.append(item)

    logger.info("Loaded %d valid question(s) from %s", len(valid), path)
    return valid


# ── 2. RAG call ────────────────────────────────────────────────────────────────

def call_ask(question: str, n_results: int = 5) -> dict[str, Any]:
    """Call query_rag() from ask.py and return its raw output."""
    try:
        # Imported here to avoid loading heavy models at orchestrator startup
        from ask import query_rag  # type: ignore[import]
        return query_rag(question, n_results=n_results)
    except Exception as exc:
        logger.error("query_rag failed for %r: %s", question[:60], exc)
        return {
            "question": question,
            "answer": f"[ERROR: {exc}]",
            "contexts": [],
            "source_ids": [],
            "file_names": [],
        }


def normalize_ask_output(raw: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """
    Merge the original question record with the RAG output.

    Preserves all original fields (id, question, ground_truth, ...) and
    appends answer, contexts, and optional metadata.
    """
    result: dict[str, Any] = {**original}
    result["answer"] = raw.get("answer", "")
    result["contexts"] = raw.get("contexts", [])

    if raw.get("source_ids"):
        result["source_ids"] = raw["source_ids"]
    if raw.get("file_names"):
        result["file_names"] = raw["file_names"]

    if not result["contexts"]:
        logger.warning("Empty contexts for: %s", original.get("question", "")[:60])

    return result


# ── 3. Checkpoint / incremental save ──────────────────────────────────────────

def load_checkpoint(output_path: str) -> dict[Any, dict[str, Any]]:
    """
    Load the existing output file as {id_or_question → record} for resume support.
    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    p = Path(output_path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}
        checkpoint: dict[Any, dict[str, Any]] = {}
        errors = 0
        for item in data:
            # Skip items that failed — they will be retried on the next run
            if str(item.get("answer", "")).startswith("[ERROR:"):
                errors += 1
                continue
            key = item.get("id") if item.get("id") is not None else item.get("question", "")
            if key is not None:
                checkpoint[key] = item
        logger.info(
            "Checkpoint loaded: %d done, %d error(s) will be retried",
            len(checkpoint), errors,
        )
        return checkpoint
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not load checkpoint from %s — starting fresh", output_path)
        return {}


def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically: write to a temp file then rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except OSError as exc:
        logger.error("Failed to save %s: %s", path, exc)
        raise


def save_incremental_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Atomically save the current results list after every question."""
    _atomic_write_json(output_path, results)


# ── 4. Final multi-format save ─────────────────────────────────────────────────

def save_all_formats(results: list[dict[str, Any]], output_path: str) -> None:
    """Save the enriched dataset as JSON, JSONL, and CSV."""
    base = Path(output_path)

    _atomic_write_json(str(base), results)
    logger.info("Saved JSON  : %s", base)

    jsonl_path = base.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved JSONL : %s", jsonl_path)

    csv_path = base.with_suffix(".csv")
    scalar_keys = ["id", "question", "ground_truth", "answer"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys + ["contexts"], extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = {k: item.get(k, "") for k in scalar_keys}
            row["contexts"] = " ||| ".join(item.get("contexts", []))
            writer.writerow(row)
    logger.info("Saved CSV   : %s", csv_path)


# ── 5. RAGAS version detection ─────────────────────────────────────────────────

def _detect_ragas_version() -> tuple[int, int]:
    """Return (major, minor) of the installed ragas package."""
    try:
        import ragas  # noqa: PLC0415
        parts = ragas.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except ImportError:
        raise ImportError(
            "ragas is not installed.\n"
            "Run:  pip install 'ragas>=0.2.0' langchain-openai openai"
        )
    except (ValueError, IndexError):
        return (0, 2)  # assume modern API if version string is unexpected


# ── 6. RAGAS schema mapping ────────────────────────────────────────────────────

def map_to_ragas_schema(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Map external naming to RAGAS internal schema:
      question     → user_input
      answer       → response
      contexts     → retrieved_contexts
      ground_truth → reference
    """
    mapped: list[dict[str, Any]] = []
    for item in results:
        if not item.get("answer"):
            logger.warning(
                "Skipping item with empty answer: %s", item.get("question", "")[:60]
            )
            continue
        mapped.append(
            {
                "user_input": item["question"],
                "response": item["answer"],
                "retrieved_contexts": item.get("contexts", []),
                "reference": item.get("ground_truth", ""),
            }
        )
    return mapped


# ── 7. RAGAS model configuration ──────────────────────────────────────────────

def configure_ragas_models() -> tuple[Any, Any]:
    """
    Configure and return (llm_wrapper, embeddings_wrapper) for RAGAS.

    Env vars:
      RAGAS_LLM_PROVIDER    = openai | ollama           (default: openai)
      RAGAS_JUDGE_MODEL     = model name                (default: gpt-4o)
      RAGAS_EMBEDDING_MODEL = embedding model name      (default: text-embedding-3-small)
      OPENAI_API_KEY        = required when provider=openai
      OLLAMA_URL            = Ollama base URL           (default: http://localhost:11434)
    """
    try:
        from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Could not import ragas wrappers: %s", exc)
        sys.exit(1)

    provider = os.getenv("RAGAS_LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OPENAI_API_KEY is required when RAGAS_LLM_PROVIDER=openai. "
                "Set it in your .env file."
            )
            sys.exit(1)
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: PLC0415
        except ImportError:
            logger.error("langchain-openai not installed. Run: pip install langchain-openai")
            sys.exit(1)

        judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
        emb_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
        llm = LangchainLLMWrapper(
            ChatOpenAI(model=judge_model, temperature=0, api_key=api_key)
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=emb_model, api_key=api_key)
        )
        logger.info("RAGAS judge  : OpenAI %s", judge_model)
        logger.info("RAGAS embeds : OpenAI %s", emb_model)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: PLC0415
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama  # noqa: PLC0415
                from langchain_community.embeddings import OllamaEmbeddings  # noqa: PLC0415
            except ImportError:
                logger.error(
                    "langchain-ollama not installed. Run: pip install langchain-ollama"
                )
                sys.exit(1)

        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        judge_model = os.getenv("RAGAS_JUDGE_MODEL", "llama3.2")
        emb_model = os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text")
        llm = LangchainLLMWrapper(
            ChatOllama(base_url=ollama_url, model=judge_model, temperature=0)
        )
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(base_url=ollama_url, model=emb_model)
        )
        logger.info("RAGAS judge  : Ollama %s @ %s", judge_model, ollama_url)
        logger.info("RAGAS embeds : Ollama %s", emb_model)

    else:
        logger.error(
            "Unknown RAGAS_LLM_PROVIDER=%r — valid values: openai, ollama", provider
        )
        sys.exit(1)

    return llm, embeddings


# ── 8. RAGAS metrics ───────────────────────────────────────────────────────────

# In ragas>=0.4.x the metric classes live in private submodules and are NOT
# re-exported at the ragas.metrics top level. We search each submodule first,
# then fall back to the public namespace for older versions.
_METRIC_SEARCH_PATHS: list[tuple[list[str], list[str], str]] = [
    (
        ["ragas.metrics._faithfulness", "ragas.metrics"],
        ["Faithfulness"],
        "faithfulness",
    ),
    (
        ["ragas.metrics._answer_relevance", "ragas.metrics"],
        ["AnswerRelevancy", "ResponseRelevancy"],
        "answer_relevancy",
    ),
    (
        ["ragas.metrics._context_precision", "ragas.metrics"],
        ["ContextPrecision", "LLMContextPrecisionWithReference"],
        "context_precision",
    ),
    (
        ["ragas.metrics._context_recall", "ragas.metrics"],
        ["ContextRecall", "LLMContextRecall"],
        "context_recall",
    ),
    (
        ["ragas.metrics._answer_correctness", "ragas.evaluation", "ragas.metrics"],
        ["AnswerCorrectness"],
        "answer_correctness",
    ),
]


def _try_import_metric(module_paths: list[str], names: list[str]) -> tuple[Any, str]:
    """Search module_paths for the first class whose name is in names."""
    import importlib

    for mod_path in module_paths:
        try:
            mod = importlib.import_module(mod_path)
            for name in names:
                cls = getattr(mod, name, None)
                if cls is not None:
                    return cls, name
        except ImportError:
            continue
    return None, None  # type: ignore[return-value]


def _load_metrics_v2() -> list[Any]:
    """
    Instantiate RAGAS metrics (0.2.x / 0.4.x).
    LLM/embeddings are injected later via the evaluate() call.
    """
    loaded: list[Any] = []
    loaded_names: list[str] = []

    for module_paths, class_names, label in _METRIC_SEARCH_PATHS:
        cls, found_name = _try_import_metric(module_paths, class_names)
        if cls is None:
            logger.warning("Metric '%s' not found — skipping", label)
            continue
        try:
            loaded.append(cls())
            loaded_names.append(found_name)
        except Exception as exc:
            logger.warning("Could not instantiate %s: %s — skipping", found_name, exc)

    if not loaded:
        raise ImportError(
            "No RAGAS metrics could be loaded. "
            "Check that ragas>=0.2.0 is installed correctly."
        )

    logger.info("Metrics loaded: %s", loaded_names)
    return loaded


# ── 9. RAGAS evaluation runners ────────────────────────────────────────────────

def run_ragas_evaluation(results: list[dict[str, Any]]) -> Any:
    """Dispatch to the correct RAGAS API based on installed version."""
    major, minor = _detect_ragas_version()
    logger.info("Detected ragas %d.%d", major, minor)

    mapped = map_to_ragas_schema(results)
    if not mapped:
        logger.error("No valid samples to evaluate after mapping.")
        sys.exit(1)

    if major == 0 and minor < 2:
        return _run_ragas_v1(mapped)
    return _run_ragas_v2(mapped)


def _run_ragas_v1(mapped: list[dict[str, Any]]) -> Any:
    """Legacy RAGAS 0.1.x path."""
    logger.info("Using legacy RAGAS 0.1.x API")
    try:
        from ragas import evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Could not import RAGAS 0.1.x components: %s", exc)
        sys.exit(1)

    dataset = Dataset.from_dict(
        {
            "question": [m["user_input"] for m in mapped],
            "answer": [m["response"] for m in mapped],
            "contexts": [m["retrieved_contexts"] for m in mapped],
            "ground_truth": [m["reference"] for m in mapped],
        }
    )
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


def _run_ragas_v2(mapped: list[dict[str, Any]]) -> Any:
    """Modern RAGAS 0.2.x path."""
    logger.info("Using RAGAS 0.2.x API")
    try:
        from ragas import evaluate, EvaluationDataset  # noqa: PLC0415
    except ImportError:
        try:
            from ragas import evaluate  # noqa: PLC0415
            from ragas.evaluation import EvaluationDataset  # noqa: PLC0415
        except ImportError as exc:
            logger.error("Could not import RAGAS 0.2.x components: %s", exc)
            sys.exit(1)

    try:
        from ragas.dataset_schema import SingleTurnSample  # noqa: PLC0415
    except ImportError:
        try:
            from ragas import SingleTurnSample  # noqa: PLC0415
        except ImportError as exc:
            logger.error("Could not import SingleTurnSample: %s", exc)
            sys.exit(1)

    llm, embeddings = configure_ragas_models()
    metrics = _load_metrics_v2()

    samples = [
        SingleTurnSample(
            user_input=m["user_input"],
            response=m["response"],
            retrieved_contexts=m["retrieved_contexts"],
            reference=m["reference"],
        )
        for m in mapped
    ]
    dataset = EvaluationDataset(samples=samples)

    try:
        from ragas import RunConfig  # noqa: PLC0415
        # Local models (Ollama) handle one request at a time.
        # max_workers=1 prevents parallel calls that overwhelm the local server.
        run_config = RunConfig(timeout=600, max_retries=2, max_wait=60, max_workers=1)
    except ImportError:
        run_config = None

    kwargs: dict[str, Any] = dict(
        dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings
    )
    if run_config is not None:
        kwargs["run_config"] = run_config

    return evaluate(**kwargs)


# ── 10. Save RAGAS outputs ─────────────────────────────────────────────────────

def save_ragas_outputs(result: Any, output_dir: str) -> None:
    """Save RAGAS results as CSV and JSON, and print the scores summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        df = result.to_pandas()
        csv_path = out / f"ragas_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("Saved RAGAS CSV : %s", csv_path)

        json_path = out / f"ragas_results_{timestamp}.json"
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        logger.info("Saved RAGAS JSON: %s", json_path)
    except Exception as exc:
        logger.warning("Could not export RAGAS detailed results: %s", exc)

    # Print the aggregate scores
    print("\n─── RAGAS Scores ────────────────────────────────────")
    try:
        scores = result.scores if hasattr(result, "scores") else {}
        if not scores and hasattr(result, "__iter__"):
            # v1 result may be dict-like
            scores = dict(result)
        for metric_name, score in scores.items():
            try:
                print(f"  {metric_name:<35} {float(score):.4f}")
            except (TypeError, ValueError):
                print(f"  {metric_name:<35} {score}")
    except Exception:
        print(f"  Raw result: {result}")
    print("─────────────────────────────────────────────────────\n")


# ── 11. Collect mode ───────────────────────────────────────────────────────────

def run_collect(
    input_path: str, output_path: str, n_results: int = 5
) -> list[dict[str, Any]]:
    """
    Run RAG for every question in input_path.

    Skips questions already present in output_path (resume support).
    Saves incrementally after every question to prevent data loss on interruption.
    """
    questions = load_questions(input_path)
    checkpoint = load_checkpoint(output_path)

    # Build ordered results list: checkpoint first (preserves order), then new items
    results: list[dict[str, Any]] = list(checkpoint.values())
    processed_keys: set[Any] = set(checkpoint.keys())

    total = len(questions)
    skipped = 0
    processed = 0

    for i, item in enumerate(questions, start=1):
        key: Any = item.get("id") if item.get("id") is not None else item.get("question", "")

        if key in processed_keys:
            skipped += 1
            logger.info("[%d/%d] Skipping (already processed): %s", i, total, str(key)[:60])
            continue

        logger.info("[%d/%d] Processing: %s", i, total, item["question"][:70])

        raw = call_ask(item["question"], n_results=n_results)
        enriched = normalize_ask_output(raw, item)

        results.append(enriched)
        processed_keys.add(key)
        processed += 1

        # Save after every question (checkpoint)
        save_incremental_results(results, output_path)
        logger.info("  → %d context(s) | checkpoint saved", len(enriched.get("contexts", [])))

    logger.info(
        "Collect done — processed: %d, skipped: %d, total: %d",
        processed,
        skipped,
        total,
    )

    save_all_formats(results, output_path)
    return results


# ── 12. Evaluate mode ──────────────────────────────────────────────────────────

def run_evaluate(input_path: str, output_dir: str) -> None:
    """Load respostas_rag.json and run RAGAS evaluation."""
    p = Path(input_path)
    if not p.exists():
        logger.error("Enriched dataset not found: %s", input_path)
        sys.exit(1)

    with open(p, encoding="utf-8") as f:
        results: list[dict[str, Any]] = json.load(f)

    if not isinstance(results, list) or not results:
        logger.error("Empty or invalid dataset in %s", input_path)
        sys.exit(1)

    logger.info("Starting RAGAS evaluation on %d item(s)", len(results))
    result = run_ragas_evaluation(results)
    save_ragas_outputs(result, output_dir)


# ── 13. CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG evaluation pipeline: collect answers then evaluate with RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "evaluate", "all"],
        required=True,
        help=(
            "collect: run RAG for all questions and save enriched dataset; "
            "evaluate: run RAGAS on enriched dataset; "
            "all: collect then evaluate"
        ),
    )
    parser.add_argument(
        "--input",
        default="perguntas.json",
        help=(
            "Input file. For collect/all: path to perguntas.json. "
            "For evaluate: path to respostas_rag.json. "
            "(default: perguntas.json)"
        ),
    )
    parser.add_argument(
        "--output",
        default="respostas_rag.json",
        help="Output path for the enriched dataset (collect/all). (default: respostas_rag.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="./resultados_ragas",
        help="Directory for RAGAS results (evaluate/all). (default: ./resultados_ragas)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of RAG chunks to retrieve per question. (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=== RAG Evaluation Pipeline | mode=%s ===", args.mode)

    if args.mode == "collect":
        run_collect(args.input, args.output, n_results=args.n_results)

    elif args.mode == "evaluate":
        run_evaluate(args.input, args.output_dir)

    elif args.mode == "all":
        run_collect(args.input, args.output, n_results=args.n_results)
        run_evaluate(args.output, args.output_dir)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
