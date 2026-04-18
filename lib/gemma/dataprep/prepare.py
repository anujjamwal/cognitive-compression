"""Orchestrator for the Gemma 4 hierarchical-CoT data pipeline.

Stage 1 (`sample.py`) produces `{question, raw_trace, expected_answer, ...}`
records by running an open-reasoning dataset through Gemma 4 with thinking
enabled.  This script (stage 2) loads those records, calls Gemini via
`segment.py` to annotate them with nested structure, validates the output,
and pushes the resulting hierarchical dataset to the HF hub.

Usage:
    python -m lib.gemma.dataprep.prepare \\
        --source-repo anujjamwal/OpenThoughts-GemmaE4B-Traces \\
        --out-repo    anujjamwal/OpenThoughts-GemmaE4B-Hierarchical \\
        --model gemini-3.1-pro-preview \\
        --parallelism 8 \\
        --mode append
"""
from __future__ import annotations

import argparse
import logging

from datasets import Dataset, concatenate_datasets, load_dataset

from . import segment
from .validate import is_well_formed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-repo", required=True,
                   help="HF repo of stage-1 output (question, raw_trace, ...)")
    p.add_argument("--out-repo", required=True,
                   help="HF repo to push the hierarchical dataset to")
    p.add_argument("--model", default=segment.DEFAULT_MODEL,
                   help="Teacher model id (Gemini 3.x)")
    p.add_argument("--parallelism", type=int, default=segment.DEFAULT_PARALLELISM)
    p.add_argument("--output-dir", default=segment.DEFAULT_OUTPUT_DIR)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Max records to process (default: all remaining)")
    p.add_argument("--mode", choices=["append", "overwrite"], required=True)
    return p.parse_args()


def _load_source(repo: str, offset: int, limit: int | None) -> Dataset:
    ds = load_dataset(repo, split="train")
    if offset:
        ds = ds.select(range(offset, len(ds)))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    logger.info("Loaded %d stage-1 records from %s", len(ds), repo)
    return ds


def _load_existing(repo: str) -> Dataset | None:
    try:
        ds = load_dataset(repo, split="train")
        logger.info("Loaded %d existing hierarchical records from %s", len(ds), repo)
        return ds
    except Exception:
        logger.info("No existing dataset at %s — starting fresh", repo)
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    source = _load_source(args.source_repo, args.offset, args.limit)
    existing = _load_existing(args.out_repo) if args.mode == "append" else None

    # Skip records that already have a valid hierarchical_cot in `existing`.
    to_process = source
    if existing is not None:
        done_questions = set(
            existing.filter(lambda x: is_well_formed(x.get("hierarchical_cot", "")))[
                "question"
            ]
        )
        to_process = source.filter(lambda x: x["question"] not in done_questions)
        logger.info(
            "%d already segmented, %d remaining",
            len(source) - len(to_process), len(to_process),
        )

    if len(to_process) == 0:
        logger.info("Nothing to segment.")
        return

    records = [
        {
            "question": r["question"],
            "raw_trace": r["raw_trace"],
            "expected_answer": r.get("expected_answer", ""),
        }
        for r in to_process
    ]
    results = segment.segment_batch(
        records,
        model=args.model,
        parallelism=args.parallelism,
        output_dir=args.output_dir,
    )

    def _attach(row, idx):
        r = results[idx]
        if isinstance(r, Exception):
            row["hierarchical_cot"] = ""
            row["hierarchical_cot_raw"] = ""
        else:
            hcot, raw = r
            row["hierarchical_cot"] = hcot
            row["hierarchical_cot_raw"] = raw
        row["segment_model"] = args.model
        return row

    newly = to_process.map(_attach, with_indices=True, num_proc=1)
    newly = newly.filter(lambda x: is_well_formed(x["hierarchical_cot"]))
    logger.info("%d records passed well-formedness check", len(newly))

    # Assemble the final dataset to push.
    if args.mode == "append" and existing is not None:
        kept = existing.filter(
            lambda x: x["question"] not in set(newly["question"])
        )
        final = concatenate_datasets([kept, newly]) if len(newly) else existing
    else:
        final = newly

    logger.info("Pushing %d records to %s", len(final), args.out_repo)
    final.push_to_hub(args.out_repo)
    logger.info("Done.")


if __name__ == "__main__":
    main()
