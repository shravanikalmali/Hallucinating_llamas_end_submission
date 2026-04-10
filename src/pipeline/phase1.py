"""
Phase 1 Orchestrator — Offline Indexing

Runs the complete Phase 1 pipeline by importing and calling
the individual stage modules directly (no subprocess).

Stages:
  1. NER extraction
  2. Coreference resolution
  3. Entity filtering
  4. Knowledge graph construction
"""

import sys
import argparse
from pathlib import Path

from src.config import find_data_directory

# Stage modules
from src.nlp import ner, coreference, entity_filter
from src.graph import builder


STAGES = {
    "ner": {"name": "Named Entity Recognition", "module": ner},
    "coref": {"name": "Coreference Resolution", "module": coreference},
    "filter": {"name": "Entity Filtering", "module": entity_filter},
    "graph": {"name": "Knowledge Graph", "module": builder},
}


def run_stage(stage_key: str, book_id: str = None,
              processed_dir: str = None, **kwargs) -> bool:
    """Run a single stage. Returns True on success."""
    stage = STAGES[stage_key]
    print(f"\n{'='*60}")
    print(f"  Stage: {stage['name']}")
    print(f"{'='*60}\n")

    try:
        stage["module"].run(
            book_id=book_id,
            processed_dir=processed_dir,
            **kwargs,
        )
        return True
    except Exception as e:
        print(f"\n❌ Stage '{stage_key}' failed: {e}")
        return False


def run(book_id: str = None, processed_dir: str = None,
        stage: str = None, skip_ner: bool = False, skip_coref: bool = False,
        skip_filter: bool = False,
        skip_graph: bool = False, threshold: int = 5,
        coref_engine: str = "pronoun", device: str = "cpu",
        overwrite: bool = False):
    """Run the Phase 1 pipeline."""
    processed = str(find_data_directory(processed_dir))

    print("\n" + "=" * 60)
    print("  PHASE 1: Offline Indexing (Graph Building)")
    print("=" * 60)
    print(f"  Target: {'Single book (' + book_id + ')' if book_id else 'All books'}")
    print(f"  Data dir: {processed}")
    print("=" * 60)

    if stage:
        stages_to_run = [stage]
    else:
        stages_to_run = []
        if not skip_ner:
            stages_to_run.append("ner")
        if not skip_coref:
            stages_to_run.append("coref")
        if not skip_filter:
            stages_to_run.append("filter")
        if not skip_graph:
            stages_to_run.append("graph")

    if not stages_to_run:
        print("\n⚠️  All stages skipped.")
        return

    results = {}
    for stage_key in stages_to_run:
        kwargs = {"overwrite": overwrite}
        if stage_key == "coref":
            kwargs["engine"] = coref_engine
            kwargs["device"] = device
        if stage_key == "filter":
            kwargs["threshold"] = threshold

        success = run_stage(stage_key, book_id=book_id,
                           processed_dir=processed, **kwargs)
        results[stage_key] = success
        if not success:
            print(f"\n❌ Pipeline halted at stage '{stage_key}'")
            break

    # Summary
    print(f"\n{'='*60}")
    print("  PHASE 1 PIPELINE SUMMARY")
    print("=" * 60)
    for sk in stages_to_run:
        status = "✅ Success" if results.get(sk) else "❌ Failed"
        print(f"  {status:12} — {STAGES[sk]['name']}")
    print("=" * 60)

    all_success = all(results.values())
    return all_success
