#!/usr/bin/env python3
"""
============================================================
  INLP Project — Hallucinating Llamas
  Narrative Question Answering over Project Gutenberg Books
============================================================

Unified CLI entry point with subcommands:

  python main.py preprocess          # Phase 0:   Clean and chunk books
  python main.py embed               # Phase 0.5: Generate embeddings
  python main.py split               # Phase 0.5: Split embeddings per-book
  python main.py faiss               # Phase 0.5: Build FAISS indexes
  python main.py index               # Phase 1:   Full offline indexing pipeline
  python main.py entity-index        # Phase 1.1: Build direct entity lookup index
  python main.py query               # Phase 2:   Retrieval + answer
  python main.py verify              # Run verification checks
"""

import argparse
import sys


def cmd_preprocess(args):
    """Run text preprocessing and chunking."""
    from src.preprocessing.cleaner import run as run_clean
    from src.preprocessing.chunker import run as run_chunk
    from src.preprocessing.metadata import run as run_metadata

    print("═" * 60)
    print("  Phase 0: Data Preparation")
    print("═" * 60)

    if not args.skip_clean:
        run_clean(data_dir=args.data_dir, documents_csv=args.documents_csv)
    if not args.skip_metadata:
        run_metadata(processed_dir=args.processed_dir)
    if not args.skip_chunk:
        run_chunk(processed_dir=args.processed_dir)


def cmd_embed(args):
    """Generate embeddings for all chunks."""
    from src.indexing.embedder import run
    run(processed_dir=args.processed_dir, device=args.device)


def cmd_split(args):
    """Split global embeddings into per-book files."""
    from src.indexing.splitter import run
    run(processed_dir=args.processed_dir)


def cmd_faiss(args):
    """Build FAISS indexes."""
    from src.indexing.faiss_builder import run
    run(book_id=args.book, processed_dir=args.processed_dir)


def cmd_index(args):
    """Run the full Phase 1 offline indexing pipeline."""
    from src.pipeline.phase1 import run
    success = run(
        book_id=args.book, processed_dir=args.processed_dir,
        stage=args.stage, skip_ner=args.skip_ner, skip_coref=args.skip_coref,
        skip_filter=args.skip_filter,
        skip_graph=args.skip_graph, threshold=args.threshold,
        coref_engine=args.coref_engine, device=args.device,
        overwrite=args.overwrite,
    )
    sys.exit(0 if success else 1)


def cmd_entity_index(args):
    """Build direct entity-to-chunk lookup index."""
    from src.indexing.entity_indexer import run
    run(book_id=args.book, processed_dir=args.processed_dir, overwrite=args.overwrite)


def cmd_query(args):
    """Run query-time retrieval and answer extraction."""
    # Load config file if provided
    if args.config:
        from src.retrieval_config import load_config
        load_config(args.config)
    from src.retrieval.pipeline import run
    run(
        book_id=args.book, question=args.question,
        questions_file=args.questions_file, processed_dir=args.processed_dir,
        top_k=args.top_k, mmr_lambda=args.mmr_lambda,
        final_count=args.final_count, expansion_max=args.expansion_max,
        no_entity_expansion=args.no_entity_expansion,
        no_cross_encoder=args.no_cross_encoder,
        no_neighbors=args.no_neighbors,
        no_graph_reasoning=args.no_graph_reasoning,
        no_hybrid_llm=args.no_hybrid_llm,
        output=args.output, verbose=args.verbose,
    )


def cmd_verify(args):
    """Run verification checks."""
    if args.target == "embeddings":
        from tests.verify_embeddings import run
        run(processed_dir=args.processed_dir)
    elif args.target == "faiss":
        from tests.verify_faiss import run
        run(book_id=args.book, processed_dir=args.processed_dir)
    elif args.target == "chunking":
        from tests.test_chunking import main
        main()
    else:
        from tests.verify_embeddings import run as run_emb
        from tests.verify_faiss import run as run_faiss
        print("── Verifying Embeddings ──")
        run_emb(processed_dir=args.processed_dir)
        print("\n── Verifying FAISS ──")
        run_faiss(processed_dir=args.processed_dir)


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="INLP Hallucinating Llamas — Narrative QA Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # ── preprocess ────────────────────────────────────────────────────────
    p_prep = subparsers.add_parser("preprocess", help="Phase 0: Clean + chunk books")
    p_prep.add_argument("--data-dir", help="Raw data directory")
    p_prep.add_argument("--documents-csv", help="Path to documents.csv")
    p_prep.add_argument("--processed-dir", help="Output directory")
    p_prep.add_argument("--skip-clean", action="store_true")
    p_prep.add_argument("--skip-metadata", action="store_true")
    p_prep.add_argument("--skip-chunk", action="store_true")

    # ── embed ─────────────────────────────────────────────────────────────
    p_emb = subparsers.add_parser("embed", help="Phase 0.5: Generate embeddings")
    p_emb.add_argument("--processed-dir", help="Processed data directory")
    p_emb.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # ── split ─────────────────────────────────────────────────────────────
    p_split = subparsers.add_parser("split", help="Phase 0.5: Split embeddings per-book")
    p_split.add_argument("--processed-dir", help="Processed data directory")

    # ── faiss ─────────────────────────────────────────────────────────────
    p_faiss = subparsers.add_parser("faiss", help="Phase 0.5: Build FAISS indexes")
    p_faiss.add_argument("--book", help="Single book ID")
    p_faiss.add_argument("--processed-dir", help="Processed data directory")

    # ── index ─────────────────────────────────────────────────────────────
    p_idx = subparsers.add_parser("index", help="Phase 1: Full offline indexing")
    p_idx.add_argument("--book", help="Single book ID")
    p_idx.add_argument("--processed-dir", help="Processed data directory")
    p_idx.add_argument("--stage", choices=["ner", "coref", "filter", "graph"])
    p_idx.add_argument("--skip-ner", action="store_true")
    p_idx.add_argument("--skip-coref", action="store_true")
    p_idx.add_argument("--skip-filter", action="store_true")
    p_idx.add_argument("--skip-graph", action="store_true")
    p_idx.add_argument("--threshold", type=int, default=5)
    p_idx.add_argument("--coref-engine", choices=["pronoun", "neural", "name"], default="pronoun")
    p_idx.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p_idx.add_argument("--overwrite", action="store_true")

    # ── entity-index ──────────────────────────────────────────────────────
    p_eidx = subparsers.add_parser("entity-index", help="Phase 1.1: Direct entity lookup index")
    p_eidx.add_argument("--book", help="Single book ID")
    p_eidx.add_argument("--processed-dir", help="Processed data directory")
    p_eidx.add_argument("--overwrite", action="store_true")

    # ── query ─────────────────────────────────────────────────────────────
    p_q = subparsers.add_parser("query", help="Phase 2: Retrieval + Answer")
    p_q.add_argument("--book", required=True, help="Book ID")
    p_q.add_argument("--question", help="Question to answer")
    p_q.add_argument("--questions-file", help="JSONL file with questions")
    p_q.add_argument("--processed-dir", help="Processed data directory")
    p_q.add_argument("--top-k", type=int, default=25)
    p_q.add_argument("--mmr-lambda", type=float, default=0.7)
    p_q.add_argument("--final-count", type=int, default=8)
    p_q.add_argument("--expansion-max", type=int, default=3)
    p_q.add_argument("--no-entity-expansion", action="store_true")
    p_q.add_argument("--no-cross-encoder", action="store_true")
    p_q.add_argument("--no-neighbors", action="store_true")
    p_q.add_argument("--no-graph-reasoning", action="store_true")
    p_q.add_argument("--no-hybrid-llm", action="store_true")
    p_q.add_argument("--config", help="Path to JSON config file for ablation experiments")
    p_q.add_argument("--output", help="Output JSONL path")
    p_q.add_argument("--verbose", action="store_true")

    # ── verify ────────────────────────────────────────────────────────────
    p_v = subparsers.add_parser("verify", help="Run verification checks")
    p_v.add_argument("--target", choices=["embeddings", "faiss", "chunking", "all"],
                     default="all")
    p_v.add_argument("--book", help="Single book ID (faiss only)")
    p_v.add_argument("--processed-dir", help="Processed data directory")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "preprocess": cmd_preprocess,
        "embed": cmd_embed,
        "split": cmd_split,
        "faiss": cmd_faiss,
        "index": cmd_index,
        "entity-index": cmd_entity_index,
        "query": cmd_query,
        "verify": cmd_verify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
