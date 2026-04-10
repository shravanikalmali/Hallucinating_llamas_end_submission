"""
Comprehensive Test Suite for QAPS Dataset
- Parses qaps.csv to extract books and questions
- Tests first 10 books
- Auto-indexes books if needed
- Runs all questions and tracks accuracy
"""
import sys
import csv
import subprocess
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval_config import reset_config
from src.config import find_data_directory


def parse_qaps_csv(csv_file: Path) -> dict:
    """Parse qaps.csv and group questions by book_id."""
    books_data = defaultdict(list)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_id = row['document_id']
            books_data[book_id].append({
                'question': row['question'],
                'answer1': row['answer1'],
                'answer2': row['answer2'],
            })

    return books_data


def book_exists(book_id: str, processed_dir: Path) -> bool:
    """Check if book folder exists in processed_data."""
    book_path = processed_dir / book_id
    return book_path.exists() and book_path.is_dir()


def is_indexed(book_id: str, processed_dir: Path) -> bool:
    """Check if book has been indexed (key files exist)."""
    book_path = processed_dir / book_id
    required_files = [
        'chunks.jsonl',
        'embeddings.npy',
        'filtered_chunk_entities.json'
    ]
    return all((book_path / f).exists() for f in required_files)


def index_book(book_id: str, processed_dir: str) -> bool:
    """Run Phase 1 indexing for a book."""
    print(f"\n  ⏳ Indexing {book_id[:16]}...")
    try:
        result = subprocess.run(
            ['python3', 'main.py', 'index', '--book', book_id, '--processed-dir', processed_dir],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"  ✅ Indexing successful")
            return True
        else:
            print(f"  ❌ Indexing failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Indexing timeout (>5 min)")
        return False
    except Exception as e:
        print(f"  ❌ Indexing error: {str(e)}")
        return False


def test_book(book_id: str, questions: list, processed_dir: str) -> dict:
    """Test all questions for a single book."""
    reset_config()

    print(f"\n{'─' * 100}")
    print(f"  📚 BOOK: {book_id}")
    print(f"  📋 QUESTIONS: {len(questions)}")
    print(f"{'─' * 100}\n")

    results = {
        'exact_matches': 0,
        'partial_matches': 0,
        'incorrect': 0,
        'errors': 0,
        'total': len(questions),
        'accuracy': 0.0,
        'combined_accuracy': 0.0,
    }

    # Only test the first 3 questions for speed/visibility
    questions = questions[:3]
    
    for idx, test_case in enumerate(questions, 1):
        question = test_case['question']
        ground_truth = test_case['answer1']

        try:
            result = retrieve_and_answer(
                book_id=book_id,
                question=question,
                verbose=True, # Enable stage-by-stage printing
            )

            answer = result.get('answer', 'N/A').strip()

            # Normalize for comparison
            answer_lower = answer.lower().strip()
            truth_lower = ground_truth.lower().strip()

            if answer_lower == truth_lower:
                results['exact_matches'] += 1
                status = "✅ EXACT MATCH"
            elif truth_lower in answer_lower or answer_lower in truth_lower:
                results['partial_matches'] += 1
                status = "⚠️  PARTIAL MATCH"
            else:
                results['incorrect'] += 1
                status = "❌ INCORRECT"

            # Print detailed output with both answers clearly
            print(f"\n  [{idx:2d}] {status}")
            print(f"       ❓ QUESTION:")
            print(f"          {question}")
            print(f"       ✅ GROUND TRUTH:")
            print(f"          {ground_truth}")
            print(f"       🤖 OUR ANSWER:")
            print(f"          {answer}")

        except Exception as e:
            results['errors'] += 1
            print(f"\n  [{idx:2d}] 🔴 ERROR")
            print(f"       ❓ QUESTION:")
            print(f"          {question}")
            print(f"       ✅ GROUND TRUTH:")
            print(f"          {ground_truth}")
            print(f"       ⚠️  ERROR MESSAGE:")
            print(f"          {str(e)[:100]}")

    results['accuracy'] = (100 * results['exact_matches']) // results['total']
    results['combined_accuracy'] = (100 * (results['exact_matches'] + results['partial_matches'])) // results['total']

    return results


def run_benchmark():
    """Run full benchmark on first 10 books from qaps.csv."""
    # Parse CSV
    csv_file = Path(__file__).parent.parent / 'qaps.csv'
    if not csv_file.exists():
        print(f"❌ qaps.csv not found at {csv_file}")
        return

    print(f"\n{'=' * 100}")
    print(f"  QAPS DATASET BENCHMARK - First 10 Books")
    print(f"{'=' * 100}")

    books_data = parse_qaps_csv(csv_file)
    unique_books = list(books_data.keys())[:10]

    print(f"\n  Total unique books in CSV: {len(books_data)}")
    print(f"  Testing first 10 books: {len(unique_books)}")

    try:
        processed_dir = find_data_directory()
    except Exception:
        print(f"❌ Could not find processed_data directory")
        return

    processed_dir_str = str(processed_dir)

    overall_stats = {
        'books_tested': 0,
        'books_indexed': 0,
        'books_failed': 0,
        'total_questions': 0,
        'total_correct': 0,
        'total_partial': 0,
    }

    book_results = []

    for book_idx, book_id in enumerate(unique_books, 1):
        print(f"\n\n{'#' * 100}")
        print(f"  [{book_idx}/10] Processing Book: {book_id[:16]}...")
        print(f"{'#' * 100}")

        # Step 1: Check if book exists
        if not book_exists(book_id, processed_dir):
            print(f"  ❌ Book folder does not exist in processed_data")
            overall_stats['books_failed'] += 1
            continue

        print(f"  ✅ Book folder found")

        # Step 2: Check if indexed
        if not is_indexed(book_id, processed_dir):
            print(f"  ⚠️  Book not indexed yet")
            if not index_book(book_id, processed_dir_str):
                overall_stats['books_failed'] += 1
                continue
        else:
            print(f"  ✅ Book already indexed")

        overall_stats['books_indexed'] += 1

        # Step 3: Run all questions
        questions = books_data[book_id]
        results = test_book(book_id, questions, processed_dir_str)
        book_results.append({
            'book_id': book_id,
            'results': results
        })

        overall_stats['books_tested'] += 1
        overall_stats['total_questions'] += results['total']
        overall_stats['total_correct'] += results['exact_matches']
        overall_stats['total_partial'] += results['partial_matches']

        # Print book summary
        print(f"\n  {'─' * 80}")
        print(f"  BOOK SUMMARY:")
        print(f"  ✅ Exact Matches: {results['exact_matches']}/{results['total']} ({results['accuracy']}%)")
        print(f"  ⚠️  Partial Matches: {results['partial_matches']}")
        print(f"  📊 Combined: {results['exact_matches'] + results['partial_matches']}/{results['total']} ({results['combined_accuracy']}%)")
        print(f"  {'─' * 80}")

    # Print final summary
    print(f"\n\n{'=' * 100}")
    print(f"  FINAL SUMMARY - QAPS BENCHMARK")
    print(f"{'=' * 100}")
    print(f"\n  📊 Books Processed:")
    print(f"     ✅ Successfully indexed: {overall_stats['books_indexed']}")
    print(f"     ❌ Failed/Skipped: {overall_stats['books_failed']}")

    if overall_stats['books_tested'] > 0:
        overall_accuracy = (100 * overall_stats['total_correct']) // overall_stats['total_questions']
        combined_accuracy = (100 * (overall_stats['total_correct'] + overall_stats['total_partial'])) // overall_stats['total_questions']

        print(f"\n  🎯 Questions Tested: {overall_stats['total_questions']}")
        print(f"     ✅ Exact Matches: {overall_stats['total_correct']} ({overall_accuracy}%)")
        print(f"     ⚠️  Partial Matches: {overall_stats['total_partial']}")
        print(f"     📊 Combined: {overall_stats['total_correct'] + overall_stats['total_partial']} ({combined_accuracy}%)")

        print(f"\n  📈 Per-Book Results:")
        for item in book_results:
            book_id = item['book_id'][:16]
            results = item['results']
            print(f"     {book_id}: {results['exact_matches']}/{results['total']} ({results['accuracy']}%) | "
                  f"Combined: {results['exact_matches'] + results['partial_matches']}/{results['total']} ({results['combined_accuracy']}%)")

    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    run_benchmark()
