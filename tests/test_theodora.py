"""
Test script for Theodora questions with ground truth comparison
Uses improved LLM prompts for constrained answering
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval_config import reset_config
test_cases = [
    {
        "question": "Who is Theodora Fitzgerald arranged to marry?",
        "ground_truth": "Josiah Brown",
    },
    {
        "question": "What is Theodora's Father's name?",
        "ground_truth": "Dominic",
    },
    {
        "question": "Who is Theodora's father courting?",
        "ground_truth": "Mrs. McBride",
    },
    {
        "question": "Who does Theodora fall in love with?",
        "ground_truth": "Lord Hector Bracondale",
    },
    {
        "question": "Who switches Theodora's letters?",
        "ground_truth": "Morella Winmarleigh",
    },
    {
        "question": "Where do Theodora and Hector go on the day of their romantic encounter?",
        "ground_truth": "Versailles",
    },
    {
        "question": "Who plays matchmaker for Theodora?",
        "ground_truth": "Mrs. McBride",
    },
    {
        "question": "Where is Josiah Brown from?",
        "ground_truth": "Australia",
    },
    {
        "question": "Where does Theodora see her father again a year after her wedding?",
        "ground_truth": "Paris",
    },
    {
        "question": "Who sends Hector the letter really meant for him?",
        "ground_truth": "Josiah",
    },
]

BOOK_ID = "845186ccb0a481e73525e98cbef89279c2f5721e"


def run_test():
    """Run all test cases and display results with improved LLM prompt"""
    # Use default config (LLM enabled with graph reasoning)
    reset_config()

    print("\n" + "=" * 80)
    print("  THEODORA TEST SUITE (IMPROVED LLM PROMPTS)")
    print("=" * 80)
    print(f"  Book ID: {BOOK_ID}")
    print(f"  Total Questions: {len(test_cases)}")
    print(f"  Strategy: Graph Retrieval + Constrained LLM Prompts")
    print("=" * 80 + "\n")

    correct = 0

    for idx, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"\n{'─' * 80}")
        print(f"[Q{idx}] {question}")
        print(f"{'─' * 80}")

        try:
            result = retrieve_and_answer(
                book_id=BOOK_ID,
                question=question,
                verbose=False,
            )

            answer = result.get("answer", "N/A")
            reasoning = result.get("reasoning", "N/A")
            is_inference = result.get("is_inference", False)

            print(f"\n  🔹 ANSWER:\n     {answer}")
            print(f"\n  🔹 REASONING:\n     {reasoning}")
            print(f"\n  🔹 INFERENCE: {is_inference}")
            print(f"\n  🔹 GROUND TRUTH:\n     {ground_truth}")

            # Check if correct (case-insensitive, strip whitespace)
            answer_normalized = answer.lower().strip()
            truth_normalized = ground_truth.lower().strip()

            if answer_normalized == truth_normalized or truth_normalized in answer_normalized:
                print(f"\n  ✅ CORRECT!")
                correct += 1
            else:
                print(f"\n  ❌ INCORRECT")

        except Exception as e:
            print(f"\n  ❌ ERROR: {str(e)}")

        print()

    print("\n" + "=" * 80)
    print(f"  TEST RESULTS: {correct}/{len(test_cases)} Correct ({100*correct//len(test_cases)}%)")


if __name__ == "__main__":
    run_test()
