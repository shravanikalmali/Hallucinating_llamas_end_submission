"""
Test script for "The Children of the New Forest" questions with ground truth comparison
Uses improved sentence extraction (no LLM) for reliable answering
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval_config import reset_config

test_cases = [
    {
        "question": "What is the title of this story?",
        "ground_truth": "The Children of the New Forest",
    },
    {
        "question": "Who saves the 4 orphans; Edward, Humphrey, Alice and Edith?",
        "ground_truth": "Jacob Armitage",
    },
    {
        "question": "Where did Jacob hide the 4 orphans?",
        "ground_truth": "In his isolated cottage",
    },
    {
        "question": "What is the name of the hostile Puritan gamekeeper?",
        "ground_truth": "Corbould",
    },
    {
        "question": "Who been awarded the old Arnwood estate?",
        "ground_truth": "Heatherstone",
    },
    {
        "question": "Why were Edward's sisters sent away?",
        "ground_truth": "To be brought up as aristocratic ladies",
    },
    {
        "question": "Who eventually falls in love with Edward?",
        "ground_truth": "Patience",
    },
    {
        "question": "What relation do Edward and his siblings pretend to be towards Jacob Armitage?",
        "ground_truth": "His grandchildren",
    },
    {
        "question": "What was Edward's job title while working for Heatherstone?",
        "ground_truth": "Secretary",
    },
    {
        "question": "Who are the four orphan children of the Arnwood house?",
        "ground_truth": "Edward, Humphrey, Alice and Edith",
    },
    {
        "question": "Who rescues the children from fire at Arnwood?",
        "ground_truth": "Jacob Armitage",
    },
    {
        "question": "Who does Edward work for as a secretary?",
        "ground_truth": "Heatherstone",
    },
    {
        "question": "Where are Humphrey live after Edward leaves?",
        "ground_truth": "New Forest",
    },
    {
        "question": "After Edward leaves who raises his sisters Alice and Edith?",
        "ground_truth": "Aristocratic ladies",
    },
    {
        "question": "What is the name of the antagonist Puritan gamekeeper?",
        "ground_truth": "Corbould",
    },
    {
        "question": "When does the story begin?",
        "ground_truth": "1647",
    },
    {
        "question": "What is believed to have happen to Edward, Humphrey, Alice and Edith?",
        "ground_truth": "They were believed to have died in the flames at Arnwood.",
    },
    {
        "question": "Who does Edward love?",
        "ground_truth": "Patience",
    },
    {
        "question": "Who does Edward feel betrayed by and why?",
        "ground_truth": "Edward feels betrayed by Heatherstone when he learns that Heatherstone was awarded the Arnwood estate.",
    },
    {
        "question": "Where does Jacob Armitage hide the four orphan children when he saves them from the fire?",
        "ground_truth": "his isolated cottage",
    },
    {
        "question": "What are the children disquised as by Jacob Armitage?",
        "ground_truth": "his grandchildren",
    },
    {
        "question": "Who takes charge of the children after Armitage's death?",
        "ground_truth": "Edward",
    },
    {
        "question": "What is Pablo rescued from?",
        "ground_truth": "a pitfall trap",
    },
    {
        "question": "What is the name of Heatherstone's daughter?",
        "ground_truth": "Patience",
    },
    {
        "question": "What is Heatherstone's daughter rescued from by Armitage?",
        "ground_truth": "a house fire",
    },
    {
        "question": "What two things cause Edward distress and forces him to flee to France?",
        "ground_truth": "Heatherstone has been awarded the old Arnwood estate and Patience's apparent rejection of his love",
    },
    {
        "question": "Why did Heatherstone acquire the Arnwood estate?",
        "ground_truth": "To give to Edward",
    },
    {
        "question": "What job did Edward do for Heatherstone?",
        "ground_truth": "He was his secretary",
    },
    {
        "question": "Who was originally defeated that caused the soldiers to first search for the New Forest?",
        "ground_truth": "King Charles I",
    },
    {
        "question": "What king has just been defeated when the story begins?",
        "ground_truth": "King Charles I.",
    },
    {
        "question": "Who lives in Arnwood?",
        "ground_truth": "Colonel Beverly.",
    },
    {
        "question": "Who are the orphans in the house saved by?",
        "ground_truth": "Jacob Armitage.",
    },
    {
        "question": "Who does Jacob disguise the children as?",
        "ground_truth": "His grandchildren.",
    },
    {
        "question": "Who do the children rescue from a trap?",
        "ground_truth": "Pablo.",
    },
    {
        "question": "Who wants to harm the family?",
        "ground_truth": "Corbould.",
    },
    {
        "question": "Who does Edward work for?",
        "ground_truth": "Heatherstone.",
    },
    {
        "question": "What is Edward's job?",
        "ground_truth": "He is a secretary.",
    },
    {
        "question": "Who joins the army of King Charles II?",
        "ground_truth": "Edward.",
    },
    {
        "question": "Who acquired the estate for Edward?",
        "ground_truth": "Heatherwood.",
    },
]

BOOK_ID = "04d0a3d15a1e39a94524a3958e433a88ca01fdf9"


def run_test():
    """Run all test cases and display results with sentence extraction"""
    reset_config()

    print("\n" + "=" * 80)
    print("  CHILDREN OF THE NEW FOREST TEST SUITE")
    print("=" * 80)
    print(f"  Book ID: {BOOK_ID}")
    print(f"  Total Questions: {len(test_cases)}")
    print(f"  Strategy: Hybrid Retrieval + Sentence Extraction")
    print("=" * 80 + "\n")

    correct = 0
    partial_matches = 0

    for idx, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"\n{'─' * 80}")
        print(f"[Q{idx:2d}] {question}")
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

            if answer_normalized == truth_normalized:
                print(f"\n  ✅ EXACT MATCH")
                correct += 1
            elif truth_normalized in answer_normalized or answer_normalized in truth_normalized:
                print(f"\n  ⚠️  PARTIAL MATCH")
                partial_matches += 1
            else:
                print(f"\n  ❌ INCORRECT")

        except Exception as e:
            print(f"\n  ❌ ERROR: {str(e)}")

        print()

    print("\n" + "=" * 80)
    accuracy = (100 * correct) // len(test_cases)
    partial_accuracy = (100 * (correct + partial_matches)) // len(test_cases)
    print(f"  TEST RESULTS:")
    print(f"  ✅ Exact Matches: {correct}/{len(test_cases)} ({accuracy}%)")
    print(f"  ⚠️  Partial Matches: {partial_matches}/{len(test_cases)}")
    print(f"  📊 Combined Accuracy: {correct + partial_matches}/{len(test_cases)} ({partial_accuracy}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_test()
