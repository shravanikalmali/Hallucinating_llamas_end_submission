import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval_config import reset_config

def run_debug_test():
    book_id = "c8e25c068b7c8a00ba00096e73ce7ea893c69aba"
    questions = [
        "Who stands up for Phil at his trial?",
        "Whom does Phil serve after he is acquitted of charges?",
        "What war does Phil serve in, and on which side?",
        "Where is the last place Phil travels?",
        "Who is Phil?",
        "What happens to Phil at the end?",
        "How does Phil survive the war?",
        "Who is Sir John Bristol?",
        "What are the charges against Phil?",
        "Why does Phil join the Royalists?"
    ]

    print("=" * 100)
    print(f"DEBUGGING RETRIEVAL PIPELINE")
    print(f"Book ID: {book_id}")
    print("=" * 100)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/10] QUESTION: {question}")
        print("-" * 50)
        
        # Reset config to defaults for each question
        reset_config()
        
        # Run with verbose=True to see all stages
        result = retrieve_and_answer(
            book_id=book_id,
            question=question,
            verbose=True
        )

        print("-" * 50)
        print(f"FINAL ANSWER: {result.get('answer')}")
        print(f"SCORE       : {result.get('answer_score', 0):.4f}")
        print(f"REASONING   : {result.get('reasoning')}")
        print("=" * 100)

if __name__ == "__main__":
    run_debug_test()
