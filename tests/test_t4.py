
import json
from src.retrieval.pipeline import retrieve_and_answer

def test_t4():
    book_id = "c8e25c068b7c8a00ba00096e73ce7ea893c69aba"
    question = "What happens to Will Canty when he tries to escape and get help?"
    result = retrieve_and_answer(book_id, question, verbose=True)
    
    print("\n--- RESULTS T4 ---")
    print(f"Answer: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Is Inference: {result.get('is_inference', False)}")

if __name__ == "__main__":
    test_t4()
