
import json
from src.retrieval.pipeline import retrieve_and_answer

def test_t5():
    book_id = "c8e25c068b7c8a00ba00096e73ce7ea893c69aba"
    question = "Why did Phil refuse to testify against the pirates?"
    result = retrieve_and_answer(book_id, question, verbose=True)

    print("\n--- RESULTS T5 ---")
    print(f"Answer: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Retrieved IDs: {result['retrieval_log']['retrieved_chunk_ids'][:20]}...")
    print(f"Contains 374: {374 in result['retrieval_log']['retrieved_chunk_ids']}")

if __name__ == "__main__":
    test_t5()
