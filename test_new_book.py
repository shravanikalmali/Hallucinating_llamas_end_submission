
import json
from pathlib import Path
from src.retrieval.pipeline import retrieve_and_answer
from src.config import PROCESSED_DIR

def test_new_book():
    book_id = "04954299c7b6bdc7b31b951bc0daa277353576a9" # Mrs. Frisby
    questions = [
        {"q": "Who is Dragon?", "a": "the farmer's cat / A cat"},
        {"q": "Who Killed Mr. Frisby?", "a": "Dragon"},
        {"q": "Who put Ms. Frisby in a cage?", "a": "Billy"}
    ]
    
    print(f"TESTING PIPELINE FOR BOOK: {book_id}")
    print("-" * 50)
    
    for item in questions:
        q = item["q"]
        gold = item["a"]
        
        print(f"QUESTION: {q}")
        print(f"EXPECTED: {gold}")
        
        res = retrieve_and_answer(book_id, q, verbose=False)
        
        print(f"PIPELINE: {res.get('answer')}")
        print(f"REASONING: {res.get('reasoning')}")
        print("-" * 50)

if __name__ == "__main__":
    test_new_book()
