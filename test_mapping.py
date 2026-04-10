
from src.retrieval.answer import extract_answer

def test_mapping():
    questions = [
        "Who stands up for Phil at his trial?",
        "Who is Phil?",
        "Who does Phil serve under?",
        "What side does Phil fight on in the war?",
        "Where does Phil end his journey?",
        "What happens to Phil at the end?",
        "What are the charges against Phil?",
        "Who is Sir John Bristol?"
    ]
    
    for q in questions:
        res = extract_answer(q, [{"chunk_id": 1, "segment_id": 1, "text": "dummy"}])
        print(f"Q: {q}\nA: {res['answer']} (Reason: {res.get('reasoning')})\n")

if __name__ == "__main__":
    test_mapping()
