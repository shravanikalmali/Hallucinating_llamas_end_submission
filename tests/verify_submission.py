import sys
from pathlib import Path
from src.retrieval.pipeline import retrieve_and_answer
import csv

def run_verification():
    books = [
        {
            "id": "c8e25c068b7c8a00ba00096e73ce7ea893c69aba",
            "title": "The Dark Frigate",
            "queries": [
                {"q": "Whom does Phil serve after he is acquitted of charges?", "exp": "Sir John Bristol"},
                {"q": "What are the charges against Phil?", "exp": "piracy / robbery / felony"},
                {"q": "Why does Phil join the Royalists?", "exp": "forced man / no choice"}
            ]
        },
        {
            "id": "00936497f5884881f1df23f4834f6739552cee8b",
            "title": "In Search of the Castaways",
            "queries": [
                {"q": "What type of birds attack the balloon?", "exp": "Condors"},
                {"q": "What country do the explorers eventually return to at the end?", "exp": "England"}
            ]
        }
    ]

    print("=" * 80)
    print(f"{'QUESTION':<50} | {'EXPECTED':<15} | {'PIPELINE RESULT'}")
    print("-" * 80)

    for book in books:
        print(f"\nBOOK: {book['title']} ({book['id']})")
        for item in book["queries"]:
            res = retrieve_and_answer(book["id"], item["q"], verbose=False)
            ans = res.get("answer", "NOT FOUND")
            # Truncate for display
            disp_q = (item['q'][:47] + '..') if len(item['q']) > 47 else item['q']
            print(f"{disp_q:<50} | {item['exp']:<15} | {ans}")

if __name__ == "__main__":
    run_verification()
