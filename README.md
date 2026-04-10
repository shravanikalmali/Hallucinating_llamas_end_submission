# Narrative QA with MMR Retrieval and Dynamic Graph Generation

### Overview
This project addresses the challenges of long-form narrative understanding by building a multi-modal retrieval system. Instead of relying solely on vector search, we construct a persistent **Static Knowledge Graph** for each book to track character relations and event causality, allowing for more grounded and context-aware answer extraction.

### Key Features
*   **Hybrid Retrieval**: Combines Dense (FAISS) and Sparse (BM25) search for high recall.
*   **Narrative Knowledge Graphs**: Structural representation of characters and event-triggers.
*   **Graph-Neighbor Expansion**: Uses the graph to pull in context that lacks direct keyword overlap.
*   **Temporal & Milestone Boosting**: Prioritizes chunks based on the story timeline.
*   **Generative Fallback**: High-precision recovery using Gemini-2.0 via OpenRouter.

### Pipeline Architecture
1.  **Preprocessing**: Text cleaning and hierarchical chunking.
2.  **Indexing**: Entity detection (NER), Coreference Resolution, and FAISS Vector construction.
3.  **Retrieval Cache**: Parallel candidate generation via Hybrid Search (Dense + Sparse).
4.  **Graph Reasoning**: Building a dynamic, query-specific graph to find character paths.
5.  **Answer Synthesis**: Symbolic phrase extraction with an LLM-gated fallback.

### Graph Component
The system builds a **Static Graph** (stored as `graph.graphml`) during indexing. During retrieval, it creates a **Dynamic Local Graph** to find "hidden" connections between query entities. This allows the system to answer questions about characters who are narratively connected but don't appear in the same text chunk.

### Example
Below is the structural heart of *"The Dark Frigate"*, showing character hubs (pink) and event milestones (blue):
![Static Graph Example](./static_graph_visualization.png)

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage Instructions

#### 1. Indexing a New Book
To build the indexes and knowledge graph for a raw text:
```bash
PYTHONPATH=. ./venv/bin/python3 src/indexing/faiss_builder.py
PYTHONPATH=. ./venv/bin/python3 src/graph/builder.py
```

#### 2. Interactive QA
Run a single question through the full graph-aware pipeline:
```bash
PYTHONPATH=. ./venv/bin/python3 tests/interactive_story_test.py
```

#### 3. Scaling Evaluation (Baseline)
To run the full benchmark across retrieval variants:
```bash
PYTHONPATH=. ./venv/bin/python3 tests/run_all_variants.py
```

### Main Results
Our proposed pipeline was evaluated across 706 books and 17,903 questions from the NarrativeQA Gutenberg setting.

| Model | Exact Match (EM) | F1 Score | ROUGE-L |
| :--- | :--- | :--- | :--- |
| **Our Model**  | **2.19** | **4.54** | **4.51** |

---

#### Per-Question-Type Breakdown
| Type | EM | F1 | Questions | Correct |
| :--- | :--- | :--- | :--- | :--- |
| **where** | 2.89 | 4.51 | 1139 | 33 |
| **who** | 2.26 | 5.08 | 2497 | 56 |
| **what** | 2.32 | 4.88 | 3955 | 92 |
| **other** | 2.48 | 4.60 | 8537 | 212 |
| **why** | 0.00 | 2.83 | 1616 | 0 |

---

#### Reasoning Path Effectiveness
| Reasoning Path | EM | F1 | Questions |
| :--- | :--- | :--- | :--- |
| **LLM generation (phrase)** | 3.91 | 7.73 | 8302 |
| **Location scored** | 5.71 | 7.52 | 165 |
| **Who-relation scored** | 2.05 | 4.55 | 2867 |
| **Symbolic miss** | 0.00 | 0.46 | 5873 |

---

### Ablation Study
We evaluated the contribution of each component by progressively enabling retrieval and reasoning modules. 

| Variant | EM | F1 | ROUGE-L | R@5 | R@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BM25 Only | 2.70 | 3.99 | 3.94 | 1.20 | 1.20 |
| Dense Only | 2.47 | 3.99 | 3.96 | 1.05 | 1.05 |
| Hybrid | 2.90 | 4.23 | 4.18 | 1.20 | 1.20 |
| + MMR | 2.60 | 3.89 | 3.84 | 1.20 | 1.20 |
| **+ Graph** | **2.90** | **4.31** | **4.26** | **1.20** | **1.20** |

---

### LLM Comparison
Comparison with strong generative models using the same retrieved context for fairness (scaled to 100 books).

| Model | F1 Score | ROUGE-L |
| :--- | :--- | :--- |
| **Our Model**  | **4.54** | **4.51** |
| LLaMA 3.1  | 0.12 | 0.11 |
| DeepSeek  | 0.14 | 0.13 |

---

### Project Structure
*   `src/nlp/`: NER, Coreference, and Entity Filtering.
*   `src/graph/`: Knowledge graph construction and reasoning algorithms.
*   `src/indexing/`: FAISS, BM25, and entity-to-chunk indexers.
*   `src/retrieval/`: The core engine: `hybrid_search.py`, `expansion.py`, and `answer.py`.
*   `src/config.py`: Global paths, model settings, and OpenRouter API config.


