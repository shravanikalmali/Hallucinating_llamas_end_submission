# INLP_Project_Hallucinating-Llamas


### **Phase 0: Data Preparation** (unchanged)
- Download, clean, segment, chunk. Store `segment_id`, `timeline`, text.
- Data stored in Drive : https://drive.google.com/drive/folders/1CVtwxIQON9uhSL5vciqoaNzZN7klcizs

---

### **Phase 1: Offline Indexing (Robustified)**

#### **1.1 Embeddings** (unchanged)
- `nomic-ai/nomic-embed-text-v1.5`, store in FAISS.

#### **1.2 NER + Global Entity Filtering** (Your Point 5)
- Run spaCy `en_core_web_sm` on all chunks.
- Build a **Global Entity Frequency Dictionary**: count mentions per canonical entity (after basic normalization: lowercase, strip articles).
- **Filter graph nodes**: only keep entities with frequency ≥ threshold (e.g., 5 for a 100k‑word story). This removes noise like "London" if it's just a passing mention.

#### **1.3 Coreference (Your Point 2 - MUST HAVE)**
- Use `fastcoref` on the **full story text** (not per chunk).
  - Input: entire story as one string.
  - Output: clusters of mentions (e.g., `{ "Kylo Ren", "he", "the son of Han", "the young knight" }`).
- Store mapping from each mention to a **canonical entity ID** (use the most frequent proper name in the cluster).
- During NER, replace all mentions in a chunk with their canonical ID. This is **critical** for graph construction.

#### **1.4 Static Graph Construction**
- Nodes: canonical entities (from coref) that pass the frequency filter.
- Edges:
  - **Co‑occurrence**: if two entities appear in the same segment, add an edge with weight = co‑occurrence count.
  - **Sequential**: connect the same entity across adjacent segments (to track state changes).
- Store with NetworkX.

#### **1.5 Segment‑Level Sentiment (Prep for Relation Extraction)**
- For each segment, compute a **sentiment score** using a simple model (e.g., `TextBlob` or a small transformer).
- Store segment sentiment to help with relation inference later.

---

### **Phase 2: Query‑Time Retrieval**

#### **2.1 Initial Vector Search** (unchanged)
- Embed question, retrieve top 15 chunks.

#### **2.2 Stratified MMR with "Summary‑Bias"** (Your Point 3)
- Group retrieved chunks by `segment_id`.
- For each segment with hits:
  - Compute MMR score for each chunk:
    `score = λ * sim(q, chunk) - (1-λ) * max_sim(chunk, selected)`
  - **Add bias**: multiply the score of the **first chunk of the segment** by 1.2 (or add a small constant). This boosts "topic sentences" that bridge to summary language.
  - Select top 2 per segment.
- Force‑include at least one chunk from final segment (segment_id = N-1).
- Result: ~8 diverse, summary‑aware chunks.

#### **2.3 Entity‑Based Expansion** (unchanged)
- Collect canonical entities from current chunks.
- Query static graph for other chunks (not yet selected) containing same entities but from different segments. Add up to 3.

---

### **Phase 3: Information Extraction (Your Point 1 - The Critical Fix)**

Instead of fragile verb‑pattern extraction, use **Co‑occurrence + Sentiment + Key Terms**:

#### **3.1 Coreference Resolution (Again)**
- Apply coref mapping to unify mentions within the retrieved chunks.

#### **3.2 Extract "Conflict" and "Affiliation" Relations**
For each pair of entities `(A, B)` in the same chunk:
- **Sentiment**: if chunk sentiment is strongly negative (≤ -0.3) and chunk contains death‑related words (`kill`, `die`, `blade`, `blood`), add `Conflict(A, B)` edge.
- **Affiliation**: if chunk sentiment is strongly positive (≥ 0.3) and contains alliance words (`ally`, `friend`, `help`, `save`), add `Allied(A, B)` edge.
- **Generic Co‑occurrence**: if none of the above, still add a weak `Co‑occur(A, B)` edge (confidence 0.3) for possible use in reasoning.

#### **3.3 Extract Role/State for Single Entities**
For each entity `E` in a chunk:
- Look for role‑indicating words nearby (within 5 tokens) using a small lexicon:
  - `hero`: hero, brave, noble, savior
  - `villain`: villain, evil, dark, traitor
  - `victim`: victim, innocent, captive
  - `mentor`: mentor, teacher, guide
- Assign a confidence score based on match strength.
- Store as `role(E, role_name, confidence, segment_id)`.

#### **3.4 Build Role State Vectors (Your Point 4)**
- For each entity, collect all role observations across segments.
- Create a **state vector** per segment: e.g.,
  `{ "hero": 0.9, "villain": 0.1 }` for segment 1
  `{ "hero": 0.2, "villain": 0.8 }` for segment 10
- Normalize so probabilities sum to 1 per segment (heuristic).
- Store this sequence for reasoning.

---

### **Phase 4: Dynamic Knowledge Graph Construction**

- Build a **per‑query graph** using only entities and relations from the current chunks.
- Nodes: entities with their current role vector (from the most relevant segment).
- Edges: the extracted relations (`Conflict`, `Allied`, `Co‑occur`) with confidence scores.
- Add temporal edges connecting the same entity across segments (ordered by `segment_id`).

This graph is small and query‑specific, making reasoning fast.

---

### **Phase 5: Graph‑Based Reasoning (Symbolic)**

#### **5.1 Parse Question**
- Use spaCy to extract:
  - Question type: `who`, `what`, `why`, `where`
  - Target entity (if any, e.g., "Han Solo")
  - Relation clue (e.g., "kill", "betray") – map to our relation types using a simple lexicon.

#### **5.2 Answer "Who" Questions (Entity Resolution)**
- If question asks "Who kills Han Solo?":
  - Find node "Han Solo".
  - Look for incoming `Conflict` edges from other nodes (especially with high confidence).
  - Candidates: `{ Kylo: 0.8, Darth: 0.3 }`.
  - If multiple, use **temporal rule**: prefer the entity appearing in the **latest segment** (for final outcome questions).
    For development questions (e.g., "Who tried to kill him?"), you might need to look at all segments.

#### **5.3 Answer "What/Why" Questions (Path Finding)**
- Example: "Why did Kylo turn evil?"
- Find node "Kylo".
- Traverse its state vector across segments: detect a **maximum delta** in the `villain` role.
  If `villain` jumps from 0.1 to 0.8 between segment 5 and 6, look at the chunks around segment 5‑6 for explanatory events (e.g., "Snoke corrupted him").
- Return the event description from that chunk (or a relation like `Corrupted(Snoke, Kylo)` if extracted).

#### **5.4 Role‑Conflict Resolution (Your Point 4)**
- If question asks "Is Kylo a hero or villain?":
  - Look at final segment's role vector: `villain: 0.8, hero: 0.2`.
  - Answer: "Villain" (with optional note: "He was a hero earlier, but turned evil.")
- This explicitly handles the plot twist and shows reasoning.

---

### **Phase 6: Answer Generation**

- For entity answers (who), output the canonical name directly.
- For role questions, output the dominant role from final segment.
- For why questions, output the event sentence (from the chunk around the state change).
- **No LLM needed** – all reasoning is symbolic.

If you want natural language fluency, you can use a simple template:
- `"{entity} is a {role}."`
- `"{entity} killed {target}."`
- `"{entity} changed from {old_role} to {new_role} because {event}."`

---

### **Phase 7: Answer Quality Evaluation (NEW)**

#### **7.1 Confidence Scoring**
Each answer includes a **combined confidence score** that blends model quality with reasoning structure:

```python
answer_score = 0.6 * cross_encoder_score + 0.4 * graph_confidence
```

- **cross_encoder_score** (0.0-1.0): Quality of extracted answer sentence (from deep re-ranking model)
- **graph_confidence** (0.0-1.0): Strength of graph-based reasoning (based on entity neighbors, relation types, graph size)

#### **7.2 Answer Quality Metrics**

Four complementary metrics evaluate answer correctness:

| Metric | Purpose | Range | Interpretation |
|--------|---------|-------|-----------------|
| **EM** (Exact Match) | Exact string match after normalization | [0.0, 1.0] | 1.0 = perfect match |
| **F1** | Token-level precision/recall overlap | [0.0, 1.0] | Partial credit for partial matches |
| **BLEU** | N-gram precision with brevity penalty | [0.0, 1.0] | Balances phrase-level accuracy |
| **ROUGE-L** | Longest common subsequence | [0.0, 1.0] | Sequence preservation score |

All metrics use normalized answers (lowercase, punctuation removed, articles stripped).

#### **7.3 Failure Type Classification**

When an answer is incorrect, the system classifies the failure:

- **Retrieval Failure**: Answer chunk not in top-5 retrieved candidates → core retrieval issue
- **Extraction Failure**: Correct chunk retrieved, but wrong sentence extracted (F1 < 0.3) → answer extraction issue
- **Reasoning Failure**: Other errors → reasoning/graph construction issue
- **No Failure**: Exact match → correct answer

#### **7.4 Batch Evaluation & Reporting**

Run evaluation across multiple questions to get:
- Aggregate metrics (micro-averaged EM, F1, BLEU, ROUGE-L)
- Failure analysis breakdown (% retrieval vs extraction vs reasoning)
- Per-question-type analysis (WHO vs WHERE vs WHEN vs WHY)
- Confidence statistics (mean, min, max)

Example output:
```
Total Questions: 100
Aggregate Metrics: EM: 0.52 | F1: 0.71 | BLEU: 0.58 | ROUGE-L: 0.68
Failure Analysis:
  none: 52 (52.0%)
  retrieval: 18 (18.0%)
  extraction: 22 (22.0%)
  reasoning: 8 (8.0%)
Per-Question-Type Analysis:
  who: 25 questions, EM: 0.68, F1: 0.79
  where: 25 questions, EM: 0.44, F1: 0.64
  when: 25 questions, EM: 0.48, F1: 0.72
  why: 25 questions, EM: 0.48, F1: 0.62
```

#### **7.5 Ablation Studies**

The modular answer confidence enables systematic ablation:
- Disable cross-encoder → see impact of model quality
- Lower graph confidence → see impact of reasoning
- Modify relation types → see impact of extraction quality

Each configuration produces its own evaluation report for comparison.

---

## **Why This Plan Now Works**

| Challenge | Your Fix | Implementation |
|-----------|----------|----------------|
| Literary prose kills verb patterns | Co‑occurrence + sentiment + keywords | `Conflict` edge from negative sentiment + death words |
| Pronouns break entity tracking | Coreference (`fastcoref`) | Unify mentions before graph building |
| MMR misses summary‑style questions | Summary‑bias on segment‑first chunks | Boost first chunk of each segment |
| Plot twists confuse reasoning | State vectors over time | Track role probabilities per segment |
| Noisy NER clutters graph | Frequency filtering | Only keep entities with >5 mentions |
| No QA quality measurement | Multi-metric evaluation | EM, F1, BLEU, ROUGE-L + failure classification |
| Confidence is heuristic | Model-aware scoring | 60% model + 40% reasoning structure |

---

## **Implementation Roadmap (MVP → Full)**

### **Week 1‑2: Core Pipeline**
- Data prep + chunking
- Embeddings + FAISS
- Basic retrieval (no MMR yet)
- Simple answer extraction (top sentence)

### **Week 3‑4: Add Narrative Awareness**
- NER + frequency filtering
- Coreference integration
- Stratified MMR with summary‑bias
- Entity‑based expansion

### **Week 5‑6: Add Reasoning**
- Segment‑level sentiment
- Conflict/role extraction (lexicon‑based)
- State vectors
- Graph traversal for who/what questions

### **Week 7‑8: Polish & Evaluate**
- Handle role‑conflict resolution
- Template‑based generation
- Answer confidence scoring (model + graph)
- QA metrics evaluation (EM, F1, BLEU, ROUGE-L)
- Ablation experiments
- Write report with per-question-type analysis

---

## **Evaluation & Metrics**

### **Available Modules**

**`src/evaluation.py`** – Comprehensive metrics implementation

**Answer Quality Metrics:**
- `exact_match(predicted, gold)` – EM score
- `f1_score(predicted, gold)` – Token-level F1
- `bleu_score(predicted, gold)` – BLEU with n-gram precision
- `rouge_l(predicted, gold)` – LCS-based ROUGE-L
- `compute_answer_metrics()` – All answer metrics for one QA pair
- `micro_average_metrics()` – Aggregate across questions

**Retrieval-Level Metrics:**
- `recall_at_k(retrieved_chunk_ids, gold_chunk_id, k)` – Recall@k (top-k candidate retrieval)
- `mean_reciprocal_rank(retrieved_chunk_ids, gold_chunk_id)` – MRR (position-weighted ranking)
- `ndcg_at_k(retrieved_chunk_ids, gold_chunk_id, k)` – NDCG@k (normalized ranking quality)
- `compute_retrieval_metrics()` – All retrieval metrics

**Comprehensive Evaluation:**
- `comprehensive_evaluation(qa_results, gold_chunks)` – Full evaluation with component breakdown
  - Answer metrics + retrieval metrics
  - Failure classification (retrieval/extraction/reasoning)
  - Per-question-type analysis
  - Component quality breakdown
  - Confidence calibration analysis

**`src/evaluation_integration.py`** – Evaluation harness
- `evaluate_qa_result()` – Single result with metrics + failure classification
- `batch_evaluate()` – Batch evaluation with per-type analysis
- `print_evaluation_report()` – Formatted output for simple eval
- `print_comprehensive_report()` – Formatted output for comprehensive eval (NEW)

**`METRICS_GUIDE.md`** – Detailed usage documentation and examples

### **Answer Result Format**

Each answer includes full provenance for transparency:

```python
{
    "answer": "extracted sentence",
    "answer_score": 0.75,  # Combined: 0.6*cross_encoder + 0.4*graph
    "cross_encoder_score": 0.8,  # Model quality (0.0-1.0)
    "graph_confidence": 0.65,  # Reasoning quality (0.0-1.0)
    "source_chunk_id": 42,
    "source_segment": 5,
    "context": "full chunk text",
    "reasoning": "graph_guided",  # one of: graph, graph_guided, fallback
    "graph_target": "Han Solo",  # reasoning target entity
    "graph_nodes": 15,  # size of dynamic graph
    "graph_edges": 28,
}
```

### **Example: Running Evaluation**

```python
from src.evaluation_integration import batch_evaluate, print_evaluation_report

# List of results from query pipeline
qa_results = [
    {
        "question": "Who killed Han Solo?",
        "predicted_answer": "Kylo Ren killed Han Solo.",
        "gold_answer": "Kylo Ren",
        "confidence": 0.82,
        "retrieval_rank": 1,
        "metadata": {"qtype": "who"}
    },
    # ... more results
]

report = batch_evaluate(qa_results)
print_evaluation_report(report)
```

### **Example: Comprehensive Evaluation (with retrieval metrics)**

```python
from src.evaluation_integration import comprehensive_evaluation, print_comprehensive_report

qa_results = [
    {
        "question": "Who killed Han Solo?",
        "predicted_answer": "Kylo Ren",
        "gold_answer": "Kylo Ren",
        "confidence": 0.82,
        "retrieval_rank": 1,
        "retrieved_chunk_ids": [42, 51, 63, 78, 89],  # Retrieved chunks
        "gold_chunk_id": 42,  # Chunk with answer
        "metadata": {"qtype": "who", "graph_nodes": 25, "graph_edges": 18}
    },
    # ... more results
]

report = comprehensive_evaluation(qa_results)
print_comprehensive_report(report)

# Report outputs:
# - Answer quality (EM, F1, BLEU, ROUGE-L)
# - Retrieval quality (Recall@5, Recall@10, MRR, NDCG@5, NDCG@10)
# - Component breakdown (extraction F1, reasoning success rate)
# - Failure types (retrieval vs extraction vs reasoning)
# - Confidence calibration (is model confidence well-calibrated?)
# - Per-question-type analysis (WHO vs WHERE vs WHEN vs WHY)
```

---

## **Tools Recap**
- Coreference: `fastcoref`
- NER: spaCy
- Sentiment: `TextBlob` or `transformers` pipeline
- Embeddings: `sentence-transformers`
- Vector search: FAISS
- Graph: NetworkX
- Evaluation: `src/evaluation.py`, `src/evaluation_integration.py`
- Lexicons: manually create small lists (20‑30 words each) for roles and relation triggers

This plan is **ambitious but achievable**, and every component is **explainable** – perfect for a course project. The symbolic approach means you can visualize and debug each step, and your professor will see the thoughtful engineering behind it.

The **evaluation infrastructure** enables rigorous ablation studies and provides quantitative proof of what actually works.

Go build it! And keep asking questions when you hit roadblocks.
