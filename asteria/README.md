# Asteria — Modular Python Package

Converted from `asteria.ipynb` into clean, importable `.py` files with
documented inputs and outputs for integration.

## Project Structure

```
asteria/
├── __init__.py             # Public API exports
├── config.py               # AsteriaConfig dataclass — all tuneable params
├── semantic_element.py     # SemanticElement — the cache unit
├── embedding_model.py      # EmbeddingModel — Qwen3-Embedding-0.6B
├── semantic_judger.py      # SemanticJudger — Qwen3-Reranker-0.6B
├── sine_index.py           # SineIndex — two-stage ANN + judger retrieval
├── cache.py                # AsteriaCache, ExactMatchCache, LRU/LFU variants
├── workload.py             # QA knowledge base + workload generators
└── experiments.py          # run_experiment, check_accuracy
run_all_experiments.py      # Reproduces all 8 notebook experiments
```

## Install

```bash
pip install sentence-transformers faiss-cpu transformers torch numpy pandas tqdm
```

## Module Input/Output Reference

### `config.py` → `AsteriaConfig`
No inputs. Provides all defaults:
```
tau_sim=0.75, tau_lsm=0.80, cache_capacity=50, markov_theta=0.30,
remote_latency_ms=350, remote_cost_per_call=0.005, embedding_dim=1024
```

### `semantic_element.py` → `SemanticElement`
```
Input (constructor):
    query:       str
    answer:      str
    embedding:   np.ndarray (1024,)
    cost:        float      [default 0.005]
    latency_ms:  float      [default 350]
    staticity:   float      [default 5.0, range 1-10]

Output (properties):
    .is_expired  → bool
    .lcfu_score  → float   (higher = more valuable, less likely evicted)
    .summary()   → dict
```

### `embedding_model.py` → `EmbeddingModel`
```
Input:
    encode(texts: List[str])     → np.ndarray (N, 1024) float32 L2-normalised
    encode_one(text: str)        → np.ndarray (1024,) float32 L2-normalised

Output:
    Dense vectors where inner product = cosine similarity.
```

### `semantic_judger.py` → `SemanticJudger`
```
Role 1 — Relevance (query time):
    Input:  score(new_query: str, cached_answer: str)
    Output: float [0, 1]  — P(cached answer correctly answers new query)

    Input:  score_batch(pairs: List[Tuple[str, str]])
    Output: List[float]   — batch of scores

Role 2 — Staticity (insertion time):
    Input:  staticity_score(query: str, answer: str)
    Output: float [1, 10] — how time-invariant the answer is
```

### `sine_index.py` → `SineIndex`
```
Input:
    add(se: SemanticElement)     → int (faiss_id)
    remove(se: SemanticElement)  → None
    rebuild(ses: List[SE])       → None

    lookup(query: str, query_vec: np.ndarray, ann_only: bool = False)
    
Output:
    → (matched_se: SemanticElement | None, debug: dict)
    
    debug keys:
        ann_candidates: int      — how many passed τ_sim
        judger_scores:  list     — confidence scores from judger
        hit:            bool     — whether a match was confirmed
```

### `cache.py` → `AsteriaCache`
```
Input:
    lookup(query: str, ann_only: bool = False)
    → (cached_answer: str | None, debug: dict)
    
    debug keys:
        hit:                bool
        source:             "prefetch" | "sine"
        cache_lookup_ms:    float
        prefetch_triggered: str | None
        ann_candidates:     int
        judger_scores:      list

    insert(query: str, answer: str, cost: float, latency_ms: float)
    → None  (volatile data auto-discarded based on staticity)

    stats_summary()
    → dict with cache_hits, cache_misses, hit_rate_%, api_calls, api_cost_$, ses_in_cache
```

### `cache.py` → `ExactMatchCache`
```
Input:
    lookup(query: str) → str | None
    insert(query: str, answer: str) → None
```

### `experiments.py` → `run_experiment`
```
Input:
    workload:  List[Tuple[query, answer, staticity]]
    cache:     AsteriaCache | ExactMatchCache | None
    mode:      "asteria" | "exact" | "vanilla" | "ann_only"

Output: dict
    mode, n, hit_rate (float 0-1), hits, misses,
    total_cost_$ (float), avg_latency_ms, p50_latency_ms, p95_latency_ms,
    latencies (list), hit_mask (list[bool]), answers (list[str])
```

### `experiments.py` → `check_accuracy`
```
Input:   workload (same format), served_answers: List[str]
Output:  float [0, 1] — fraction of correct answers
```

## Integration Example

```python
from asteria import EmbeddingModel, SemanticJudger, AsteriaCache

emb = EmbeddingModel()
judger = SemanticJudger()
cache = AsteriaCache(emb, judger, capacity=100, tau_sim=0.75, tau_lsm=0.80)

# On incoming query:
answer, debug = cache.lookup("Who painted the Mona Lisa?")

if answer is not None:
    print(f"Cache HIT: {answer}")
    print(f"  Judger confidence: {debug.get('judger_scores')}")
    print(f"  Lookup latency: {debug['cache_lookup_ms']:.1f}ms")
else:
    # Call real API
    real_answer = call_remote_api("Who painted the Mona Lisa?")
    cache.insert("Who painted the Mona Lisa?", real_answer,
                 cost=0.005, latency_ms=320)
    print(f"Cache MISS → API: {real_answer}")

print(cache.stats_summary())
```

## Integration with AssetOpsBench IoT Agent

```python
from asteria import EmbeddingModel, SemanticJudger, AsteriaCache

emb = EmbeddingModel()
judger = SemanticJudger()
cache = AsteriaCache(emb, judger, capacity=500)

def cached_iot_agent(request: str) -> dict:
    # 1. Check cache
    answer, debug = cache.lookup(request)
    if answer is not None:
        return {"answer": answer, "review": {"status": "Accomplished"}, 
                "summary": "Served from Asteria cache"}
    
    # 2. Miss → call real IoT agent
    response = real_iot_agent_stub(request)
    
    # 3. Cache the result
    cache.insert(request, response["answer"], cost=0.005, latency_ms=300)
    
    return response
```

## Running All Experiments

```bash
python run_all_experiments.py
```

Reproduces all 8 experiments from the notebook:
1. Zipfian workload comparison (vanilla vs exact vs ANN-only vs Asteria)
2. Rolling hit rate over time
3. Bursty workload
4. Eviction policies (LRU vs LFU vs LCFU)
5. Answer accuracy
6. Threshold sensitivity sweep
7. Markov prefetching
8. Cache size ratio sweep
