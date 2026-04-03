#!/usr/bin/env python3
"""
run_all_experiments.py — reproduces all 8 experiments from the notebook.

Requires: pip install sentence-transformers faiss-cpu transformers torch
          pip install numpy pandas matplotlib seaborn tqdm

Usage:  python run_all_experiments.py
"""

import random

import numpy as np
import pandas as pd

from asteria import (
    AsteriaCache,
    AsteriaConfig,
    DEFAULT_CONFIG,
    EmbeddingModel,
    ExactMatchCache,
    LFUSemanticCache,
    LRUSemanticCache,
    SemanticJudger,
    check_accuracy,
    make_bursty_workload,
    make_sequential_workload,
    make_zipfian_workload,
    run_experiment,
)

random.seed(DEFAULT_CONFIG.seed)
np.random.seed(DEFAULT_CONFIG.seed)

# ── Load models ──────────────────────────────────────────────────────────────
print("Loading models...")
embedding_model = EmbeddingModel()
judger = SemanticJudger()

# Sanity check embeddings
v1 = embedding_model.encode_one("Who painted the Mona Lisa?")
v2 = embedding_model.encode_one("What artist created the Mona Lisa?")
v3 = embedding_model.encode_one("What is the capital of France?")
print(f"\nSimilarity (same topic):  {np.dot(v1, v2):.4f}  ← should be high (>0.85)")
print(f"Similarity (diff topic):  {np.dot(v1, v3):.4f}  ← should be low (<0.50)")


# ── Experiment 1: Zipfian workload comparison ────────────────────────────────
print("\n" + "=" * 55)
print("Experiment 1: Zipfian workload comparison")
print("=" * 55)

workload = make_zipfian_workload(n=200)

res_vanilla = run_experiment(workload, cache=None, mode="vanilla")

exact_cache = ExactMatchCache(capacity=DEFAULT_CONFIG.cache_capacity)
res_exact = run_experiment(workload, cache=exact_cache, mode="exact")

ann_cache = AsteriaCache(embedding_model, judger, capacity=DEFAULT_CONFIG.cache_capacity)
res_ann = run_experiment(workload, cache=ann_cache, mode="ann_only")

asteria_cache = AsteriaCache(embedding_model, judger, capacity=DEFAULT_CONFIG.cache_capacity)
res_asteria = run_experiment(workload, cache=asteria_cache, mode="asteria")

results = [res_vanilla, res_exact, res_ann, res_asteria]
labels = ["Vanilla", "Exact-match", "ANN-only", "Asteria (full)"]

df = pd.DataFrame([
    {
        "System": label,
        "Hit Rate %": round(r["hit_rate"] * 100, 1),
        "Avg Lat (ms)": r["avg_latency_ms"],
        "P95 Lat (ms)": r["p95_latency_ms"],
        "API Cost ($)": r["total_cost_$"],
        "API Calls": r["misses"],
    }
    for label, r in zip(labels, results)
])
print(df.to_string(index=False))


# ── Experiment 3: Bursty workload ────────────────────────────────────────────
print("\n" + "=" * 55)
print("Experiment 3: Bursty workload")
print("=" * 55)

bursty_workload = make_bursty_workload(n=200)
res_b_vanilla = run_experiment(bursty_workload, None, mode="vanilla")
ec2 = ExactMatchCache(capacity=DEFAULT_CONFIG.cache_capacity)
res_b_exact = run_experiment(bursty_workload, ec2, mode="exact")
ac2 = AsteriaCache(embedding_model, judger, capacity=DEFAULT_CONFIG.cache_capacity)
res_b_asteria = run_experiment(bursty_workload, ac2, mode="asteria")

print(f"Vanilla:     hit_rate={res_b_vanilla['hit_rate']*100:.1f}%  cost=${res_b_vanilla['total_cost_$']}")
print(f"Exact-match: hit_rate={res_b_exact['hit_rate']*100:.1f}%  cost=${res_b_exact['total_cost_$']}")
print(f"Asteria:     hit_rate={res_b_asteria['hit_rate']*100:.1f}%  cost=${res_b_asteria['total_cost_$']}")


# ── Experiment 4: Eviction policies ──────────────────────────────────────────
print("\n" + "=" * 55)
print("Experiment 4: Eviction policies (capacity=10)")
print("=" * 55)

SMALL_CAP = 10
wl4 = make_zipfian_workload(n=200)

res_lru = run_experiment(wl4, LRUSemanticCache(embedding_model, judger, capacity=SMALL_CAP), mode="asteria")
res_lfu = run_experiment(wl4, LFUSemanticCache(embedding_model, judger, capacity=SMALL_CAP), mode="asteria")
res_lcfu = run_experiment(wl4, AsteriaCache(embedding_model, judger, capacity=SMALL_CAP), mode="asteria")

eviction_df = pd.DataFrame([
    {"Policy": "LRU",  "Hit Rate %": round(res_lru["hit_rate"]*100, 1),  "Avg Lat (ms)": res_lru["avg_latency_ms"]},
    {"Policy": "LFU",  "Hit Rate %": round(res_lfu["hit_rate"]*100, 1),  "Avg Lat (ms)": res_lfu["avg_latency_ms"]},
    {"Policy": "LCFU", "Hit Rate %": round(res_lcfu["hit_rate"]*100, 1), "Avg Lat (ms)": res_lcfu["avg_latency_ms"]},
])
print(eviction_df.to_string(index=False))


# ── Experiment 5: Accuracy ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Experiment 5: Answer accuracy")
print("=" * 55)

acc_vanilla = check_accuracy(workload, res_vanilla["answers"])
acc_ann = check_accuracy(workload, res_ann["answers"])
acc_asteria = check_accuracy(workload, res_asteria["answers"])

print(f"Vanilla (no cache):   {acc_vanilla*100:.1f}%")
print(f"ANN-only (no judger): {acc_ann*100:.1f}%")
print(f"Asteria (full):       {acc_asteria*100:.1f}%")


# ── Experiment 7: Markov prefetching ─────────────────────────────────────────
print("\n" + "=" * 55)
print("Experiment 7: Markov prefetching")
print("=" * 55)

sequential_wl = make_sequential_workload(n_pairs=80)
res_with_pf = run_experiment(sequential_wl, AsteriaCache(embedding_model, judger, enable_prefetch=True), mode="asteria")
res_without_pf = run_experiment(sequential_wl, AsteriaCache(embedding_model, judger, enable_prefetch=False), mode="asteria")

print(f"With prefetching:    hit_rate={res_with_pf['hit_rate']*100:.1f}%  avg_lat={res_with_pf['avg_latency_ms']:.1f}ms")
print(f"Without prefetching: hit_rate={res_without_pf['hit_rate']*100:.1f}%  avg_lat={res_without_pf['avg_latency_ms']:.1f}ms")


# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
summary = pd.DataFrame([
    {"Experiment": "Zipfian — Vanilla",        "hit_rate": res_vanilla["hit_rate"],   "avg_lat": res_vanilla["avg_latency_ms"],   "cost": res_vanilla["total_cost_$"]},
    {"Experiment": "Zipfian — Exact-match",    "hit_rate": res_exact["hit_rate"],     "avg_lat": res_exact["avg_latency_ms"],     "cost": res_exact["total_cost_$"]},
    {"Experiment": "Zipfian — ANN-only",       "hit_rate": res_ann["hit_rate"],       "avg_lat": res_ann["avg_latency_ms"],       "cost": res_ann["total_cost_$"]},
    {"Experiment": "Zipfian — Asteria (full)", "hit_rate": res_asteria["hit_rate"],   "avg_lat": res_asteria["avg_latency_ms"],   "cost": res_asteria["total_cost_$"]},
    {"Experiment": "Bursty  — Asteria (full)", "hit_rate": res_b_asteria["hit_rate"], "avg_lat": res_b_asteria["avg_latency_ms"], "cost": res_b_asteria["total_cost_$"]},
])
summary["hit_rate"] = (summary["hit_rate"] * 100).round(1).astype(str) + "%"
print(summary.to_string(index=False))
