#!/usr/bin/env python3
# ============================================================
# Summary script for all LLM scores
# Processes all llm_scores_*.json files and generates comparison
# ============================================================

import json
import numpy as np

# List of llm_scores files to process
llm_files = [
    "llm_scores_vector_rewrite_True.json",
    "llm_scores_vector_rewrite_False.json",
    "llm_scores_hybrid_rewrite_True.json",
    "llm_scores_hybrid_rewrite_False.json",
    "llm_scores_hybrid_rerank_rewrite_True.json",
    "llm_scores_hybrid_rerank_rewrite_False.json"
]

# Initialize results dictionary
results = {}

# Process each llm_scores file
for llm_file in llm_files:
    try:
        # Load file
        with open(llm_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract scores
        scores = {k: [] for k in ["correctness", "completeness", "safety", "helpfulness"]}
        
        for item in data:
            if "llm_score" in item:
                score = item["llm_score"]
                for k in scores:
                    if k in score:
                        scores[k].append(score[k])
        
        # Calculate mean scores
        mean_scores = {}
        for k in scores:
            if scores[k]:
                mean_scores[k] = np.mean(scores[k])
            else:
                mean_scores[k] = 0
        
        # Add to results
        results[llm_file] = mean_scores
        
    except Exception as e:
        print(f"Error processing {llm_file}: {e}")

# Print comparison table
print("=" * 120)
print(f"{'File':<60} {'Correctness':<12} {'Completeness':<12} {'Safety':<12} {'Helpfulness':<12}")
print("=" * 120)

for llm_file, mean_scores in results.items():
    file_name = llm_file.replace("llm_scores_", "").replace(".json", "")
    print(f"{file_name:<60} {mean_scores.get('correctness', 0):<12.4f} {mean_scores.get('completeness', 0):<12.4f} {mean_scores.get('safety', 0):<12.4f} {mean_scores.get('helpfulness', 0):<12.4f}")

print("=" * 120)

# Save summary to file
with open("llm_scores_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nSummary saved to llm_scores_summary.json")
