#!/usr/bin/env python3
# ============================================================
# Comprehensive summary script for all evaluation metrics
# Combines ROUGE, BERTScore, and LLM judge scores
# ============================================================

import json
import numpy as np

# List of eval files to process
eval_files = [
    "eval_vector_rewrite_True.json",
    "eval_vector_rewrite_False.json",
    "eval_hybrid_rewrite_True.json",
    "eval_hybrid_rewrite_False.json",
    "eval_hybrid_rerank_rewrite_True.json",
    "eval_hybrid_rerank_rewrite_False.json"
]

# Initialize results dictionary
results = {}

# Process each eval file
for eval_file in eval_files:
    try:
        # Load eval file to get ROUGE, BERTScore, etc.
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        # Extract summary metrics
        summary = eval_data.get("summary", {})
        rouge_l = summary.get("rouge_l", 0)
        bert_f1 = summary.get("bert_f1", 0)
        avg_response_time = summary.get("avg_response_time", 0)
        keyword_coverage = summary.get("keyword_coverage", 0)
        checklist_coverage = summary.get("checklist_coverage", 0)
        
        # Generate corresponding llm_scores file name
        llm_file = f"llm_scores_{eval_file.replace('eval_', '').replace('.json', '')}.json"
        
        # Load llm_scores file
        with open(llm_file, "r", encoding="utf-8") as f:
            llm_data = json.load(f)
        
        # Calculate mean LLM scores
        llm_scores = {"correctness": [], "completeness": [], "safety": [], "helpfulness": []}
        for item in llm_data:
            if "llm_score" in item:
                score = item["llm_score"]
                for k in llm_scores:
                    if k in score:
                        llm_scores[k].append(score[k])
        
        mean_llm_scores = {}
        for k in llm_scores:
            if llm_scores[k]:
                mean_llm_scores[k] = np.mean(llm_scores[k])
            else:
                mean_llm_scores[k] = 0
        
        # Add to results
        results[eval_file] = {
            "rouge_l": rouge_l,
            "bert_f1": bert_f1,
            "avg_response_time": avg_response_time,
            "keyword_coverage": keyword_coverage,
            "checklist_coverage": checklist_coverage,
            **mean_llm_scores
        }
        
    except Exception as e:
        print(f"Error processing {eval_file}: {e}")

# Print comprehensive comparison table
print("=" * 160)
print(f"{'File':<60} {'ROUGE-L':<10} {'BERT-F1':<10} {'Time (s)':<10} {'Keyword':<10} {'Checklist':<10} {'Correctness':<12} {'Completeness':<12} {'Safety':<12} {'Helpfulness':<12}")
print("=" * 160)

for eval_file, metrics in results.items():
    file_name = eval_file.replace("eval_", "").replace(".json", "")
    print(f"{file_name:<60} {metrics.get('rouge_l', 0):<10.4f} {metrics.get('bert_f1', 0):<10.4f} {metrics.get('avg_response_time', 0):<10.2f} {metrics.get('keyword_coverage', 0):<10.4f} {metrics.get('checklist_coverage', 0):<10.4f} {metrics.get('correctness', 0):<12.4f} {metrics.get('completeness', 0):<12.4f} {metrics.get('safety', 0):<12.4f} {metrics.get('helpfulness', 0):<12.4f}")

print("=" * 160)

# Save comprehensive summary to file
with open("comprehensive_scores_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nComprehensive summary saved to comprehensive_scores_summary.json")
