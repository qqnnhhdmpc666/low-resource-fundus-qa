#!/usr/bin/env python3
# ============================================================
# Test script for LLM evaluator
# Processes all eval_*.json files and generates llm_scores_*.json
# ============================================================

import os
import subprocess

# List of eval files to process
eval_files = [
    "eval_vector_rewrite_True.json",
    "eval_vector_rewrite_False.json",
    "eval_hybrid_rewrite_True.json",
    "eval_hybrid_rewrite_False.json",
    "eval_hybrid_rerank_rewrite_True.json",
    "eval_hybrid_rerank_rewrite_False.json"
]

# Process each eval file
for eval_file in eval_files:
    if os.path.exists(eval_file):
        # Generate output file name
        output_file = f"llm_scores_{eval_file.replace('eval_', '').replace('.json', '')}.json"
        
        print(f"Processing {eval_file}...")
        print(f"Output will be saved to {output_file}")
        
        # Run llm_judge_offline.py
        cmd = f"python3 llm_judge_offline.py --input {eval_file} --output {output_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Print results
        print("Return code:", result.returncode)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("=" * 80)
    else:
        print(f"File {eval_file} not found, skipping.")
        print("=" * 80)

print("All files processed!")
