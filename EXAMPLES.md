# Example Usage

This directory contains example scripts for using the fundus QA system.

## Basic Usage

```python
from qa_system import EyeQASystem

# Initialize the QA system
qa_system = EyeQASystem(
    retrieval_mode="hybrid_rerank",  # Options: vector, hybrid, hybrid_rerank
    use_query_rewrite=True,          # Enable/disable query rewriting
    stream_output=False                # Enable/disable streaming output
)

# Ask a question
question = "What should I do if my eyes feel dry after working on a computer?"
answer = qa_system.answer(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Retrieval Modes Comparison

```python
from qa_system import EyeQASystem

# Compare different retrieval modes
modes = ["vector", "hybrid", "hybrid_rerank"]
question = "What are the signs of cataracts?"

for mode in modes:
    qa_system = EyeQASystem(retrieval_mode=mode, use_query_rewrite=True)
    answer = qa_system.answer(question)
    print(f"\n{mode.upper()}:")
    print(answer)
```

## Batch Processing

```python
from qa_system import EyeQASystem
import json

# Load test questions
with open("test_set_en.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Initialize system
qa_system = EyeQASystem(retrieval_mode="hybrid_rerank", use_query_rewrite=True)

# Process batch
results = []
for item in test_data:
    question = item["question"]
    answer = qa_system.answer(question)
    results.append({
        "question": question,
        "answer": answer
    })

# Save results
with open("batch_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Processed {len(results)} questions")
```

## Chinese Questions

```python
from qa_system import EyeQASystem

# Initialize system
qa_system = EyeQASystem(retrieval_mode="hybrid_rerank", use_query_rewrite=True)

# Ask a Chinese question
question = "干眼症怎么治疗？"
answer = qa_system.answer(question)

print(f"问题: {question}")
print(f"回答: {answer}")
```

## Evaluation

```bash
# Run evaluation on test set
python evaluate.py --retrieval_mode hybrid_rerank --use_query_rewrite True --test_file test_set_en.json

# Run all ablation experiments
python run_all_experiments.py
```
