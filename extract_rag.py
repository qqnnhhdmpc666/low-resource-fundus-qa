from datasets import load_from_disk, concatenate_datasets
import random
import json

# 加载本地数据集
dataset = load_from_disk("./EYE_QA_PLUS_LOCAL")

# 合并train + test
all_data = concatenate_datasets([dataset["train"], dataset["test"]])

# 提取完整QA对作为知识片段，优先选答案长度>200字符的（更完整）
qa_pairs = []
for item in all_data:
    if len(item["output"]) > 200:
        qa_pairs.append({
            "question": item["input"],
            "answer": item["output"]
        })

# 随机取5000条（如果少于5000就取全部）
selected_count = min(5000, len(qa_pairs))
selected_qa = random.sample(qa_pairs, selected_count)

# 保存为JSON格式的知识库（结构化QA）
with open("fundus_rag.json", "w", encoding="utf-8") as f:
    json.dump(selected_qa, f, ensure_ascii=False, indent=2)

# 同时也保存为TXT格式，每个QA对一行，便于阅读
with open("fundus_rag.txt", "w", encoding="utf-8") as f:
    for qa in selected_qa:
        f.write(f"Q: {qa['question'].strip()}\n")
        f.write(f"A: {qa['answer'].strip()}\n\n")

print(f"✅ RAG知识库保存完成！")
print(f"   - fundus_rag.json (结构化QA): {selected_count} 条")
print(f"   - fundus_rag.txt (文本格式): {selected_count} 条")