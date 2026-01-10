from datasets import load_from_disk, concatenate_datasets  # 加concatenate_datasets
import random

# 加载本地数据集
dataset = load_from_disk("./EYE_QA_PLUS_LOCAL")

# 合并train + test（修复bug）
all_data = concatenate_datasets([dataset["train"], dataset["test"]])

# 提取output作为知识片段，优先选长度>200字符的（更完整）
all_outputs = [s["output"] for s in all_data if len(s["output"]) > 200]

# 随机取200条（如果少于200就取全部）
selected = random.sample(all_outputs, min(200, len(all_outputs)))

# 保存为TXT（每段知识后加空行，便于阅读和RAG切分）
with open("fundus_rag.txt", "w", encoding="utf-8") as f:
    for text in selected:
        f.write(text.strip() + "\n\n")

print(f"✅ RAG知识库保存到 fundus_rag.txt！共 {len(selected)} 条知识片段")