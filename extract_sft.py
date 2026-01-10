from datasets import load_from_disk, concatenate_datasets  # 加concatenate
import json
import re

# 加载本地数据集
dataset = load_from_disk("./EYE_QA_PLUS_LOCAL")

# 关键词（眼底相关，调整加词）
keywords = ["视网膜", "近视", "眼底", "retina", "myopia", "fundus", "变性", "degeneration"]

# 筛选函数
def is_relevant(sample):
    text = sample["input"] + " " + sample["output"]
    return any(re.search(kw, text, re.IGNORECASE) for kw in keywords)

# 合并train + test（用concatenate_datasets修bug）
all_data = concatenate_datasets([dataset["train"], dataset["test"]])

# 筛选，取前500条相关
filtered = [s for s in all_data if is_relevant(s)][:500]

# 打印数量
print(f"筛选到 {len(filtered)} 条相关样本（关键词: {keywords}）")

# 保存JSONL
with open("fundus_finetune.jsonl", "w", encoding="utf-8") as f:
    for sample in filtered:
        record = {"question": sample["input"], "answer": sample["output"]}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ SFT数据保存到 fundus_finetune.jsonl！")