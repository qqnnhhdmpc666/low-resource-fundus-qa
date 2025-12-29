from datasets import load_dataset
import json

# 第一步：下载数据集（第一次需要联网）
dataset = load_dataset("QIAIUNCC/EYE-QA-PLUS")

# 第二步：保存为完整本地备份（结构化格式）
dataset.save_to_disk("./EYE_QA_PLUS_LOCAL")

# 第三步：额外导出为 JSONL（人类可读，永久兜底）
with open("EYE_QA_PLUS_full_backup.jsonl", "w", encoding="utf-8") as f:
    for sample in dataset["train"]:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("✅ 数据集已完整保存到本地！")