from datasets import load_from_disk, concatenate_datasets
import random
import re

# 加载本地数据集
dataset = load_from_disk("./EYE_QA_PLUS_LOCAL")

# 合并train + test（修复bug）
all_data = concatenate_datasets([dataset["train"], dataset["test"]])

# 眼底病相关关键词（中英文）
keywords = [
    "视网膜", "近视", "眼底", "retina", "myopia", "fundus", 
    "变性", "degeneration", "青光眼", "glaucoma", "白内障", 
    "cataract", "黄斑", "macular", "糖尿病", "diabetic", 
    "治疗", "treatment", "症状", "symptom", "诊断", "diagnosis",
    "预防", "prevention", "病因", "cause", "手术", "surgery"
]

def is_relevant(sample):
    """检查样本是否与眼底病相关"""
    text = sample["output"].lower()
    return any(kw.lower() in text for kw in keywords)

def is_high_quality(sample):
    """检查样本质量"""
    output = sample["output"]
    # 1. 长度适中（100-1000字符）
    if len(output) < 100 or len(output) > 1000:
        return False
    # 2. 包含医学关键词
    if not is_relevant(sample):
        return False
    # 3. 包含完整句子结构
    if not re.search(r'[.!?。！？]', output):
        return False
    return True

# 筛选高质量相关样本
high_quality = [s for s in all_data if is_high_quality(s)]

# 去重（基于内容）
unique_outputs = []
seen_texts = set()
for sample in high_quality:
    text = sample["output"].strip()
    if text not in seen_texts:
        unique_outputs.append(text)
        seen_texts.add(text)

print(f"从 {len(all_data)} 条原始数据中筛选到 {len(unique_outputs)} 条高质量唯一知识片段")

# 随机取4000条（如果少于4000就取全部）
selected = random.sample(unique_outputs, min(4000, len(unique_outputs)))

# 保存为TXT（每段知识后加空行，便于阅读和RAG切分）
with open("fundus_rag.txt", "w", encoding="utf-8") as f:
    for text in selected:
        f.write(text.strip() + "\n\n")

print(f"✅ RAG知识库保存到 fundus_rag.txt！共 {len(selected)} 条知识片段")
