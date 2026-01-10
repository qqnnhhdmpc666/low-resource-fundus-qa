from datasets import load_dataset

print("正在加载 EYE-QA-PLUS 数据集...")
ds = load_dataset("QIAIUNCC/EYE-QA-PLUS")

print("\n数据集 splits:", list(ds.keys()))

# 打印第一个 split 的第一条数据
for split in ds.keys():
    print(f"\n{split} split 第一条数据：")
    print(ds[split][0])
    print("\n字段名：", list(ds[split][0].keys()))
    print("\n第一条 question/answer 示例：")
    print("question:", ds[split][0].get("question", ds[split][0].get("query", ds[split][0].get("instruction", "无"))))
    print("answer:", ds[split][0].get("answer", ds[split][0].get("response", ds[split][0].get("output", "无"))))
    break