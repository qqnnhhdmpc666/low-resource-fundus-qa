from datasets import load_from_disk  # 导入加载工具

# 加载本地备份（不用网）
dataset = load_from_disk("./EYE_QA_PLUS_LOCAL")  # 你的保存路径

# 打印数据集信息 + 第一条样本
print("✅ 本地加载成功！")
print("总样本数 (train + test):", len(dataset["train"]) + len(dataset["test"]))
print("第一条train样本:", dataset["train"][0])  # 检查结构