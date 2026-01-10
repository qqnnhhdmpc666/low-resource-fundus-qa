from datasets import load_dataset

print("正在下载 EYE-QA-PLUS 数据集...")
ds = load_dataset("QIAIUNCC/EYE-QA-PLUS")

rag_file = "fundus_rag.txt"

count = 0

with open(rag_file, "a", encoding="utf-8") as f:
    for split_name in ds.keys():
        print(f"处理 {split_name} split，共 {len(ds[split_name])} 条")
        for item in ds[split_name]:
            # 提取问题（input 字段里有 "Question: " 开头）
            input_text = item.get("input", "").strip()
            if input_text.startswith("Question:"):
                question = input_text[len("Question:"):].strip()
            else:
                question = input_text  # 备用

            answer = item.get("output", "").strip()

            if question and answer:
                f.write(f"问题：{question}\n")
                f.write(f"回答：{answer}\n\n")
                count += 1

print(f"✅ 成功追加 {count} 条专业眼科问答到 {rag_file}！")
print("追加完成！接下来运行：python build_rag.py 重建向量库")