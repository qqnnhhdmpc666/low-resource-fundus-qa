from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json

print("正在加载结构化QA知识库 fundus_rag.json...")
with open("fundus_rag.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

print(f"加载完成，共 {len(qa_pairs)} 个QA对")

print("正在构建Document对象（每个QA对作为一个独立chunk）...")
texts = []
for idx, qa in enumerate(qa_pairs):
    # 将QA对组合成一个完整的知识片段
    page_content = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
    metadata = {
        "source": "EYE-QA-PLUS",
        "qa_id": idx,
        "question": qa["question"]
    }
    texts.append(Document(page_content=page_content, metadata=metadata))

print(f"构建完成，共 {len(texts)} 个chunk（每个chunk对应一个QA对）")

print("正在加载嵌入模型（第一次会下载 ~100MB，稍慢）...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("正在构建 FAISS 向量库...")
db = FAISS.from_documents(texts, embeddings)
db.save_local("./fundus_faiss")

print("✅ RAG 向量库构建完成！保存到 ./fundus_faiss 文件夹")
print("你可以开始运行 qa_system.py 测试问答了！")