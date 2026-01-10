from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("正在加载 fundus_rag.txt...")
loader = TextLoader("fundus_rag.txt", encoding="utf-8")
documents = loader.load()

print(f"加载完成，共 {len(documents)} 段文本")

print("正在切分文本...")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"切分完成，共 {len(texts)} 个chunk")

print("正在加载嵌入模型（第一次会下载 ~100MB，稍慢）...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("正在构建 FAISS 向量库...")
db = FAISS.from_documents(texts, embeddings)
db.save_local("./fundus_faiss")

print("✅ RAG 向量库构建完成！保存到 ./fundus_faiss 文件夹")
print("你可以开始运行 qa_system.py 测试问答了！")

# PEP8：文件末尾加空行