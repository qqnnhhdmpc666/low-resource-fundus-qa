#!/usr/bin/env python3
"""
按QA对构建RAG向量库
每个chunk是一个完整的QA对，保持语义完整性
"""

import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def load_qa_data(file_path):
    """加载QA数据，支持jsonl格式"""
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data.get('question', data.get('input', ''))
                answer = data.get('answer', data.get('output', ''))
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
            except json.JSONDecodeError:
                continue
    return qa_pairs


def create_qa_documents(qa_pairs):
    """将QA对转换为Document对象，每个QA对作为一个chunk"""
    documents = []
    for idx, qa in enumerate(qa_pairs):
        # 将QA对组合成文本
        qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
        
        # 创建Document对象，保留元数据
        doc = Document(
            page_content=qa_text,
            metadata={
                'question': qa['question'],
                'answer': qa['answer'],
                'qa_id': idx,
                'source': 'fundus_qa_dataset'
            }
        )
        documents.append(doc)
    
    return documents


def build_qa_vectorstore(documents, output_path="./fundus_faiss_qa"):
    """构建QA向量库"""
    print(f"共 {len(documents)} 个QA对")
    
    # 加载嵌入模型
    print("正在加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    # 构建向量库
    print("正在构建FAISS向量库...")
    db = FAISS.from_documents(documents, embeddings)
    
    # 保存向量库
    db.save_local(output_path)
    print(f"✅ QA向量库构建完成！保存到 {output_path}")
    
    return db


def test_retrieval(db, query, k=3):
    """测试检索效果"""
    print(f"\n测试查询: {query}")
    results = db.similarity_search(query, k=k)
    
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"相似度分数: {doc.metadata.get('score', 'N/A')}")
        print(f"内容: {doc.page_content[:200]}...")


if __name__ == "__main__":
    # 配置
    QA_FILE = "fundus_finetune.jsonl"  # QA数据文件
    OUTPUT_PATH = "./fundus_faiss_qa"
    
    # 1. 加载QA数据
    print(f"正在加载QA数据: {QA_FILE}")
    try:
        qa_pairs = load_qa_data(QA_FILE)
        print(f"加载完成，共 {len(qa_pairs)} 个QA对")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {QA_FILE}")
        print("请确保QA数据文件存在，格式为jsonl:")
        print('{"question": "...", "answer": "..."}')
        exit(1)
    
    # 2. 创建Document对象（每个QA对作为一个chunk）
    print("正在创建QA文档...")
    documents = create_qa_documents(qa_pairs)
    
    # 3. 构建向量库
    db = build_qa_vectorstore(documents, OUTPUT_PATH)
    
    # 4. 测试检索
    test_queries = [
        "什么是糖尿病视网膜病变？",
        "如何预防近视？",
        "眼睛突然看不清怎么办？"
    ]
    
    print("\n" + "="*50)
    print("检索测试")
    print("="*50)
    
    for query in test_queries:
        test_retrieval(db, query, k=2)
    
    print("\n✅ 全部完成！")
