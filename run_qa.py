#!/usr/bin/env python3
# 运行QA系统并测试效果

import sys
import os

# 添加code目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

# 导入QA系统
from qa_system import EyeQASystem

# 初始化QA系统
print("初始化QA系统...")
qa_system = EyeQASystem(
    retrieval_mode="hybrid_rerank",
    use_query_rewrite=True,
    stream_output=False,
    alpha=0.5
)
print("QA系统初始化完成！")

# 测试问题
test_questions = [
    "如何预防近视？",
    "什么是青光眼？",
    "我有飞蚊症，是否需要立即就医？",
    "我突然视力模糊，伴有眼痛，应该怎么办？"
]

# 运行测试
print("\n开始测试QA系统...")
for i, question in enumerate(test_questions, 1):
    print(f"\n=== 测试问题 {i} ===")
    print(f"问题: {question}")
    print("生成回答中...")
    
    try:
        response = qa_system.answer(question, max_new_tokens=200)
        print("\n回答:")
        print(response["answer"])
        
        print("\n证据:")
        for j, ev in enumerate(response["evidence"], 1):
            print(f"{j}. Doc ID: {ev['doc_id']}")
            print(f"   Score: {ev['score']}")
            print(f"   Rank: {ev['rank']}")
            print(f"   Source: {ev['source']}")
            print(f"   Text: {ev['text'][:100]}...")
            print()
            
    except Exception as e:
        print(f"错误: {e}")

print("\n测试完成！")
