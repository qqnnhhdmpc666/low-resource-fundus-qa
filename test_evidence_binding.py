#!/usr/bin/env python3
# 测试证据绑定功能

import sys
import os

# 添加code目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

# 导入QA系统
from qa_system import get_eye_qa_system

# 初始化QA系统
eye_qa = get_eye_qa_system(
    retrieval_mode="hybrid_rerank",
    use_query_rewrite=True,
    stream_output=False,
    alpha=0.5
)

# 测试问题
test_question = "如何预防近视？"

# 获取回答
response = eye_qa(test_question, max_new_tokens=100)

# 打印结果
print("=== 测试证据绑定功能 ===")
print(f"问题: {test_question}")
print(f"\n回答: {response['answer']}")
print(f"\n证据:")
for i, ev in enumerate(response['evidence'], 1):
    print(f"{i}. Doc ID: {ev['doc_id']}")
    print(f"   Score: {ev['score']}")
    print(f"   Rank: {ev['rank']}")
    print(f"   Source: {ev['source']}")
    print(f"   Text: {ev['text'][:100]}...")
    print()

print("=== 测试完成 ===")
