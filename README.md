# 眼底病智能问答系统（低资源医疗 RAG + LoRA）

基于 Qwen2.5-7B-Instruct 构建的眼底病垂直领域智能问答系统，采用 RAG 架构与 LoRA 微调技术，支持中英文问答。

## 项目特点

- ✅ **完全离线**：本地部署，无需联网
- ✅ **跨语言支持**：MarianMT 中英互译
- ✅ **混合检索**：向量检索 + BM25 + Cross-Encoder 重排序
- ✅ **防幻觉机制**：多层防护确保医学安全性
- ✅ **完整评测**：ROUGE、BERTScore、LLM-as-a-Judge

## 技术栈

- Python | PyTorch | Transformers
- LoRA | FAISS | BM25
- MarianMT | Cross-Encoder
- Sentence-Transformers

## 项目结构

```
low-resource-fundus-qa-master/
├── qa_system.py              # QA 系统核心实现
├── finetune.py              # LoRA 微调脚本
├── translator.py            # 中英翻译模块
├── evaluate.py              # 评测脚本
├── build_rag.py             # 构建 RAG 知识库
├── download_dataset.py       # 数据集下载
├── extract_rag.py           # 提取 RAG 数据
├── extract_sft.py           # 提取 SFT 数据
├── llm_judge_offline.py    # 离线 LLM 评测
├── generation.py            # 流式生成
└── requirements.txt         # 依赖列表
```

## 核心功能

### 1. LoRA 微调
- 数据集：QIAIUNCC/EYE-QA-PLUS（33k+ 样本）
- 参数：r=8, lora_alpha=32
- 模型：Qwen2.5-7B-Instruct

### 2. 混合检索
- 向量检索：FAISS + all-MiniLM-L6-v2
- BM25：关键词匹配
- 融合策略：线性加权（alpha=0.7）
- 重排序：Cross-Encoder (ms-marco-MiniLM-L-6-v2)

### 3. 防幻觉机制
- Prompt 约束：禁止诊断与处方
- 确定性生成：temperature=0.0
- RAG 知识增强：基于医学文献生成
- 后处理：医疗免责声明

### 4. 查询改写
- 移除礼貌用语
- 标准化医学术语
- 提升检索准确性

## 评测结果

### 检索策略消融实验

| 配置 | ROUGE-L | BERTScore F1 | Checklist | Keyword |
|------|----------|--------------|-----------|----------|
| vector | 0.1802 | 0.8655 | 0.5250 | 0.1036 |
| hybrid (alpha=0.5) | 0.1780 | 0.8660 | 0.5417 | 0.0970 |
| hybrid (alpha=0.7) | 0.1778 | 0.8655 | 0.5250 | 0.0973 |
| hybrid (alpha=0.8) | 0.1750 | 0.8650 | 0.5417 | 0.0910 |
| vector_rerank | 0.1850 | 0.8680 | 0.5833 | 0.1110 |
| hybrid_rerank | 0.1850 | 0.8670 | 0.5667 | 0.1098 |

### 混合检索参数调优

| alpha | ROUGE-L | BERTScore F1 | 性能趋势 |
|-------|---------|--------------|----------|
| 0.5 | 0.1780 | 0.8660 | **最佳** |
| 0.7 | 0.1778 | 0.8655 | 下降 |
| 0.8 | 0.1750 | 0.8650 | 继续下降 |

### LLM 评测（hybrid_rerank）

- Correctness: 4.75 / 5.00
- Completeness: 5.00 / 5.00
- Safety: 4.70 / 5.00
- Helpfulness: 4.35 / 5.00

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 下载基础模型
pip install modelscope
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct')

# 下载 LoRA 权重（需要自己训练）
# 或使用预训练权重（如果提供）
```

### 3. 运行系统

```python
from qa_system import EyeQASystem

# 初始化系统
qa_system = EyeQASystem(
    retrieval_mode="hybrid_rerank",
    use_query_rewrite=True
)

# 回答问题
answer = qa_system.answer("What should I do if my eyes feel dry?")
print(answer)
```

## 消融实验

```bash
# 运行所有配置的评测
python run_all_experiments.py

# 生成综合报告
python generate_comprehensive_summary.py
```

## 注意事项

- ⚠️ 本项目需要 GPU（推荐 16GB+ 显存）
- ⚠️ 模型文件较大，首次运行需要下载
- ⚠️ 医疗回答仅供参考，不能替代专业医疗建议

## 许可证

MIT License

## 联系方式

- 项目链接：[GitHub Repository URL]
- 问题反馈：[Issues]

## 致谢

- Qwen 团队提供的基础模型
- Hugging Face 提供的数据集和工具
- 开源社区的支持
