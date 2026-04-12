# 低资源眼底病智能问答系统（RAG + QLoRA + Type-Aware）

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 **Qwen2.5-7B-Instruct** 构建的眼底病垂直领域智能问答系统，采用 **RAG** 架构、**QLoRA** 微调技术与 **Type-Aware** 智能路由策略，支持中英文问答，适配 **A4000 16GB** 低算力环境。

[GitHub 项目地址](https://github.com/qqnnhhdmpc666/low-resource-fundus-qa)

---

## 🌟 项目亮点

- ✅ **Type-Aware 智能路由**：基于 NLI 模型自动识别问题类型，动态选择最优检索策略
- ✅ **四策略检索体系**：vector / hybrid / vector_rerank / hybrid_rerank 全面覆盖
- ✅ **混合检索优化**：α=0.5 最优权重，平衡语义相似度与关键词匹配
- ✅ **科研级评测体系**：MRR/NDCG/MAP 排名指标 + 同基座模型评测 + 统计显著性检验
- ✅ **完全离线部署**：本地部署，无需联网，保护医疗数据隐私
- ✅ **低算力适配**：A4000 16GB 显存可完整运行推理与微调
- ✅ **医疗安全保障**：多层防护机制，自动追加医疗免责声明
- ✅ **Evidence Traceability**：完整的证据溯源与可解释性支持

---

## 📊 实验结果速览

### 四种检索策略性能对比

| 策略 | ROUGE-L | BERT-F1 | Judge Score | Checklist Coverage | Avg Time(s) |
|------|---------|---------|-------------|-------------------|-------------|
| vector | 0.1813 | 0.8658 | 4.525 | 0.525 | 71.07 |
| hybrid | 0.1770 | 0.8654 | 4.520 | 0.525 | 68.94 |
| vector_rerank | 0.133 | 0.857 | 4.435 | 0.630 | 84.81 |
| **hybrid_rerank** ⭐ | **0.1850** | **0.8670** | **4.585** | **0.5667** | 78.14 |

### Type-Aware 最优策略映射

| 问题类型 | 最优策略 | Judge Score | 策略说明 |
|---------|---------|-------------|---------|
| emergency | vector | 4.667 | 纯语义检索对紧急情况更安全 |
| disease_definition | hybrid_rerank | 4.604 | 混合+重排序在定义类问题上最优 |
| daily_advice | hybrid_rerank | 4.593 | 混合+重排序在日常建议类问题上最优 |
| decision | hybrid_rerank | 4.500 | 混合+重排序在决策类问题上最优 |

---

## 📋 测试集设计（50题 / 4类问题）

本项目的评测基于**人工构建的 50 题中文测试集**，覆盖眼底病领域的典型问答场景，按问题类型分为 **4 大类**：

### 问题类型分布

| 问题类型 | 数量 | 占比 | 典型场景 | 示例 |
|---------|------|------|---------|------|
| 🟢 **daily_advice** (日常建议) | 15题 | 30% | 护眼建议、生活方式指导 | "高度近视日常要注意什么？" |
| 🔵 **disease_definition** (疾病定义) | 12题 | 24% | 病因、症状、病理机制 | "高度近视会引起视网膜脱离吗？" |
| 🟡 **decision** (决策建议) | 13题 | 26% | 治疗选择、手术决策 | "高度近视可以做激光近视手术吗？" |
| 🔴 **emergency** (紧急情况) | 10题 | 20% | 需立即处理的症状 | "飞蚊症突然多了怎么办？" |

### Type-Aware 分类依据

使用 **DeBERTa-v3-base-mnli** 模型对问题进行自动分类，分类结果用于：
- 动态选择最优检索策略
- 分析不同策略在各类型问题上的表现差异
- 为医疗安全机制提供输入（emergency 类优先安全约束）

### 50 题完整列表

#### 🟢 daily_advice — 日常建议类 (15题)

| ID | 问题 |
|----|------|
| 5 | 高度近视日常要注意什么？ |
| 6 | 孩子近视加深快该怎么办？ |
| 10 | 视疲劳怎么缓解？ |
| 12 | 高度近视需要多久检查一次眼底？ |
| 15 | 糖尿病眼底病变怎么预防？ |
| 25 | 手机用久了眼睛干涩怎么缓解？ |
| 26 | 孩子近视了要不要戴眼镜？ |
| 27 | 老年人眼底检查重要吗？ |
| 29 | 戴隐形眼镜会加重干眼吗？ |
| 31 | 晚上看手机会加深近视吗？ |
| 33 | 吃蓝莓能保护眼底吗？ |
| 35 | 眼睛经常酸胀是怎么回事？ |
| 37 | 孩子看手机时间长眼睛会疼怎么办？ |
| 39 | 眼睛干涩看电脑模糊怎么改善？ |
| 40 | 高度近视能玩VR游戏吗？ |

#### 🔵 disease_definition — 疾病定义类 (12题)

| ID | 问题 |
|----|------|
| 1 | 高度近视会引起视网膜脱离吗？ |
| 2 | 糖尿病会引起眼底出血吗？ |
| 3 | 干眼症怎么治疗？ |
| 11 | 眼底照片里有黄色斑点是什么意思？ |
| 13 | 视盘边界不清是什么问题？ |
| 14 | 近视眼底会出现什么变化？ |
| 16 | 玻璃体出血怎么办？ |
| 18 | 儿童飞蚊症严重吗？ |
| 19 | 眼底出血会失明吗？ |
| 20 | 视网膜静脉阻塞怎么治疗？ |
| 41 | 老年人看东西模糊是白内障吗？ |
| 45 | 眼睛经常流泪是怎么回事？ |

#### 🟡 decision — 决策建议类 (13题)

| ID | 问题 |
|----|------|
| 7 | 眼底激光后要注意什么？ |
| 8 | 白内障手术后视力又模糊了怎么办？ |
| 17 | 高度近视可以做激光近视手术吗？ |
| 21 | 高度近视平时可以坐过山车吗？ |
| 23 | 高度近视能打篮球吗？ |
| 28 | 开车时突然眼前黑影飘动怎么办？ |
| 30 | 高度近视能潜水吗？ |
| 34 | 高度近视能跳伞吗？ |
| 36 | 高度近视能生孩子用力吗？ |
| 38 | 高度近视能举重吗？ |
| 42 | 高度近视能开飞机吗？ |
| 44 | 高度近视能打羽毛球吗？ |
| 48 | 高度近视能戴隐形眼镜吗？ |

#### 🔴 emergency — 紧急情况类 (10题)

| ID | 问题 |
|----|------|
| 4 | 飞蚊症突然多了怎么办？ |
| 9 | 孕妇高度近视能顺产吗？ |
| 22 | 高度近视揉眼睛会引起视网膜脱离吗？ |
| 24 | 高度近视怀孕能顺产吗？ |
| 32 | 糖尿病患者多久查一次眼底？ |
| 43 | 孩子眯眼看电视是怎么回事？ |
| 46 | 高度近视能考驾照吗？ |
| 47 | 孩子写作业眼睛离本子很近怎么办？ |
| 49 | 晚上开车灯光刺眼是怎么回事？ |
| 50 | 高度近视能当兵吗？ |

> 💡 详细测试集说明请参阅 [测试集说明文档](测试集说明文档.md)

---

## 🏗️ 系统架构

### 整体流程

```
用户问题输入
    ↓
[翻译模块] MarianMT 中英互译
    ↓
[Query Rewrite] 查询改写优化
    ↓
[Type-Aware 路由] NLI 分类器识别问题类型
    ↓
[检索策略选择] vector / hybrid / hybrid_rerank
    ↓
[检索执行]
    ├── 向量检索: FAISS + all-MiniLM-L6-v2
    ├── BM25检索: rank_bm25
    └── 重排序: Cross-Encoder/ms-marco-MiniLM-L-6-v2
    ↓
[上下文构建] 最多 8 个文档
    ↓
[Prompt 组装] 医疗安全约束
    ↓
[LLM 生成] Qwen2.5-7B-Instruct + LoRA
    ↓
[后处理] 医疗免责声明追加
    ↓
[证据绑定] Evidence Traceability
    ↓
返回答案 + 证据来源
```

### 核心模块

| 模块 | 技术实现 | 说明 |
|-----|---------|------|
| **知识库构建** | FAISS + BM25 | 5000条QA对，结构化存储 |
| **向量检索** | all-MiniLM-L6-v2 | 384维语义向量 |
| **混合检索** | α=0.5 线性加权 | 平衡语义与关键词 |
| **重排序** | Cross-Encoder | ms-marco-MiniLM-L-6-v2 |
| **Type-Aware** | DeBERTa-v3-base-mnli | 4类问题分类 |
| **大模型** | Qwen2.5-7B-Instruct | 4位NF4量化 |
| **微调** | QLoRA (r=8, α=32) | 可训练参数 0.055% |
| **翻译** | MarianMT | 本地中英互译 |
| **评测** | 科研级指标体系 | MRR/NDCG/MAP + LLM Judge |

---

## 📁 项目结构

```
low-resource-fundus-qa/
│
├── 📂 核心代码
│   ├── qa_system.py                 # QA系统核心实现（检索+生成）
│   ├── question_classifier.py       # NLI问题类型分类器
│   ├── evaluator_scientific.py      # 科研级评测系统
│   ├── evaluator_enhanced.py        # 增强版评测系统
│   ├── finetune.py                  # QLoRA微调脚本
│   ├── translator.py                # MarianMT翻译模块
│   ├── evaluate.py                  # 基础评测脚本
│   ├── build_rag.py                 # 构建FAISS向量库
│   ├── extract_rag.py               # 提取RAG知识库数据
│   ├── extract_sft.py               # 提取SFT微调数据
│   └── download_dataset.py          # 数据集下载
│
├── 📂 评测与实验
│   ├── run_all_experiments.py       # 批量实验脚本
│   ├── generate_comprehensive_summary.py  # 综合报告生成
│   ├── summary_all_llm_scores.py    # LLM评分汇总
│   ├── llm_judge_new.py             # LLM-as-a-Judge评测
│   └── llm_judge_offline.py         # 离线LLM评测
│
├── 📂 数据文件
│   ├── fundus_rag.json              # 结构化QA知识库（5000条）
│   ├── fundus_rag.txt               # 文本格式知识库
│   ├── fundus_finetune.jsonl        # SFT微调数据（500条）
│   ├── test_set.json                # 测试集（50题，4类问题）
│   └── text_set_hard.json           # 英文扩展测试集
│
├── 📂 实验结果
│   ├── comprehensive_scores_summary.json      # 综合评分汇总
│   ├── llm_scores_*.json                      # 各配置LLM评分
│   ├── eval_*.json                            # 各配置评测结果
│   ├── 论文实验数据汇总.md                     # 实验数据整理
│   └── 测试集说明文档.md                       # 测试集详细说明（含50题分类）
│
├── 📂 模型文件（运行时生成）
│   ├── fundus_lora/                 # LoRA微调权重
│   └── fundus_faiss/                # FAISS向量索引
│
├── 📄 文档
│   ├── README.md                    # 项目说明（本文件）
│   ├── 技术实现细节说明书.md         # 详细技术文档
│   ├── EXAMPLES.md                  # 使用示例
│   ├── CONTRIBUTING.md              # 贡献指南
│   └── requirements.txt             # Python依赖
│
└── 📂 其他
    ├── run_qa.py                    # QA系统运行入口
    ├── generation.py                # 流式生成（预留）
    └── test_evidence_binding.py     # 证据绑定测试
```

---

## 🚀 快速开始

### 环境要求

- **GPU**: NVIDIA A4000 16GB 或同等算力（RTX 3090/4090）
- **Python**: 3.10+
- **CUDA**: 11.8+
- **内存**: 32GB+
- **存储**: 100GB+ SSD

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/qqnnhhdmpc666/low-resource-fundus-qa.git
cd low-resource-fundus-qa

# 创建虚拟环境
python -m venv fundus_env
source fundus_env/bin/activate  # Linux/Mac
# 或 fundus_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 下载数据集
python download_dataset.py

# 提取RAG知识库（5000条QA对）
python extract_rag.py

# 提取SFT微调数据（500条）
python extract_sft.py

# 构建FAISS向量索引
python build_rag.py
```

### 3. 模型准备

```bash
# 方式1：使用Hugging Face下载（需要联网）
# 模型会自动下载到本地缓存

# 方式2：使用ModelScope镜像（国内推荐）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"

# 方式3：手动下载后放入 hf_cache/ 目录
```

### 4. 运行QA系统

```python
from qa_system import EyeQASystem

# 初始化系统（推荐配置）
qa = EyeQASystem(
    retrieval_mode="hybrid_rerank",  # 最优策略
    use_query_rewrite=True,          # 启用查询改写
    use_type_aware=True              # 启用Type-Aware路由
)

# 方式1：简单问答
answer = qa.answer("高度近视患者应该注意什么？")
print(answer)

# 方式2：获取详细检索信息（用于评测）
result = qa.answer_with_retrieval("高度近视患者应该注意什么？")
print(f"答案: {result['answer']}")
print(f"召回文档数: {len(result['retrieved_docs'])}")
print(f"检索模式: {result['retrieval_info']['mode']}")
print(f"问题类型: {result['retrieval_info']['question_type']}")
```

### 5. 运行评测

```bash
# 运行所有实验配置
python run_all_experiments.py

# 生成综合报告
python generate_comprehensive_summary.py

# 科研级评测
python -c "
from evaluator_scientific import ScientificEvaluator
from qa_system import EyeQASystem

evaluator = ScientificEvaluator(use_base_model_judge=True)
qa = EyeQASystem(use_type_aware=True)

# 评测单个问题
result = qa.answer_with_retrieval('问题')
eval_result = evaluator.evaluate_single(
    question='问题',
    reference='参考答案',
    answer=result['answer'],
    retrieved_docs=result['retrieved_docs'],
    doc_scores=result['doc_scores']
)
print(eval_result)
"
```

---

## 🔬 科研级评测体系

### 评测维度

| 维度 | 指标 | 说明 |
|-----|------|------|
| **排名质量** | MRR, NDCG@K, P@K, R@K, MAP | 信息检索标准指标 |
| **检索性能** | Precision, Diversity, Coverage | 召回质量评估 |
| **生成质量** | Correctness, Completeness, Safety, Helpfulness, Groundedness | LLM-as-a-Judge |
| **统计检验** | t-test, Wilcoxon, Cohen's d | 显著性检验+效应量 |

### 评测示例

```python
from evaluator_scientific import ScientificEvaluator

evaluator = ScientificEvaluator(use_base_model_judge=True)

# 批量评测
results = evaluator.evaluate_batch(predictions)

# 方法比较（统计显著性检验）
comparison = evaluator.compare_methods(
    method_a_results=results_hybrid_rerank,
    method_b_results=results_vector,
    metric_key="overall"
)

print(f"显著性: {comparison['paired_t_test']['significance']}")
print(f"效应量: {comparison['effect_size']['interpretation']}")
```

---

## 📈 实验复现

### 复现四种策略对比实验

```bash
# 1. vector策略
python evaluate.py \
    --retrieval_mode vector \
    --test_file test_set.json \
    --output eval_vector.json

# 2. hybrid策略
python evaluate.py \
    --retrieval_mode hybrid \
    --test_file test_set.json \
    --output eval_hybrid.json

# 3. hybrid_rerank策略（最优）
python evaluate.py \
    --retrieval_mode hybrid_rerank \
    --test_file test_set.json \
    --output eval_hybrid_rerank.json

# 4. 生成对比报告
python generate_comprehensive_summary.py
```

### 复现α寻优实验

```bash
# 测试不同α值
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    python evaluate.py \
        --retrieval_mode hybrid \
        --alpha $alpha \
        --test_file test_set.json \
        --output eval_hybrid_alpha_${alpha}.json
done
```

---

## 💾 显存占用参考（A4000 16GB）

| 阶段 | 显存占用 | 说明 |
|-----|---------|------|
| 模型加载（4位NF4） | ~6GB | Qwen2.5-7B-Instruct |
| RAG向量库 | ~2GB（内存） | FAISS CPU模式 |
| 推理峰值 | ~8GB | 含Cross-Encoder |
| 训练峰值（QLoRA） | ~12GB | r=8, batch=4 |

---

## ⚠️ 医疗免责声明

**重要提醒**：本系统仅作为眼底病知识参考工具，不构成医疗建议。系统生成的回答不能替代专业医生的诊断和治疗建议。对于任何健康问题，尤其是紧急情况，请立即咨询专业医疗人员。

系统在设计过程中已考虑医疗安全性，但由于AI技术的局限性，不能保证所有回答的绝对准确性。使用本系统时，请始终保持谨慎，并以专业医疗建议为准。

---

## 📚 相关文档

- [技术实现细节说明书](技术实现细节说明书.md) - 详细技术实现文档
- [论文实验数据汇总](论文实验数据汇总.md) - 完整实验数据
- [测试集说明文档](测试集说明文档.md) - 测试集详细信息
- [EXAMPLES.md](EXAMPLES.md) - 更多使用示例
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南

---

## 🤝 贡献

欢迎提交Issue和Pull Request！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 基础大语言模型
- [Hugging Face](https://huggingface.co/) - 模型和数据集平台
- [LangChain](https://python.langchain.com/) - RAG框架参考
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调
- [EYE-QA-PLUS](https://huggingface.co/datasets/QIAIUNCC/EYE-QA-PLUS) - 眼底病问答数据集

---

## 📧 联系方式

- **项目主页**: https://github.com/qqnnhhdmpc666/low-resource-fundus-qa
- **问题反馈**: [GitHub Issues](https://github.com/qqnnhhdmpc666/low-resource-fundus-qa/issues)
- **邮件联系**: [your-email@example.com]

---

**如果本项目对您有帮助，请给个 ⭐ Star 支持一下！**
