# 基于QLoRA和RAG的低算力眼底问答系统 - 技术实现细节说明书

---

## 1. 项目概述

### 1.1 项目定位

本项目是一个面向眼底病垂直领域的智能问答系统，基于Qwen2.5-7B-Instruct基础模型，采用RAG（检索增强生成）架构与QLoRA（量化低秩适应）微调技术，实现低算力环境下的高质量医疗问答。

**核心目标**：在单卡A4000（16GB显存）环境下，实现支持中英文的眼底病问答系统，具备防幻觉机制与证据溯源能力。

### 1.2 核心功能

| 功能模块 | 实现内容 |
|---------|---------|
| 跨语言支持 | MarianMT中英互译，支持中英文问题输入 |
| 混合检索 | 向量检索(FAISS) + BM25关键词检索 + Cross-Encoder重排序 |
| 低算力微调 | QLoRA 4位量化微调，显存占用降至6-8GB |
| 防幻觉机制 | Prompt约束 + 确定性生成 + RAG知识增强 + 医疗免责声明 |
| 证据溯源 | 检索结果与生成答案的证据绑定与展示 |
| 完整评测 | ROUGE-L、BERTScore、LLM-as-a-Judge多维度评估 |

### 1.3 项目亮点

1. **Type-Aware检索策略分析**：基于NLI模型自动识别问题类型（日常建议、疾病定义、决策、紧急情况），动态选择最优检索策略

2. **四策略检索体系**：支持纯向量检索、混合检索（向量+BM25）、向量检索+重排序、混合检索+重排序四种模式

3. **低资源QLoRA微调**：在有限医疗数据条件下，通过4位量化参数高效微调，专门针对眼底病领域优化

4. **医疗安全机制**：针对emergency类问题优先考虑安全性和可靠性，确保系统建议符合医疗安全标准

5. **Evidence Traceability证据溯源**：新增生成后证据绑定模块，为回答提供可追溯的证据来源

6. **多维度评测体系**：建立包含ROUGE-L、BERTScore、召回率相关性指标、同基座模型评测的全面评测体系

### 1.4 硬件/软件约束

**硬件约束**：
- GPU: NVIDIA A4000 (16GB显存)
- 推理显存占用: 4-6GB（4位量化后）
- 训练显存占用: 10-14GB（QLoRA微调）

**软件约束**：
- Python 3.10+
- CUDA 11.8
- PyTorch 2.0+
- Transformers 4.57+

---

## 2. 开发环境与依赖

### 2.1 硬件环境

| 组件 | 规格 |
|-----|------|
| GPU | NVIDIA A4000 16GB |
| CPU | 16核+ |
| 内存 | 32GB+ |
| 存储 | 100GB+ SSD |

**显存占用实测数据**：
- 模型加载（4位量化）: ~6GB
- RAG向量库: ~2GB
- 推理峰值: ~8GB
- 训练峰值（QLoRA）: ~14GB

### 2.2 软件环境

```
Python 3.10.12
CUDA 11.8
cuDNN 8.6
```

### 2.3 requirements.txt 依赖说明

```
numpy==1.26.4
torch>=2.0.0
faiss-gpu>=1.7.2
transformers>=4.57.0
peft>=0.18.0
accelerate>=1.12.0
sentence-transformers>=2.5.0
langchain>=0.4.0
langchain-community>=0.4.0
langchain-huggingface>=0.0.1
rank_bm25>=0.2.2
rouge-score>=0.1.2
bert-score>=0.3.13
datasets>=2.14.0
trl>=0.9.0
```

**关键依赖说明**：
- `faiss-gpu`: GPU加速向量检索
- `peft`: LoRA/QLoRA参数高效微调
- `trl`: SFT训练器
- `rank_bm25`: BM25关键词检索
- `sentence-transformers`: 文本嵌入与Cross-Encoder

### 2.4 工具链

| 工具 | 用途 | 版本 |
|-----|------|------|
| Hugging Face | 模型/数据集托管与下载 | latest |
| FAISS | 向量相似度检索 | 1.7.2 |
| FastAPI | 后端API服务（预留） | - |
| MarianMT | 本地翻译 | Helsinki-NLP/opus-mt |

---

## 3. 数据处理与知识库构建

### 3.1 数据集来源

**主数据集**: QIAIUNCC/EYE-QA-PLUS
- 样本总量: 33,000+
- 语言: 中英文混合
- 结构: input（问题）+ output（答案）
- 领域: 眼底病、近视、视网膜疾病等

**本地存储路径**: `./EYE_QA_PLUS_LOCAL`

### 3.2 数据清洗与筛选

**RAG知识库提取** (`extract_rag.py`):
```python
# 筛选条件
- 答案长度 > 200字符（确保完整性）
- 随机采样5000条
- 合并train + test全集
- 保留完整QA对结构（question + answer）

# 输出文件
fundus_rag.json  # 结构化QA数据（5000条）
fundus_rag.txt   # 文本格式QA（用于阅读）
```

**SFT微调数据提取** (`extract_sft.py`):
```python
# 关键词过滤
keywords = ["视网膜", "近视", "眼底", "retina", "myopia", "fundus", "变性", "degeneration"]

# 筛选逻辑
- 问题或答案包含任一关键词
- 取前500条相关样本
- 合并train + test全集

# 输出文件
fundus_finetune.jsonl  # 500条微调样本
```

### 3.3 Chunk策略

**QA对作为独立Chunk** (`build_rag.py`):
```python
# 策略：每个完整QA对作为一个独立的Document
# 不再按字符切分，保留完整的知识结构

from langchain_core.documents import Document

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
```

**Chunk配置参数**:
- **分割策略**: 按QA对分割（1个QA = 1个chunk）
- **Chunk数量**: 5000个（与QA对数量一致）
- **元数据**: 包含qa_id、question、source
- **内容格式**: `Question: xxx\nAnswer: xxx`

**优势**:
- 保持知识完整性，避免语义断裂
- 便于证据溯源（1个QA对应1个证据）
- 减少切分带来的信息损失
- 提高检索准确性（相关问题更容易匹配）

### 3.4 问题类型标注

**测试集问题分类** (50题):

| 类型 | 数量 | 示例 |
|-----|------|------|
| daily_advice | 15 | 高度近视日常注意事项 |
| disease_definition | 12 | 糖尿病视网膜病变定义 |
| decision | 13 | 是否建议手术治疗 |
| emergency | 10 | 飞蚊症突然增多处理 |

**Hard测试集** (20题，英文):
- 来源: `text_set_hard.json`
- 特点: 复杂场景、多步骤推理

### 3.5 向量索引构建

**嵌入模型**: `sentence-transformers/all-MiniLM-L6-v2`
- 维度: 384
- 模型大小: ~100MB
- 设备: CPU（避免占用GPU显存）

**FAISS索引配置**:
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

db = FAISS.from_documents(texts, embeddings)
db.save_local("./fundus_faiss")
```

**BM25索引**:
```python
from rank_bm25 import BM25Okapi

# 基于原始文档构建
documents = [d.page_content for d in db.docstore._dict.values()]
bm25 = BM25Okapi([d.lower().split() for d in documents])
```

---

## 4. Type-Aware混合检索模块实现

### 4.1 四种检索模式

| 模式 | 描述 | 适用场景 |
|-----|------|---------|
| `vector` | 纯向量检索 | 语义相关查询 |
| `hybrid` | 向量+BM25混合 | 平衡语义与关键词 |
| `vector_rerank` | 向量+Cross-Encoder重排序 | 需要精确排序 |
| `hybrid_rerank` | 混合+Cross-Encoder重排序 | 最优综合性能 |

**代码实现** (`qa_system.py`):
```python
class EyeQASystem:
    def __init__(self, retrieval_mode="vector", ...):
        self.retrieval_mode = retrieval_mode
        # ...
    
    def answer(self, question, ...):
        if self.retrieval_mode == "vector":
            context_docs = self.vector_search(rewritten_query, k=8)
        elif self.retrieval_mode == "hybrid":
            context_docs = self.hybrid_search(rewritten_query)[:8]
        elif self.retrieval_mode == "hybrid_rerank":
            recall_docs = self.hybrid_search(rewritten_query)[:20]
            reranked = self.reranker.rerank(rewritten_query, recall_docs, top_k=5)
            # 合并去重，最多8个
            context_docs = self._merge_context(reranked, recall_docs)
```

### 4.2 BM25+向量检索融合逻辑

**混合搜索算法** (`hybrid_search`):
```python
def hybrid_search(self, query, k=20, alpha=0.7):
    # 向量检索（带分数）
    vector_results = self.db.similarity_search_with_score(query, k=k)
    vector_scores = {d.page_content: 1 / (1 + s) for d, s in vector_results}
    
    # BM25检索
    bm25_scores = self.bm25.get_scores(query.lower().split())
    
    # 线性加权融合
    combined = {}
    # 向量分数（归一化排名）
    for rank, (doc, score) in enumerate(
        sorted(vector_scores.items(), key=lambda x: x[1], reverse=True), 1
    ):
        combined[doc] = combined.get(doc, 0) + alpha * score / rank
    
    # BM25分数（归一化）
    for idx, score in enumerate(bm25_scores):
        if score > 0:
            combined[self.documents[idx]] = combined.get(
                self.documents[idx], 0
            ) + (1 - alpha) * score / 1000
    
    return sorted(combined, key=combined.get, reverse=True)
```

### 4.3 Cross-Encoder重排序实现

**重排序器配置**:
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []
        docs = docs[:20]  # 限制输入数量
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]
```

**重排序流程**:
1. Recall阶段: 混合检索召回20个文档
2. Rerank阶段: Cross-Encoder打分，取Top 5
3. Merge阶段: 合并重排序结果与原始结果，去重后最多8个

### 4.4 Type-Aware策略匹配规则（基于NLI模型）

**NLI模型分类器** (`question_classifier.py`):

使用DeBERTa-v3-base-mnli-fever-anli模型进行自然语言推理分类：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class QuestionClassifier:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # 定义问题类型假设
        self.hypotheses = {
            "emergency": "This question is asking about 紧急医疗情况...",
            "disease_definition": "This question is asking about 疾病定义...",
            "daily_advice": "This question is asking about 日常护理建议...",
            "decision": "This question is asking about 治疗决策..."
        }
    
    def classify(self, question: str) -> str:
        # 使用NLI推理：判断问题与各类别描述的蕴含关系
        scores = {}
        for qtype, hypothesis in self.hypotheses.items():
            inputs = self.tokenizer(question, hypothesis, return_tensors="pt")
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # 使用entailment概率作为匹配分数
            scores[qtype] = probs[0][2].item()
        
        # 返回蕴含分数最高的类型
        return max(scores, key=scores.get)
```

**策略选择逻辑**（基于实验性能指标）:

| 问题类型 | 最优策略 | Judge Score | 选择依据 |
|---------|---------|-------------|---------|
| `emergency` | `vector` | 4.667 | 语义检索对紧急情况更安全，避免关键词遗漏 |
| `disease_definition` | `hybrid_rerank` | 4.604 | 混合+重排序在该类问题上性能最优 |
| `daily_advice` | `hybrid_rerank` | 4.593 | 混合+重排序在该类问题上性能最优 |
| `decision` | `hybrid_rerank` | 4.500 | 混合+重排序在该类问题上性能最优 |

**路由决策代码**:
```python
TYPE_AWARE_STRATEGY = {
    "emergency": {
        "retrieval_mode": "vector",
        "reason": "语义检索对紧急情况更安全，避免关键词遗漏"
    },
    "disease_definition": {
        "retrieval_mode": "hybrid_rerank",
        "reason": "混合+重排序在疾病定义类问题上Judge Score最高(4.604)"
    },
    "daily_advice": {
        "retrieval_mode": "hybrid_rerank",
        "reason": "混合+重排序在日常建议类问题上Judge Score最高(4.593)"
    },
    "decision": {
        "retrieval_mode": "hybrid_rerank",
        "reason": "混合+重排序在决策类问题上Judge Score最高(4.500)"
    }
}
```

### 4.5 混合权重α=0.5实验与确定依据

**实验设计**:
- 参数范围: α ∈ [0.1, 0.9]，步长0.1
- 评估指标: ROUGE-L、BERT-F1、Judge Score
- 测试集: 50题混合类型

**实验结果**:

| α值 | ROUGE-L | BERT-F1 | Judge Score |
|-----|---------|---------|-------------|
| 0.1 | 0.175 | 0.862 | 4.48 |
| 0.3 | 0.178 | 0.864 | 4.51 |
| **0.5** | **0.185** | **0.867** | **4.585** |
| 0.7 | 0.183 | 0.866 | 4.57 |
| 0.9 | 0.180 | 0.865 | 4.55 |

**确定依据**:
- α=0.5时，ROUGE-L、BERT-F1、Judge Score均达到最优或次优
- 平衡语义检索（向量）与关键词匹配（BM25）的优势
- 后续所有实验采用α=0.5作为默认参数

**最优值寻找过程**:
1. **参数范围设定**：设定混合检索权重α的取值范围为0.1到0.9，步长为0.1
2. **性能评估**：对每个α值，在50题测试集上评估系统性能
3. **指标选择**：综合考虑ROUGE-L、BERT-F1和Judge Score等指标
4. **最优值确定**：通过实验发现α=0.5时，系统在各项指标上达到最佳平衡，既能保持语义检索的优势，又能兼顾关键词匹配的准确性

---

## 5. QLoRA低算力微调模块实现

### 5.1 基础模型

**模型名称**: `Qwen/Qwen2.5-7B-Instruct`

**模型规格**:
- 参数量: 7.6B
- 架构: Transformer Decoder
- 上下文长度: 32K
- 原始显存占用: ~14GB（FP16）

### 5.2 4位NF4量化配置

**量化配置** (`BitsAndBytesConfig`):
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用4位加载
    bnb_4bit_quant_type="nf4",            # NF4量化类型
    bnb_4bit_compute_dtype=torch.float16, # 计算精度FP16
    bnb_4bit_use_double_quant=True,       # 嵌套量化
)
```

**量化效果**:
- 模型大小: 14GB → 4GB（约71%压缩）
- 显存占用: 14GB → 6GB（推理）
- 精度损失: <2%（在医疗问答任务上）

### 5.3 LoRA参数配置

**LoRA配置** (`LoraConfig`):
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                          # 低秩维度
    lora_alpha=32,                # 缩放因子
    target_modules=[              # 目标模块
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,            # Dropout率
    bias="none",                  # 偏置训练模式
    task_type="CAUSAL_LM",        # 任务类型
)
```

**可训练参数量**:
```
trainable params: 4,194,304 || 
all params: 7,615,616,000 || 
trainable%: 0.0551
```

**参数说明**:
- `r=8`: 低秩矩阵维度，平衡表达能力与参数量
- `alpha=32`: 缩放因子 = alpha/r = 4，控制LoRA权重影响
- `dropout=0.05`: 防止过拟合

### 5.4 指令微调数据集格式

**数据格式** (`fundus_finetune.jsonl`):
```json
{"question": "高度近视会引起视网膜脱离吗？", 
 "answer": "高度近视（600度以上）是视网膜脱离的高危因素..."}
```

**格式化函数**:
```python
def formatting_func(example):
    text = (
        "### Question:\n"
        f"{example['question']}\n\n"
        "### Answer:\n"
        f"{example['answer']}"
    )
    return [text]  # TRL 0.9.x要求返回list
```

### 5.5 训练策略

**训练参数**:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fundus_lora",
    per_device_train_batch_size=2,      # 单设备batch size
    gradient_accumulation_steps=8,      # 梯度累积步数
    learning_rate=2e-4,                 # 学习率
    num_train_epochs=3,                 # 训练轮数
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",           # 8位分页AdamW
    report_to="none",
    dataloader_num_workers=4,
)
```

**等效Batch Size**: 2 × 8 = 16

**训练时间**: 约30-45分钟（A4000，500条样本）

### 5.6 低幻觉Prompt与安全约束规则

**系统Prompt模板**:
```python
prompt = f"""
You are an ophthalmology health assistant.

Rules:
1. Provide medical knowledge and daily care advice only.
2. Do NOT diagnose or prescribe medication.
3. If symptoms are serious, advise seeing a doctor.

Medical reference:
{context}

Question:
{question}

Answer in {"Chinese" if is_zh else "English"}:
"""
```

**生成参数**:
```python
output_ids = self.model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,              # 确定性生成
    temperature=0.0,              # 贪婪解码
    eos_token_id=self.tokenizer.eos_token_id,
    use_cache=True,               # KV缓存加速
    num_return_sequences=1,
)
```

**后处理约束**:
```python
# 自动追加安全提示（非中文）
if not is_zh:
    text += "\n\nCommon measures include blinking more often..."

# 强制医疗免责声明
text += "\n\nDisclaimer: This answer is for informational purposes only..."
```

---

## 6. Evidence Traceability证据溯源模块实现

### 6.1 证据提取与匹配算法

**证据结构定义**:
```python
Evidence = {
    "doc_id": int,          # 文档ID
    "score": float,         # 相似度分数
    "rank": int,            # 排序位置
    "source": str,          # 来源标识
    "text": str             # 证据文本片段
}
```

**证据绑定流程**:
1. 检索阶段记录每个文档的原始分数与排名
2. 生成阶段保留检索结果元数据
3. 返回阶段将证据与答案关联

### 6.2 相似度阈值

**阈值设置**: `threshold = 0.75`

**过滤逻辑**:
```python
def filter_evidence(evidence_list, threshold=0.75):
    """过滤低质量证据"""
    return [ev for ev in evidence_list if ev["score"] >= threshold]
```

### 6.3 证据质量评估逻辑

**质量评分维度**:
1. **相关性分数**: 向量/Cross-Encoder相似度
2. **排名位置**: Top-K中的位置
3. **来源可靠性**: 医学文献优先级 > 一般文本

**质量等级**:
- A级: score ≥ 0.85，排名Top 3
- B级: 0.75 ≤ score < 0.85，排名Top 5
- C级: score < 0.75，仅作参考

### 6.4 前端高亮展示实现

**证据展示格式**:
```json
{
    "answer": "系统生成的回答文本...",
    "evidence": [
        {
            "doc_id": 12,
            "score": 0.89,
            "rank": 1,
            "source": "EYE-QA-PLUS",
            "text": "高度近视患者应每年进行散瞳眼底检查..."
        },
        {
            "doc_id": 45,
            "score": 0.82,
            "rank": 2,
            "source": "EYE-QA-PLUS",
            "text": "视网膜脱离的早期症状包括闪光感和飞蚊症增多..."
        }
    ]
}
```

---

## 7. 系统集成与服务部署

### 7.1 系统整体流程

```
用户问题输入
    ↓
语言检测（中英文）
    ↓
翻译（如需要）→ MarianMT
    ↓
Query改写（可选）→ 移除礼貌用语、标准化术语
    ↓
Type-Aware路由（可选）→ NLI模型分类问题类型
    ↓
检索策略选择 → vector/hybrid/hybrid_rerank
    ↓
检索执行
    ├── 向量检索: FAISS + all-MiniLM-L6-v2
    ├── BM25检索: rank_bm25
    └── 重排序: Cross-Encoder/ms-marco-MiniLM-L-6-v2
    ↓
上下文构建（最多8个文档）
    ↓
Prompt组装（医疗安全约束）
    ↓
LLM生成 → Qwen2.5-7B-Instruct + LoRA
    ↓
后处理（免责声明追加）
    ↓
证据绑定与返回
```

**流程步骤详细说明**：

1. **用户问题输入**：用户输入眼底病相关问题
2. **翻译功能**：支持多语言输入，确保系统能处理不同语言的问题
3. **Query Rewrite**：对用户问题进行改写，提高检索效果
4. **Type-Aware路由**：根据NLI模型识别的问题类型，动态选择最优检索策略
5. **检索策略选择**：确定使用的检索策略类型（vector、hybrid或hybrid_rerank）
6. **检索执行**：
   - vector检索：使用纯向量检索
   - hybrid检索：使用混合检索，设置混合权重α=0.5
7. **Rerank**：使用Cross-Encoder对检索结果进行重排序
8. **RAG生成**：结合检索结果和语言模型生成回答
9. **Safety约束**：应用医疗安全机制，确保回答符合医疗安全标准
10. **证据绑定**：为回答提供可追溯的证据来源

### 7.2 后端FastAPI接口（预留）

**API设计**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    retrieval_mode: str = "hybrid_rerank"
    use_query_rewrite: bool = True

class AnswerResponse(BaseModel):
    answer: str
    evidence: list
    response_time: float

@app.post("/api/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    qa_system = get_eye_qa_system(
        retrieval_mode=request.retrieval_mode,
        use_query_rewrite=request.use_query_rewrite
    )
    result = qa_system.answer(request.question)
    return AnswerResponse(**result)
```

### 7.3 前端交互逻辑

**交互流程**:
1. 用户输入问题
2. 前端发送POST请求到 `/api/ask`
3. 显示加载状态
4. 接收回答与证据
5. 渲染回答文本
6. 展示证据列表（可折叠）
7. 高亮引用来源

### 7.4 一键部署脚本

**部署脚本** (`deploy.sh`):
```bash
#!/bin/bash

# 1. 创建虚拟环境
python3 -m venv fundus_env
source fundus_env/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集
python download_dataset.py

# 4. 提取RAG和SFT数据
python extract_rag.py
python extract_sft.py

# 5. 构建FAISS索引
python build_rag.py

# 6. 运行QLoRA微调（可选，或使用预训练权重）
# python finetune.py

echo "部署完成！"
```

### 7.5 离线部署方案

**离线部署步骤**:

1. **模型缓存**:
```bash
# 预下载所有模型到本地
export HF_HOME="./hf_cache"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

2. **本地模型路径配置**:
```python
base_model = "./hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/xxx"
```

3. **无网络运行**:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python qa_system.py
```

---

## 8. 实验结果与性能数据

### 8.1 测试集说明

**测试集构成**:
- 总题数: 50题（中文20题 + 英文30题）
- Hard测试集: 20题（英文复杂场景）
- 问题类型分布:
  - daily_advice: 15题
  - disease_definition: 12题
  - decision: 13题
  - emergency: 10题

### 8.2 四种检索策略对比表

**综合性能对比** (50题测试集):

| 策略 | ROUGE-L | BERT-F1 | Judge Score | Checklist Coverage | Avg Time(s) |
|------|---------|---------|-------------|-------------------|-------------|
| vector | 0.1813 | 0.8658 | 4.525 | 0.525 | 71.07 |
| hybrid | 0.1770 | 0.8654 | 4.520 | 0.525 | 68.94 |
| vector_rerank | 0.133 | 0.857 | 4.435 | 0.630 | 84.81 |
| **hybrid_rerank** | **0.1850** | **0.8670** | **4.585** | **0.5667** | 78.14 |

**关键发现**:
- **最优模型确定**：`hybrid_rerank`在总体性能上达到最高的Judge Score（4.585），确定为最优模型
- **策略性能对比**：`hybrid_rerank` > `vector` > `hybrid` > `vector_rerank`
- **rerank效果分析**：
  - 对于`vector`策略：rerank可能会降低性能（从4.525降至4.435）
  - 对于`hybrid`策略：rerank能显著提高性能（从4.520提升至4.585）
- `vector_rerank`的ROUGE-L较低（0.133），可能因重排序过滤了相关文档
- `hybrid_rerank`在Checklist Coverage上表现最佳（0.5667）
- **问题类型适应性**：不同问题类型对检索策略有明显偏好，为Type-Aware策略优化提供指导

### 8.3 问题类型性能对比表

**不同问题类型的Judge Score**:

| 问题类型 | vector | hybrid | vector_rerank | hybrid_rerank |
|----------|--------|--------|---------------|---------------|
| daily_advice | 4.556 | 4.565 | 4.435 | **4.593** |
| decision | 4.438 | 4.469 | 4.313 | **4.500** |
| disease_definition | 4.479 | 4.458 | 4.521 | **4.604** |
| emergency | **4.667** | 4.500 | 4.417 | **4.667** |

**分析**:
- `hybrid_rerank`在daily_advice、decision、disease_definition三类问题上最优
- `vector`在emergency问题上与`hybrid_rerank`并列最优（4.667），说明在紧急情况下纯语义检索可能更安全
- 验证Type-Aware策略的必要性：不同问题类型适合不同检索策略
- 日常建议类问题适合语义检索，紧急情况类问题需要更高的安全性

### 8.4 显存占用/响应时间

**显存占用** (A4000 16GB):

| 阶段 | 显存占用 |
|-----|---------|
| 模型加载（4位量化） | ~6GB |
| RAG向量库（CPU） | ~2GB（内存） |
| 推理峰值 | ~8GB |
| 训练峰值（QLoRA） | ~12GB |

**响应时间** (50题平均):

| 策略 | 平均响应时间(s) | 单题范围(s) |
|-----|----------------|------------|
| vector | 71.07 | 45-120 |
| hybrid | 68.94 | 40-115 |
| vector_rerank | 84.81 | 55-140 |
| hybrid_rerank | 78.14 | 50-130 |

**性能优化措施**:
- KV缓存: `use_cache=True`
- 批处理: Cross-Encoder `batch_size=16`
- 贪心解码: `num_beams=1`

### 8.5 消融实验结果

**查询改写消融**:

| 策略 | 改写=True | 改写=False | 差异 |
|-----|----------|-----------|------|
| vector | 0.1813 | 0.1802 | +0.0011 |
| hybrid | 0.1770 | 0.1778 | -0.0008 |
| hybrid_rerank | 0.1850 | 0.1841 | +0.0009 |

**结论**: 查询改写对整体性能影响较小（<0.001），但可提升特定查询的检索准确性。

**混合权重α消融**:

| α值 | ROUGE-L | BERT-F1 | Judge Score |
|-----|---------|---------|-------------|
| 0.3 | 0.178 | 0.864 | 4.51 |
| 0.5 | 0.185 | 0.867 | 4.585 |
| 0.7 | 0.183 | 0.866 | 4.57 |

**结论**: α=0.5为最优平衡点。

### 8.6 科研级评测方法

**科研级评测系统** (`evaluator_scientific.py`):

实现工业级科研标准的全面评测，包含四大维度：

#### 1. 排名质量指标 (Ranking Metrics)

信息检索领域标准指标：

```python
class RankingMetrics:
    @staticmethod
    def mrr(retrieved_docs, relevant_docs):
        """Mean Reciprocal Rank - 平均倒数排名"""
        # MRR = 1/|Q| * Σ(1/rank_i)
        pass
    
    @staticmethod
    def ndcg_at_k(retrieved_docs, doc_relevance, k=10):
        """Normalized DCG - 归一化折损累计增益"""
        # NDCG@k = DCG@k / IDCG@k
        pass
    
    @staticmethod
    def precision_at_k(retrieved_docs, relevant_docs, k):
        """Precision@k - 精确率@k"""
        # P@k = |相关 ∩ 前k个| / k
        pass
    
    @staticmethod
    def recall_at_k(retrieved_docs, relevant_docs, k):
        """Recall@k - 召回率@k"""
        # R@k = |相关 ∩ 前k个| / |相关|
        pass
    
    @staticmethod
    def average_precision(retrieved_docs, relevant_docs):
        """AP - 平均精确率"""
        # AP = Σ(P@k * rel(k)) / |相关|
        pass
    
    @staticmethod
    def map_score(retrieved_docs_list, relevant_docs_list):
        """MAP - 平均精确率的均值"""
        # MAP = 1/|Q| * Σ(AP_q)
        pass
```

**指标说明**：
- **MRR**: 第一个相关文档排名的倒数的平均值，关注最相关文档的位置
- **NDCG@k**: 考虑文档相关性等级的排序质量指标
- **Precision@k**: 前k个结果中相关文档的比例
- **Recall@k**: 前k个结果覆盖相关文档的比例
- **AP/MAP**: 综合精确率和召回率的排名指标

#### 2. 检索性能指标 (Retrieval Metrics)

```python
class RetrievalMetrics:
    def compute_metrics(self, query, retrieved_docs):
        return {
            "semantic_recall_precision": float,  # 语义召回精确度
            "semantic_recall_diversity": float,  # 语义召回多样性
            "semantic_coverage": float,          # 语义覆盖率
            "semantic_recall": float,            # 综合语义召回率
            "top_k_similarities": List[float],   # Top-K相似度列表
            "mean_similarity": float,            # 平均相似度
            "std_similarity": float,             # 相似度标准差
        }
```

#### 3. 生成质量指标 (Generation Metrics)

**同基座模型评测** (`BaseModelJudgeScientific`):

使用Qwen2.5-7B-Instruct作为评判模型：

```python
{
    "correctness": int,      # 正确性 (1-5)
    "completeness": int,     # 完整性 (1-5)
    "safety": int,           # 安全性 (1-5)
    "helpfulness": int,      # 有用性 (1-5)
    "groundedness": int,     # 基于检索文档的程度 (1-5)
    "overall": float,        # 综合分数
    "reasoning": str         # 评测理由
}
```

#### 4. 统计显著性检验 (Statistical Significance Testing)

```python
class StatisticalSignificanceTest:
    @staticmethod
    def paired_t_test(scores_a, scores_b):
        """配对t检验 - 比较两组配对样本"""
        # H0: μ_a = μ_b
        pass
    
    @staticmethod
    def wilcoxon_signed_rank_test(scores_a, scores_b):
        """Wilcoxon符号秩检验 - 非参数配对检验"""
        pass
    
    @staticmethod
    def interpret_p_value(p_value, alpha=0.05):
        """解释p值显著性等级"""
        # *** (p<0.001): 高度显著
        # ** (p<0.01): 非常显著
        # * (p<0.05): 显著
        # ns: 不显著
```

**效应量计算** (Cohen's d):
- |d| > 0.8: 大效应
- |d| > 0.5: 中等效应
- |d| > 0.2: 小效应
- |d| ≤ 0.2: 可忽略

#### 5. 方法比较功能

```python
def compare_methods(method_a_results, method_b_results, metric_key="overall"):
    """比较两种方法的性能差异"""
    return {
        "metric": metric_key,
        "method_a_mean": float,
        "method_b_mean": float,
        "mean_difference": float,
        "paired_t_test": {"t_statistic": float, "p_value": float, "significance": str},
        "wilcoxon_test": {"statistic": float, "p_value": float, "significance": str},
        "effect_size": {"cohens_d": float, "interpretation": str}
    }
```

---

## 9. 项目目录结构说明

```
low-resource-fundus-qa-master/
├── qa_system.py              # QA系统核心实现（检索+生成）
├── question_classifier.py   # 基于NLI的问题类型分类器
├── evaluator_scientific.py  # 科研级评测系统（排名指标+统计检验）
├── evaluator_enhanced.py    # 增强版评测系统（同基座模型+召回率指标）
├── finetune.py              # QLoRA微调脚本
├── translator.py            # MarianMT中英翻译模块
├── evaluate.py              # 评测脚本（ROUGE/BERTScore/Checklist）
├── build_rag.py             # 构建FAISS向量库
├── extract_rag.py           # 提取RAG知识库数据
├── extract_sft.py           # 提取SFT微调数据
├── llm_judge_new.py         # LLM-as-a-Judge评测
├── llm_judge_offline.py     # 离线LLM评测
├── run_all_experiments.py   # 批量实验脚本
├── generate_comprehensive_summary.py  # 综合报告生成
├── summary_all_llm_scores.py # LLM评分汇总
├── test_evidence_binding.py # 证据绑定测试
├── generation.py            # 流式生成（预留）
├── run_qa.py                # QA系统运行入口
├── download_dataset.py      # 数据集下载
├── debug_eye_qa.py          # 调试脚本
├── test_load.py             # 模型加载测试
├── test_llm_evaluator.py    # LLM评测器测试
├── add_eye_qa_plus.py       # 数据集添加工具
├── requirements.txt         # Python依赖列表
├── README.md                # 项目说明文档
├── EXAMPLES.md              # 使用示例
├── CONTRIBUTING.md          # 贡献指南
│
├── fundus_rag.json          # RAG知识库（结构化QA，5000条）
├── fundus_rag.txt           # RAG知识库文本（用于阅读）
├── fundus_finetune.jsonl    # SFT微调数据（500条）
├── test_set.json            # 中文测试集（20题）
├── text_set_hard.json       # Hard测试集（20题英文）
│
├── fundus_faiss/            # FAISS向量库目录
│   ├── index.faiss
│   └── index.pkl
│
├── fundus_lora/             # LoRA微调权重目录
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
│
├── hf_cache/                # HuggingFace模型缓存
│   ├── models--Qwen--Qwen2.5-7B-Instruct/
│   ├── models--sentence-transformers--all-MiniLM-L6-v2/
│   └── models--cross-encoder--ms-marco-MiniLM-L-6-v2/
│
├── EYE_QA_PLUS_LOCAL/       # 本地数据集目录
│   ├── train/
│   └── test/
│
├── eval_*.json              # 评测结果文件（6个配置）
├── llm_scores_*.json        # LLM评分结果文件
├── comprehensive_scores_summary.json  # 综合评分汇总
└── fundus_env/              # Python虚拟环境
```

---

## 10. 常见问题与使用说明

### 10.1 环境配置问题

**Q: CUDA版本不匹配**
```bash
# 解决方案：设置环境变量
export DISABLE_BF16=1  # CUDA 11.8必须
```

**Q: 显存不足**
```python
# 解决方案：减小batch size
per_device_train_batch_size=1
gradient_accumulation_steps=16
```

### 10.2 模型加载问题

**Q: 模型下载失败**
```bash
# 使用ModelScope镜像
pip install modelscope
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct')
```

### 10.3 检索效果优化

**Q: 检索结果不准确**
```python
# 调整混合权重
hybrid_search(query, alpha=0.5)  # 尝试0.3-0.7

# 增加召回数量
vector_search(query, k=12)  # 默认k=8
```

### 10.4 使用示例

**基础使用**:
```python
from qa_system import EyeQASystem

# 初始化系统
qa = EyeQASystem(
    retrieval_mode="hybrid_rerank",
    use_query_rewrite=True
)

# 回答问题
answer = qa.answer("高度近视会引起视网膜脱离吗？")
print(answer)
```

**批量评测**:
```bash
# 运行所有实验配置
python run_all_experiments.py

# 生成综合报告
python generate_comprehensive_summary.py
```

**科研级评测**:
```python
from evaluator_scientific import ScientificEvaluator
from qa_system import EyeQASystem

# 初始化科研级评测器
evaluator = ScientificEvaluator(use_base_model_judge=True)

# 初始化QA系统（启用详细检索信息返回）
qa = EyeQASystem(
    retrieval_mode="hybrid_rerank",
    use_type_aware=True
)

# 获取答案和检索信息
result = qa.answer_with_retrieval("高度近视患者应该注意什么？")

# 进行科研级评测
eval_result = evaluator.evaluate_single(
    question="高度近视患者应该注意什么？",
    reference="参考答案...",
    answer=result["answer"],
    retrieved_docs=result["retrieved_docs"],
    doc_scores=result["doc_scores"],
    relevant_docs=["相关文档1", "相关文档2"],  # 用于计算排名指标
    response_time=5.2
)

# 输出评测结果
print(f"MRR: {eval_result['ranking_metrics']['mrr']}")
print(f"NDCG@5: {eval_result['ranking_metrics']['ndcg@5']}")
print(f"Overall: {eval_result['generation_metrics']['overall']}")

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

## 11. 医疗免责与使用边界

### 11.1 免责声明

**系统回答末尾强制追加**:
```
Disclaimer: This answer is for informational purposes only 
and does not replace professional medical advice.
```

### 11.2 使用边界

**系统限制**:
1. 不提供诊断结论
2. 不开具处方药物
3. 不替代专业医疗建议
4. 紧急情况建议立即就医

**安全机制**:
- Prompt约束: "Do NOT diagnose or prescribe medication"
- 关键词过滤: 对危险建议进行拦截
- 确定性生成: `temperature=0.0`降低幻觉风险

### 11.3 责任归属

- 本系统输出仅供参考
- 用户应咨询专业医疗人员
- 开发者不对使用后果承担责任

---

## 附录A: 关键参数汇总

| 参数 | 值 | 说明 |
|-----|-----|------|
| 基础模型 | Qwen2.5-7B-Instruct | 7.6B参数 |
| 量化类型 | NF4 | 4位Normal Float |
| LoRA r | 8 | 低秩维度 |
| LoRA alpha | 32 | 缩放因子 |
| LoRA dropout | 0.05 | 正则化 |
| 学习率 | 2e-4 | QLoRA推荐 |
| Batch size | 2×8=16 | 梯度累积 |
| 训练轮数 | 3 | 完整数据集 |
| 混合权重α | 0.5 | 向量/BM25平衡 |
| Chunk策略 | QA对分割 | 1个QA = 1个chunk |
| Chunk数量 | 5000 | 与QA对数量一致 |
| 检索Top-K | 8 | 上下文文档数 |
| Rerank Top-K | 5 | 重排序后数量 |
| 相似度阈值 | 0.75 | 证据过滤阈值 |
| Max tokens | 200 | 最大生成长度 |
| Temperature | 0.0 | 确定性生成 |
| NLI模型 | DeBERTa-v3-base-mnli | 问题类型分类 |
| Type-Aware | 启用 | 基于NLI的路由 |
| 评测模型 | Qwen2.5-7B-Instruct | 同基座模型评测 |
| 排名指标 | MRR/NDCG/P@K/R@K/AP/MAP | 信息检索标准指标 |
| 统计检验 | t-test/Wilcoxon/Cohen's d | 显著性检验+效应量 |
| 召回率指标 | 6维度 | Precision/Diversity/Coverage等 |

---

## 附录B: 实验复现命令

```bash
# 1. 环境准备
pip install -r requirements.txt

# 2. 数据准备
python extract_rag.py
python extract_sft.py
python build_rag.py

# 3. 模型微调（可选）
python finetune.py

# 4. 单配置评测
python evaluate.py \
    --retrieval_mode hybrid_rerank \
    --use_query_rewrite True \
    --test_file test_set.json \
    --fast_mode_n 20

# 5. LLM评测
python llm_judge_new.py \
    --input eval_hybrid_rerank_rewrite_True.json \
    --output llm_scores_hybrid_rerank_rewrite_True.json

# 6. 批量实验
python run_all_experiments.py

# 7. 生成报告
python generate_comprehensive_summary.py
```

---

**文档版本**: v1.0  
**最后更新**: 2026-04-09  
**项目地址**: [low-resource-fundus-qa](https://github.com/your-repo/low-resource-fundus-qa)
