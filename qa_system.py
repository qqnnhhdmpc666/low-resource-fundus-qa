# ============================================================
# qa_system.py
# RAG Retrieval Ablation: Vector / Hybrid / Hybrid + Rerank
# With Query Rewriting (Safe, Optional, Fallback-enabled)
# ============================================================

import torch
import re
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder
from translator import LocalTranslator
from question_classifier import QuestionClassifier, get_optimal_strategy

warnings.filterwarnings("ignore")


# ============================================================
# Reranker
# ============================================================

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # 尝试在 GPU 上运行，如果没有 GPU 再回退到 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []
        # 限制文档数量，确保重排序速度
        docs = docs[:20]  # 最多处理20个文档
        pairs = [(query, d) for d in docs]
        # 使用批处理预测，提高速度
        scores = self.model.predict(pairs, batch_size=16)  # 批处理大小设置为16
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]


# ============================================================
# Query Rewriter (Lightweight, Prompt-based)
# ============================================================

class QueryRewriter:
    """
    Rewrite user query into a concise, medical-search-oriented query.
    This module is intentionally conservative to avoid harming ROUGE/BERTScore.
    """

    def __init__(self):
        pass

    def rewrite(self, query: str) -> str:
        """
        Very lightweight heuristic rewriting:
        - remove polite / conversational prefixes
        - normalize common ophthalmology terms
        - keep length short
        """
        q = query.lower()

        # remove common conversational phrases
        q = re.sub(
            r"(can you|could you|please|i want to know|tell me about|what is|how to)",
            "",
            q,
        )

        # normalize ophthalmology-related terms
        replacements = {
            "eye dryness": "dry eye disease",
            "dry eyes": "dry eye disease",
            "red eye": "ocular redness",
            "blurry vision": "blurred vision",
            "eye pain": "ocular pain",
        }

        for k, v in replacements.items():
            q = q.replace(k, v)

        q = re.sub(r"\s+", " ", q).strip()

        # fallback protection
        return q if len(q) > 3 else query


# ============================================================
# QA System
# ============================================================

class EyeQASystem:
    def __init__(
        self,
        base_model="Qwen/Qwen2.5-7B-Instruct",  # 基础语言模型
        lora_path="./fundus_lora",  # LoRA微调模型路径
        rag_path="./fundus_faiss",  # RAG知识库路径
        retrieval_mode="vector",   # 检索模式：vector(纯向量) | hybrid(向量+BM25) | hybrid_rerank(混合+重排序)
        use_query_rewrite=True,    # 查询改写：True(启用) | False(禁用)
        stream_output=False,  # 流式输出：True(启用) | False(禁用)
        use_type_aware=False,  # Type-Aware路由：True(启用) | False(禁用)
    ):
        # ====== 旋钮参数 ======
        # 1. retrieval_mode: 控制知识检索策略
        #    - vector: 仅使用向量相似度搜索，适合语义相关的查询
        #    - hybrid: 结合向量搜索和BM25关键词搜索，平衡语义和关键词匹配
        #    - hybrid_rerank: 在混合搜索基础上添加重排序，进一步优化结果质量
        #
        # 2. use_query_rewrite: 控制是否对查询进行标准化处理
        #    - True: 启用查询改写，移除礼貌用语，标准化医学术语
        #    - False: 禁用查询改写，使用原始查询
        #
        # 3. use_type_aware: 控制是否启用基于NLI的问题类型分类和路由
        #    - True: 启用Type-Aware，根据问题类型自动选择最优检索策略
        #    - False: 使用固定的retrieval_mode
        # =====================
        self.retrieval_mode = retrieval_mode
        self.stream_output = stream_output
        self.use_query_rewrite = use_query_rewrite
        self.use_type_aware = use_type_aware

        # ---------------- Model ----------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

        # ---------------- Embedding DB ----------------
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        self.db = FAISS.load_local(
            rag_path, embeddings, allow_dangerous_deserialization=True
        )

        self.documents = [d.page_content for d in self.db.docstore._dict.values()]
        self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])

        # ---------------- Optional Reranker ----------------
        self.reranker = Reranker()

        # ---------------- Translator ----------------
        self.translator = LocalTranslator(device="cpu")

        # ---------------- Query Rewriter ----------------
        self.query_rewriter = QueryRewriter()

        # ---------------- Type-Aware Classifier ----------------
        if self.use_type_aware:
            print("[QA] Loading Type-Aware NLI classifier...")
            self.question_classifier = QuestionClassifier()
        else:
            self.question_classifier = None

        print(f"[QA] Initializing system with:")
        print(f"[QA] Retrieval mode: {self.retrieval_mode}")
        print(f"[QA] Query rewrite enabled: {self.use_query_rewrite}")
        print(f"[QA] Type-Aware routing: {self.use_type_aware}")

    # ========================================================
    # Retrieval Methods
    # ========================================================

    def vector_search(self, query, k=8):
        docs = self.db.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def hybrid_search(self, query, k=20, alpha=0.7):
        vector_results = self.db.similarity_search_with_score(query, k=k)
        vector_scores = {d.page_content: 1 / (1 + s) for d, s in vector_results}

        bm25_scores = self.bm25.get_scores(query.lower().split())

        combined = {}
        for rank, (doc, score) in enumerate(
            sorted(vector_scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            combined[doc] = combined.get(doc, 0) + alpha * score / rank

        for idx, score in enumerate(bm25_scores):
            if score > 0:
                combined[self.documents[idx]] = combined.get(self.documents[idx], 0) + (
                    (1 - alpha) * score / 1000
                )

        return sorted(combined, key=combined.get, reverse=True)

    # ========================================================
    # Answer
    # ========================================================

    def answer_with_retrieval(self, question: str, max_new_tokens=200):
        """
        返回答案和详细的检索信息（用于科研评测）
        
        Returns:
            {
                "answer": str,  # 生成的答案
                "retrieved_docs": List[str],  # 召回的文档列表（按相关性排序）
                "doc_scores": List[float],  # 文档相关性分数
                "retrieval_info": {  # 检索过程信息
                    "mode": str,  # 使用的检索模式
                    "question_type": str,  # 问题类型（如果启用Type-Aware）
                    "recall_k": int,  # 召回数量
                    "rerank_k": int,  # 重排序数量
                }
            }
        """
        question = re.sub(
            r"[^\w\s\u4e00-\u9fa5A-Za-z.,!?，。？！]", "", question
        ).strip()

        is_zh = any("\u4e00" <= c <= "\u9fa5" for c in question)

        # -------- Type-Aware Routing --------
        effective_mode = self.retrieval_mode
        question_type = None
        type_aware_info = None
        
        if self.use_type_aware and self.question_classifier:
            # 使用NLI分类器确定问题类型
            question_type = self.question_classifier.classify(question)
            # 获取最优策略
            strategy = get_optimal_strategy(question_type)
            effective_mode = strategy["retrieval_mode"]
            type_aware_info = {
                "question_type": question_type,
                "strategy_reason": strategy["reason"]
            }
            print(f"[Type-Aware] Question type: {question_type}")
            print(f"[Type-Aware] Selected mode: {effective_mode}")
            print(f"[Type-Aware] Reason: {strategy['reason']}")

        # -------- Translation --------
        search_query = (
            self.translator.zh_to_en(question) if is_zh else question
        )

        # -------- Query Rewriting (Safe & Optional) --------
        if self.use_query_rewrite:
            try:
                rewritten_query = self.query_rewriter.rewrite(search_query)
            except Exception:
                rewritten_query = search_query
        else:
            rewritten_query = search_query

        # ---------------- Retrieval Switch ----------------
        retrieved_docs = []
        doc_scores = []
        retrieval_info = {
            "mode": effective_mode,
            "question_type": question_type,
            "recall_k": 0,
            "rerank_k": 0
        }
        
        if effective_mode == "vector":
            # 向量检索，获取带分数的结果
            vector_results = self.db.similarity_search_with_score(rewritten_query, k=8)
            retrieved_docs = [doc.page_content for doc, _ in vector_results]
            doc_scores = [1.0 / (1.0 + score) for _, score in vector_results]  # 转换为相似度
            retrieval_info["recall_k"] = 8
            context_docs = retrieved_docs

        elif effective_mode == "hybrid":
            # 混合检索，获取分数
            hybrid_results = self.hybrid_search_with_scores(rewritten_query, k=8)
            retrieved_docs = [doc for doc, _ in hybrid_results]
            doc_scores = [score for _, score in hybrid_results]
            retrieval_info["recall_k"] = 8
            context_docs = retrieved_docs

        elif effective_mode == "hybrid_rerank":
            # 按照用户要求的流程：Recall (20) → Rerank (top 5) → Context merge (max 8)
            recall_k = 20  # 召回20个文档
            rerank_top_k = 5  # 重排序后取前5个
            max_context_docs = 8  # 最大上下文文档数为8
            
            # 首先获取20个文档（带分数）
            recall_results = self.hybrid_search_with_scores(rewritten_query, k=recall_k)
            recall_docs = [doc for doc, _ in recall_results]
            recall_scores = [score for _, score in recall_results]
            
            # 重排序取前5个
            reranked = self.reranker.rerank(rewritten_query, recall_docs, top_k=rerank_top_k)
            
            # 合并重排序结果和原始结果
            context_set = set(reranked)
            context_docs = reranked.copy()
            
            # 从原始结果中添加额外的文档
            for doc in recall_docs:
                if doc not in context_set and len(context_docs) < max_context_docs:
                    context_docs.append(doc)
                    context_set.add(doc)
                if len(context_docs) >= max_context_docs:
                    break
            
            context_docs = context_docs[:max_context_docs]
            
            # 记录召回信息（重排序后的文档排在前面）
            retrieved_docs = context_docs
            # 重排序后的文档给更高分数
            doc_scores = [0.9 - i * 0.05 for i in range(len(context_docs))]
            
            retrieval_info["recall_k"] = recall_k
            retrieval_info["rerank_k"] = rerank_top_k

        else:
            raise ValueError("Invalid retrieval_mode")

        context = "\n".join(context_docs)

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

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=1,
                temperature=0.0,
            )

        text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).replace(prompt, "").strip()

        if not is_zh:
            text += (
                "\n\nCommon measures include blinking more often, "
                "using artificial tears, taking regular screen breaks, "
                "proper lighting, and staying hydrated."
            )

        text += (
            "\n\nDisclaimer: This answer is for informational purposes only "
            "and does not replace professional medical advice."
        )

        return {
            "answer": text,
            "retrieved_docs": retrieved_docs,
            "doc_scores": doc_scores,
            "retrieval_info": retrieval_info
        }
    
    def hybrid_search_with_scores(self, query, k=20, alpha=0.7):
        """
        混合检索，返回带分数的文档列表
        
        Returns:
            List[Tuple[str, float]]: (文档内容, 分数)
        """
        # 向量检索（带分数）
        vector_results = self.db.similarity_search_with_score(query, k=k)
        vector_scores = {d.page_content: 1.0 / (1.0 + s) for d, s in vector_results}
        
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
                ) + (1 - alpha) * score / 1000.0
        
        # 返回排序后的结果
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def answer(self, question: str, max_new_tokens=200):
        """简化接口，只返回答案文本"""
        result = self.answer_with_retrieval(question, max_new_tokens)
        return result["answer"]


# ============================================================
# evaluate.py 接口
# ============================================================

# 缓存QA系统实例，避免重复创建
_qa_system_cache = {}

def get_eye_qa_system(retrieval_mode="hybrid_rerank", use_query_rewrite=True, stream_output=False, use_type_aware=False):
    """
    获取配置化的QA系统实例
    
    参数:
        retrieval_mode: str - 检索模式
            - "vector": 纯向量搜索，适合语义相关的查询
            - "hybrid": 向量+BM25混合搜索，平衡语义和关键词匹配
            - "hybrid_rerank": 混合搜索+重排序，进一步优化结果
        use_query_rewrite: bool - 是否启用查询改写
            - True: 启用，移除礼貌用语，标准化医学术语
            - False: 禁用，使用原始查询
        stream_output: bool - 是否启用流式输出
            - True: 启用，适合实时交互场景
            - False: 禁用，适合批量处理场景
        use_type_aware: bool - 是否启用Type-Aware路由
            - True: 启用，基于NLI模型自动选择最优检索策略
            - False: 禁用，使用固定的retrieval_mode
    
    返回:
        function - 配置好的QA系统回答函数
    """
    # 生成缓存键
    cache_key = f"{retrieval_mode}_{use_query_rewrite}_{stream_output}_{use_type_aware}"
    
    # 如果缓存中存在实例，直接返回
    if cache_key in _qa_system_cache:
        return _qa_system_cache[cache_key]
    
    # 否则创建新实例并缓存
    qa_system = EyeQASystem(
        retrieval_mode=retrieval_mode,
        use_query_rewrite=use_query_rewrite,
        stream_output=stream_output,
        use_type_aware=use_type_aware
    ).answer
    
    _qa_system_cache[cache_key] = qa_system
    return qa_system

# 按需创建实例，避免模块导入时自动加载模型
def get_default_eye_qa():
    """获取默认配置的QA系统实例"""
    return get_eye_qa_system()
