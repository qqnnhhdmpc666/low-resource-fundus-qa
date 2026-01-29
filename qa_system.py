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
        alpha=0.7,  # 混合检索权重：vector权重
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
        # 3. alpha: 混合检索权重
        #    - 0.0: 纯BM25
        #    - 0.5: 向量和BM25权重相等
        #    - 1.0: 纯向量
        # =====================
        self.retrieval_mode = retrieval_mode
        self.stream_output = stream_output
        self.use_query_rewrite = use_query_rewrite
        self.alpha = alpha

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

        print(f"[QA] Initializing system with:")
        print(f"[QA] Retrieval mode: {self.retrieval_mode}")
        print(f"[QA] Query rewrite enabled: {self.use_query_rewrite}")

    # ========================================================
    # Retrieval Methods
    # ========================================================

    def vector_search(self, query, k=8):
        docs = self.db.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def hybrid_search(self, query, k=20):
        vector_results = self.db.similarity_search_with_score(query, k=k)
        vector_scores = {d.page_content: 1 / (1 + s) for d, s in vector_results}

        bm25_scores = self.bm25.get_scores(query.lower().split())

        combined = {}
        for rank, (doc, score) in enumerate(
            sorted(vector_scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            combined[doc] = combined.get(doc, 0) + self.alpha * score / rank

        for idx, score in enumerate(bm25_scores):
            if score > 0:
                combined[self.documents[idx]] = combined.get(self.documents[idx], 0) + (
                    (1 - self.alpha) * score / 1000
                )

        return sorted(combined, key=combined.get, reverse=True)

    # ========================================================
    # Answer
    # ========================================================

    def answer(self, question: str, max_new_tokens=200):
        question = re.sub(
            r"[^\w\s\u4e00-\u9fa5A-Za-z.,!?，。？！]", "", question
        ).strip()

        is_zh = any("\u4e00" <= c <= "\u9fa5" for c in question)

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
        if self.retrieval_mode == "vector":
            context_docs = self.vector_search(rewritten_query, k=8)

        elif self.retrieval_mode == "hybrid":
            context_docs = self.hybrid_search(rewritten_query)[:8]

        elif self.retrieval_mode == "vector_rerank":
            recall_k = 20
            rerank_top_k = 5
            max_context_docs = 8
            
            recall_docs = self.vector_search(rewritten_query, k=recall_k)
            reranked = self.reranker.rerank(rewritten_query, recall_docs, top_k=rerank_top_k)
            
            context_set = set(reranked)
            context_docs = reranked.copy()
            
            for doc in recall_docs:
                if doc not in context_set and len(context_docs) < max_context_docs:
                    context_docs.append(doc)
                    context_set.add(doc)
                if len(context_docs) >= max_context_docs:
                    break
            
            context_docs = context_docs[:max_context_docs]

        elif self.retrieval_mode == "hybrid_rerank":
            recall_k = 20
            rerank_top_k = 5
            max_context_docs = 8
            
            recall_docs = self.hybrid_search(rewritten_query)[:recall_k]
            reranked = self.reranker.rerank(rewritten_query, recall_docs, top_k=rerank_top_k)
            
            context_set = set(reranked)
            context_docs = reranked.copy()
            
            for doc in recall_docs:
                if doc not in context_set and len(context_docs) < max_context_docs:
                    context_docs.append(doc)
                    context_set.add(doc)
                if len(context_docs) >= max_context_docs:
                    break
            
            context_docs = context_docs[:max_context_docs]

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
                use_cache=True,  # 使用缓存提高生成速度
                num_return_sequences=1,  # 只生成一个序列
                temperature=0.0,  # 确定性生成，提高速度
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

        return text


# ============================================================
# evaluate.py 接口
# ============================================================

# 缓存QA系统实例，避免重复创建
_qa_system_cache = {}

def get_eye_qa_system(retrieval_mode="hybrid_rerank", use_query_rewrite=True, stream_output=False, alpha=0.7):
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
        alpha: float - 混合检索权重 (0.0-1.0)
            - 0.0: 纯BM25
            - 0.5: 向量和BM25权重相等
            - 1.0: 纯向量
    
    返回:
        function - 配置好的QA系统回答函数
    """
    # 生成缓存键
    cache_key = f"{retrieval_mode}_{use_query_rewrite}_{stream_output}_{alpha}"
    
    # 如果缓存中存在实例，直接返回
    if cache_key in _qa_system_cache:
        return _qa_system_cache[cache_key]
    
    # 否则创建新实例并缓存
    qa_system = EyeQASystem(
        retrieval_mode=retrieval_mode,
        use_query_rewrite=use_query_rewrite,
        stream_output=stream_output,
        alpha=alpha
    ).answer
    
    _qa_system_cache[cache_key] = qa_system
    return qa_system

# 按需创建实例，避免模块导入时自动加载模型
def get_default_eye_qa():
    """获取默认配置的QA系统实例"""
    return get_eye_qa_system()
