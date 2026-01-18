# qa_system.py
# ============================================================
# 眼底健康问答系统（最终稳定版）
# - 默认：非 stream（评测 / 复试 / 实验）
# - 可选：stream（本地 Demo）
# ============================================================

import torch
import re
import warnings
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from translator import LocalTranslator

warnings.filterwarnings("ignore")


class EyeQASystem:
    def __init__(
        self,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_path="./fundus_lora",
        rag_path="./fundus_faiss",
        offload_folder="./offload",
        stream_output=False,
    ):
        self.stream_output = stream_output

        # ========== 1. tokenizer ==========
        print("正在加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )

        # ========== 2. 基模型 ==========
        print("正在加载基模型（device_map=auto）...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_folder=offload_folder,
        )

        # ========== 3. LoRA ==========
        print("正在加载 LoRA 权重...")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        print("✅ LoRA 加载成功")

        # ========== 4. RAG ==========
        print("正在加载 RAG 向量库（CPU）...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.db = FAISS.load_local(
            rag_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅ RAG 向量库加载成功")

        # ========== 5. BM25 ==========
        print("正在构建 BM25 索引...")
        self.documents = [
            doc.page_content for doc in self.db.docstore._dict.values()
        ]
        tokenized_docs = [d.lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✅ BM25 索引完成：{len(self.documents)} 篇文档")

        # ========== 6. 翻译 ==========
        print("正在加载本地翻译模块...")
        self.translator = LocalTranslator(device="cpu")
        print("✅ 翻译模块加载完成\n")

    # ============================================================
    # Hybrid Retrieval（缩短版，防止 prompt 爆炸）
    # ============================================================
    def hybrid_search(self, query, top_k=5, alpha=0.7):
        vector_results = self.db.similarity_search_with_score(query, k=top_k * 2)

        vector_scores = {}
        for doc, score in vector_results:
            vector_scores[doc.page_content] = 1 / (1 + score)

        bm25_scores = self.bm25.get_scores(query.lower().split())

        combined = {}

        for rank, (doc, score) in enumerate(
            sorted(vector_scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            combined[doc] = combined.get(doc, 0) + alpha * score / rank

        for idx, score in enumerate(bm25_scores):
            if score > 0:
                combined[self.documents[idx]] = (
                    combined.get(self.documents[idx], 0)
                    + (1 - alpha) * score / 1000
                )

        return sorted(combined, key=combined.get, reverse=True)[:top_k]

    # ============================================================
    # QA 主接口
    # ============================================================
    def answer(self, question: str, max_new_tokens=200):
        # -------- 清洗 --------
        question = re.sub(
            r"[^\w\s\u4e00-\u9fa5A-Za-z.,!?，。？！]", "", question
        ).strip()

        if not question:
            return "Invalid input."

        is_zh = any("\u4e00" <= c <= "\u9fa5" for c in question)

        search_query = (
            self.translator.zh_to_en(question) if is_zh else question
        )

        # -------- RAG --------
        docs = self.hybrid_search(search_query, top_k=5)
        context = "\n".join(docs)

        # -------- Prompt（复试友好版）--------
        prompt = f"""
You are an ophthalmology health assistant, not a doctor.

Rules:
1. Provide medical knowledge and daily care advice only.
2. Do NOT diagnose or prescribe medication.
3. If symptoms are serious or unclear, advise seeing a doctor.

Medical reference:
{context}

Question:
{question}

Answer in {"Chinese" if is_zh else "English"}:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # -------- 非 stream（评测）--------
        if not self.stream_output:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            text = self.tokenizer.decode(
                output[0], skip_special_tokens=True
            ).replace(prompt, "").strip()

        # -------- stream（Demo）--------
        else:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            def _gen():
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        streamer=streamer,
                    )

            threading.Thread(target=_gen).start()
            text = "".join(token for token in streamer)

        # -------- 免责声明 --------
        if is_zh:
            text += "\n\n【免责声明】本回答仅供健康科普，不能替代医生诊断。"
        else:
            text += (
                "\n\nDisclaimer: This answer is for informational purposes only "
                "and does not replace professional medical advice."
            )

        return text.strip()


# ============================================================
# evaluate.py 使用接口（阻塞式）
# ============================================================
eye_qa = EyeQASystem(stream_output=False).answer
