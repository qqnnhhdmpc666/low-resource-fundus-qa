import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re
from translator import LocalTranslator

class EyeQASystem:
    def __init__(
        self,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_path="./fundus_lora",
        rag_path="./fundus_faiss",
        device="cuda",
        offload_folder=None,
        stream_output=True
    ):
        self.device = device
        self.stream_output = stream_output

        print("正在加载基模型和 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, cache_dir=offload_folder
        )

        print("正在加载基模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_folder=offload_folder
        )

        print("正在加载 LoRA 权重（非合并模式）...")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        print("✅ LoRA 加载成功！")

        print("正在加载 RAG 向量库...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = FAISS.load_local(rag_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ RAG 向量库加载成功！\n")

        print("正在加载本地翻译模块...")
        self.translator = LocalTranslator(device="cpu", cache_dir=offload_folder)
        print("✅ 翻译模块加载完成！\n")

    def answer(self, question: str, top_k=12, max_new_tokens=200):
        question_clean = re.sub(r'[^\w\s\u4e00-\u9fa5A-Za-z.,!?，。？！]', '', question).strip()
        if not question_clean:
            return "输入无效，请重新输入问题。"

        # 判断中英文
        is_chinese = any('\u4e00' <= c <= '\u9fa5' for c in question_clean)
        search_query = self.translator.zh_to_en(question_clean) if is_chinese else question_clean

        print(f"\n--- 检索中：{question_clean} ---")
        if is_chinese:
            print(f"翻译为英文检索：{search_query}")

        # RAG 检索
        docs = self.db.similarity_search(search_query, k=top_k)
        context = "\n".join([doc.page_content for doc in docs])

        # 构建 prompt
        prompt = f"""You are a professional ophthalmologist. Answer the patient's question strictly based on the following medical information. Do not hallucinate, do not add unrelated content.

Medical information:
{context}

Patient question: {question_clean}

Answer in {"Chinese" if is_chinese else "English"}:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if self.stream_output:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer = None

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=streamer
            )

        if streamer is None:
            answer_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            answer_text = ""  # 已经流式输出

        # 翻译回中文（英文问中文不翻译）
        if not is_chinese and self.stream_output is False:
            answer_text = self.translator.en_to_zh(answer_text)

        return answer_text

# 对外函数
eye_qa = EyeQASystem().answer
