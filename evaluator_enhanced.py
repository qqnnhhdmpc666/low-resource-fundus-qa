# ============================================================
# evaluator_enhanced.py
# 增强版评测系统
# 1. 同基座模型评测（使用Qwen2.5-7B-Instruct作为评判模型）
# 2. 召回率相关性指标（Recall Correlation Metrics）
# 3. 多维度综合评估
# ============================================================

import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class RecallCorrelationMetrics:
    """
    召回率相关性指标
    评估检索结果与问题的相关性
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        
    def compute_metrics(self, question: str, retrieved_docs: List[str]) -> Dict:
        """
        计算召回相关性指标
        
        Returns:
            {
                "recall_precision": float,  # 召回精确度（Top-K平均相似度）
                "recall_diversity": float,  # 召回多样性（文档间差异度）
                "recall_coverage": float,   # 召回覆盖率（相关文档比例）
                "semantic_recall": float    # 语义召回率（综合指标）
            }
        """
        if not retrieved_docs:
            return {
                "recall_precision": 0.0,
                "recall_diversity": 0.0,
                "recall_coverage": 0.0,
                "semantic_recall": 0.0
            }
        
        # 编码
        question_emb = self.model.encode(question, convert_to_tensor=True)
        doc_embs = self.model.encode(retrieved_docs, convert_to_tensor=True)
        
        # 1. 召回精确度：问题与每个召回文档的相似度
        similarities = util.cos_sim(question_emb, doc_embs)[0]
        recall_precision = similarities.mean().item()
        
        # 2. 召回多样性：文档间的平均差异（避免重复召回相似内容）
        if len(doc_embs) > 1:
            doc_similarities = util.cos_sim(doc_embs, doc_embs)
            # 计算非对角线的平均相似度，然后用1减去得到多样性
            mask = ~torch.eye(len(doc_embs), dtype=torch.bool, device=doc_embs.device)
            avg_doc_sim = doc_similarities[mask].mean().item()
            recall_diversity = 1 - avg_doc_sim
        else:
            recall_diversity = 0.0
        
        # 3. 召回覆盖率：相似度超过阈值的文档比例
        threshold = 0.5
        recall_coverage = (similarities > threshold).float().mean().item()
        
        # 4. 语义召回率：综合指标（精确度 × 覆盖率 × 多样性）
        semantic_recall = recall_precision * recall_coverage * (1 + recall_diversity) / 2
        
        return {
            "recall_precision": round(recall_precision, 4),
            "recall_diversity": round(recall_diversity, 4),
            "recall_coverage": round(recall_coverage, 4),
            "semantic_recall": round(semantic_recall, 4),
            "top1_similarity": round(similarities[0].item(), 4),
            "top3_avg_similarity": round(similarities[:3].mean().item(), 4) if len(similarities) >= 3 else round(similarities.mean().item(), 4)
        }


class BaseModelJudge:
    """
    同基座模型评测器
    使用Qwen2.5-7B-Instruct作为评判模型
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Judge] Loading base model: {model_name}")
        print(f"[Judge] Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        
        print("[Judge] Base model loaded successfully")
    
    def evaluate(
        self, 
        question: str, 
        reference: str, 
        answer: str,
        retrieved_docs: List[str] = None
    ) -> Dict:
        """
        使用同基座模型进行评测
        
        Returns:
            {
                "correctness": int (1-5),
                "completeness": int (1-5),
                "safety": int (1-5),
                "helpfulness": int (1-5),
                "groundedness": int (1-5),  # 基于检索文档的程度
                "overall": float,
                "reasoning": str
            }
        """
        # 构建评测Prompt
        prompt = self._build_judge_prompt(question, reference, answer, retrieved_docs)
        
        # 生成评测结果
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,  # 低温度确保确定性
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取生成的评测部分
        judgment_text = response[len(prompt):].strip()
        
        # 解析评测结果
        return self._parse_judgment(judgment_text)
    
    def _build_judge_prompt(
        self, 
        question: str, 
        reference: str, 
        answer: str,
        retrieved_docs: List[str] = None
    ) -> str:
        """构建评测Prompt"""
        
        doc_context = ""
        if retrieved_docs:
            doc_context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs[:3])])
        
        prompt = f"""You are an expert medical AI evaluator. Please evaluate the following AI-generated answer based on the question and reference answer.

## Evaluation Criteria (Score 1-5 for each):

1. **Correctness**: Is the answer medically accurate and consistent with the reference?
   - 5: Completely correct, no errors
   - 3: Mostly correct with minor inaccuracies
   - 1: Significantly incorrect or misleading

2. **Completeness**: Does the answer cover all key points from the reference?
   - 5: Comprehensive coverage of all important aspects
   - 3: Covers main points but misses some details
   - 1: Incomplete, missing critical information

3. **Safety**: Is the answer safe and responsible from a medical perspective?
   - 5: Very safe, includes appropriate warnings/disclaimers
   - 3: Generally safe but could be more cautious
   - 1: Potentially unsafe advice

4. **Helpfulness**: How helpful is the answer for the user's question?
   - 5: Directly addresses the question with actionable advice
   - 3: Addresses the question but could be more specific
   - 1: Not helpful or irrelevant

5. **Groundedness**: How well is the answer grounded in the retrieved documents?
   - 5: Fully supported by retrieved documents
   - 3: Partially supported
   - 1: Not supported or contradicts documents

## Input:

**Question**: {question}

**Reference Answer**: {reference}

**AI Generated Answer**: {answer}

**Retrieved Documents**:
{doc_context if doc_context else "No documents retrieved"}

## Output Format:
Please provide your evaluation in the following JSON format:
```json
{{
    "correctness": <int>,
    "completeness": <int>,
    "safety": <int>,
    "helpfulness": <int>,
    "groundedness": <int>,
    "overall": <float>,
    "reasoning": "<brief explanation>"
}}
```

## Your Evaluation:"""
        
        return prompt
    
    def _parse_judgment(self, judgment_text: str) -> Dict:
        """解析评测结果"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^}]+\}', judgment_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 确保所有字段存在
                result.setdefault("correctness", 3)
                result.setdefault("completeness", 3)
                result.setdefault("safety", 3)
                result.setdefault("helpfulness", 3)
                result.setdefault("groundedness", 3)
                result.setdefault("overall", 3.0)
                result.setdefault("reasoning", "Parsed from model output")
                return result
        except:
            pass
        
        # 如果解析失败，返回默认值
        return {
            "correctness": 3,
            "completeness": 3,
            "safety": 3,
            "helpfulness": 3,
            "groundedness": 3,
            "overall": 3.0,
            "reasoning": "Failed to parse judgment, using default values"
        }


class EnhancedEvaluator:
    """
    增强版综合评测器
    整合召回率指标和同基座模型评测
    """
    
    def __init__(self, use_base_model_judge=True):
        self.recall_metrics = RecallCorrelationMetrics()
        
        if use_base_model_judge:
            self.base_judge = BaseModelJudge()
        else:
            self.base_judge = None
    
    def evaluate_single(
        self,
        question: str,
        reference: str,
        answer: str,
        retrieved_docs: List[str],
        response_time: float = None
    ) -> Dict:
        """
        对单个样本进行综合评测
        """
        results = {}
        
        # 1. 召回率相关性指标
        recall_metrics = self.recall_metrics.compute_metrics(question, retrieved_docs)
        results["recall_metrics"] = recall_metrics
        
        # 2. 同基座模型评测
        if self.base_judge:
            judge_scores = self.base_judge.evaluate(
                question, reference, answer, retrieved_docs
            )
            results["judge_scores"] = judge_scores
        
        # 3. 工程指标
        if response_time:
            results["response_time"] = response_time
        
        return results
    
    def evaluate_batch(
        self,
        predictions: List[Dict],
        use_base_model: bool = True
    ) -> Dict:
        """
        批量评测
        
        Args:
            predictions: 包含question, reference, answer, retrieved_docs的列表
            
        Returns:
            综合评测报告
        """
        all_results = []
        
        for pred in predictions:
            result = self.evaluate_single(
                question=pred["question"],
                reference=pred["reference"],
                answer=pred["answer"],
                retrieved_docs=pred.get("retrieved_docs", []),
                response_time=pred.get("response_time")
            )
            all_results.append(result)
        
        # 汇总统计
        summary = self._summarize_results(all_results)
        
        return {
            "summary": summary,
            "details": all_results
        }
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """汇总统计结果"""
        summary = {
            "recall_metrics": {},
            "judge_scores": {}
        }
        
        # 召回率指标汇总
        recall_keys = ["recall_precision", "recall_diversity", "recall_coverage", 
                      "semantic_recall", "top1_similarity", "top3_avg_similarity"]
        for key in recall_keys:
            values = [r["recall_metrics"][key] for r in results if "recall_metrics" in r]
            if values:
                summary["recall_metrics"][key] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "min": round(np.min(values), 4),
                    "max": round(np.max(values), 4)
                }
        
        # 评测分数汇总
        if self.base_judge:
            judge_keys = ["correctness", "completeness", "safety", "helpfulness", 
                         "groundedness", "overall"]
            for key in judge_keys:
                values = [r["judge_scores"][key] for r in results 
                         if "judge_scores" in r and key in r["judge_scores"]]
                if values:
                    summary["judge_scores"][key] = {
                        "mean": round(np.mean(values), 4),
                        "std": round(np.std(values), 4)
                    }
        
        return summary


if __name__ == "__main__":
    # 测试
    print("=== 测试召回率相关性指标 ===")
    recall_metrics = RecallCorrelationMetrics()
    
    question = "高度近视患者应该注意什么？"
    docs = [
        "高度近视患者应每年进行散瞳眼底检查，监测视网膜健康状况。",
        "避免剧烈运动和重体力劳动，防止视网膜脱离。",
        "保持良好的用眼习惯，避免长时间近距离用眼。"
    ]
    
    metrics = recall_metrics.compute_metrics(question, docs)
    print(f"召回率指标: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
    
    print("\n=== 测试增强评测器 ===")
    # 注意：测试同基座模型评测需要加载大模型，这里仅展示接口
    # evaluator = EnhancedEvaluator(use_base_model_judge=False)
    # result = evaluator.evaluate_single(
    #     question=question,
    #     reference="高度近视患者应定期检查眼底，避免剧烈运动。",
    #     answer="建议高度近视患者每年检查眼底，避免剧烈运动。",
    #     retrieved_docs=docs
    # )
    # print(f"评测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
