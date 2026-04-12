# ============================================================
# evaluator_scientific.py
# 科研级评测系统 - 工业标准实现
# 
# 评测维度：
# 1. 检索性能指标 (Retrieval Metrics)
# 2. 排名质量指标 (Ranking Metrics)  
# 3. 生成质量指标 (Generation Metrics)
# 4. 统计显著性检验 (Statistical Significance Testing)
# ============================================================

import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy import stats
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class RankingMetrics:
    """
    排名质量指标 - 评估检索排序的准确性
    适用于信息检索领域的标准指标
    """
    
    @staticmethod
    def mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Mean Reciprocal Rank (平均倒数排名)
        
        MRR = 1/|Q| * Σ(1/rank_i)
        其中rank_i是第一个相关文档的排名位置
        
        Args:
            retrieved_docs: 按相关性排序的召回文档列表
            relevant_docs: 相关文档集合（参考答案匹配的文档）
            
        Returns:
            MRR分数 [0, 1]，越高越好
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Discounted Cumulative Gain (折损累计增益)
        
        DCG@k = Σ(rel_i / log2(i+1)) for i=1 to k
        
        Args:
            relevances: 相关性分数列表
            k: 截断位置
            
        Returns:
            DCG分数
        """
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], 
                  doc_relevance: Dict[str, float], 
                  k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain (归一化折损累计增益)
        
        NDCG@k = DCG@k / IDCG@k
        
        其中IDCG是理想排序下的DCG值
        
        Args:
            retrieved_docs: 召回文档列表（按排名排序）
            doc_relevance: 文档相关性映射 {doc: relevance_score}
            k: 截断位置
            
        Returns:
            NDCG@k分数 [0, 1]，越高越好
        """
        if not retrieved_docs or not doc_relevance:
            return 0.0
        
        # 计算DCG
        relevances = [doc_relevance.get(doc, 0.0) for doc in retrieved_docs[:k]]
        dcg = RankingMetrics.dcg_at_k(relevances, k)
        
        # 计算IDCG（理想排序）
        ideal_relevances = sorted(doc_relevance.values(), reverse=True)[:k]
        idcg = RankingMetrics.dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], 
                       relevant_docs: List[str], 
                       k: int) -> float:
        """
        Precision@k (精确率@k)
        
        P@k = |{相关文档} ∩ {前k个召回文档}| / k
        
        Args:
            retrieved_docs: 召回文档列表
            relevant_docs: 相关文档集合
            k: 截断位置
            
        Returns:
            Precision@k分数 [0, 1]
        """
        if k == 0 or not retrieved_docs:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = len(retrieved_k & relevant_set)
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], 
                    relevant_docs: List[str], 
                    k: int) -> float:
        """
        Recall@k (召回率@k)
        
        R@k = |{相关文档} ∩ {前k个召回文档}| / |{相关文档}|
        
        Args:
            retrieved_docs: 召回文档列表
            relevant_docs: 相关文档集合
            k: 截断位置
            
        Returns:
            Recall@k分数 [0, 1]
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = len(retrieved_k & relevant_set)
        return relevant_retrieved / len(relevant_set)
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], 
                         relevant_docs: List[str]) -> float:
        """
        Average Precision (AP) - 平均精确率
        
        AP = Σ(P@k * rel(k)) / |{相关文档}|
        
        其中rel(k)是指第k个文档是否相关（0或1）
        
        Args:
            retrieved_docs: 召回文档列表（按排名排序）
            relevant_docs: 相关文档集合
            
        Returns:
            AP分数 [0, 1]
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        num_relevant = len(relevant_set)
        
        precisions = []
        num_relevant_so_far = 0
        
        for k, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                num_relevant_so_far += 1
                precisions.append(num_relevant_so_far / k)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / num_relevant
    
    @staticmethod
    def map_score(retrieved_docs_list: List[List[str]], 
                  relevant_docs_list: List[List[str]]) -> float:
        """
        Mean Average Precision (MAP) - 平均精确率的均值
        
        MAP = 1/|Q| * Σ(AP_q) for all queries q
        
        Args:
            retrieved_docs_list: 每个查询的召回文档列表
            relevant_docs_list: 每个查询的相关文档列表
            
        Returns:
            MAP分数 [0, 1]
        """
        if not retrieved_docs_list or not relevant_docs_list:
            return 0.0
        
        aps = []
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            ap = RankingMetrics.average_precision(retrieved, relevant)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0


class RetrievalMetrics:
    """
    检索性能指标 - 评估召回质量和多样性
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
    
    def compute_semantic_similarity(self, query: str, docs: List[str]) -> List[float]:
        """计算查询与文档的语义相似度"""
        if not docs:
            return []
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        doc_embs = self.model.encode(docs, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        return similarities.cpu().tolist()
    
    def compute_metrics(self, 
                       query: str, 
                       retrieved_docs: List[str],
                       relevant_docs: Optional[List[str]] = None) -> Dict:
        """
        计算全面的检索指标
        
        Returns:
            {
                "semantic_recall_precision": float,  # 语义召回精确度
                "semantic_recall_diversity": float,  # 语义召回多样性
                "semantic_coverage": float,          # 语义覆盖率
                "semantic_recall": float,            # 综合语义召回率
                "top_k_similarities": List[float],   # Top-K相似度列表
                "mean_similarity": float,            # 平均相似度
                "std_similarity": float,             # 相似度标准差
            }
        """
        if not retrieved_docs:
            return {
                "semantic_recall_precision": 0.0,
                "semantic_recall_diversity": 0.0,
                "semantic_coverage": 0.0,
                "semantic_recall": 0.0,
                "top_k_similarities": [],
                "mean_similarity": 0.0,
                "std_similarity": 0.0,
            }
        
        # 计算语义相似度
        similarities = self.compute_semantic_similarity(query, retrieved_docs)
        similarities_tensor = torch.tensor(similarities)
        
        # 1. 语义召回精确度 (平均相似度)
        semantic_precision = np.mean(similarities)
        
        # 2. 语义召回多样性 (文档间差异)
        if len(retrieved_docs) > 1:
            doc_embs = self.model.encode(retrieved_docs, convert_to_tensor=True)
            doc_sims = util.cos_sim(doc_embs, doc_embs)
            mask = ~torch.eye(len(doc_embs), dtype=torch.bool, device=doc_embs.device)
            avg_doc_sim = doc_sims[mask].mean().item()
            semantic_diversity = 1.0 - avg_doc_sim
        else:
            semantic_diversity = 0.0
        
        # 3. 语义覆盖率 (相似度>阈值的文档比例)
        threshold = 0.5
        semantic_coverage = (similarities_tensor > threshold).float().mean().item()
        
        # 4. 综合语义召回率
        semantic_recall = semantic_precision * semantic_coverage * (1.0 + semantic_diversity) / 2.0
        
        return {
            "semantic_recall_precision": round(semantic_precision, 4),
            "semantic_recall_diversity": round(semantic_diversity, 4),
            "semantic_coverage": round(semantic_coverage, 4),
            "semantic_recall": round(semantic_recall, 4),
            "top_k_similarities": [round(s, 4) for s in similarities[:5]],
            "mean_similarity": round(semantic_precision, 4),
            "std_similarity": round(np.std(similarities), 4),
        }


class StatisticalSignificanceTest:
    """
    统计显著性检验 - 比较不同方法的性能差异
    """
    
    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
        """
        配对t检验 - 比较两组配对样本的均值差异
        
        H0: μ_a = μ_b (两组无显著差异)
        H1: μ_a ≠ μ_b (两组有显著差异)
        
        Args:
            scores_a: 方法A的分数列表
            scores_b: 方法B的分数列表
            
        Returns:
            (t统计量, p值)
        """
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return 0.0, 1.0
        
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        return t_stat, p_value
    
    @staticmethod
    def wilcoxon_signed_rank_test(scores_a: List[float], 
                                   scores_b: List[float]) -> Tuple[float, float]:
        """
        Wilcoxon符号秩检验 - 非参数配对检验
        
        适用于数据不满足正态分布的情况
        
        Args:
            scores_a: 方法A的分数列表
            scores_b: 方法B的分数列表
            
        Returns:
            (统计量, p值)
        """
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return 0.0, 1.0
        
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
            return stat, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
        """
        解释p值的统计显著性
        
        Returns:
            显著性等级描述
        """
        if p_value < 0.001:
            return "*** (p<0.001, 高度显著)"
        elif p_value < 0.01:
            return "** (p<0.01, 非常显著)"
        elif p_value < alpha:
            return "* (p<0.05, 显著)"
        else:
            return "ns (不显著)"


class ScientificEvaluator:
    """
    科研级综合评测器
    
    整合检索指标、排名指标、生成质量指标和统计检验
    """
    
    def __init__(self, 
                 use_base_model_judge: bool = True,
                 base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化科研级评测器
        
        Args:
            use_base_model_judge: 是否使用同基座模型评测
            base_model_name: 基座模型名称
        """
        self.retrieval_metrics = RetrievalMetrics()
        self.ranking_metrics = RankingMetrics()
        self.stat_test = StatisticalSignificanceTest()
        
        if use_base_model_judge:
            print(f"[ScientificEvaluator] Loading base model judge: {base_model_name}")
            self.base_judge = BaseModelJudgeScientific(base_model_name)
        else:
            self.base_judge = None
    
    def evaluate_single(self,
                       question: str,
                       reference: str,
                       answer: str,
                       retrieved_docs: List[str],
                       doc_scores: List[float],
                       relevant_docs: Optional[List[str]] = None,
                       response_time: Optional[float] = None) -> Dict:
        """
        对单个样本进行全面的科研级评测
        
        Args:
            question: 问题
            reference: 参考答案
            answer: 系统生成的答案
            retrieved_docs: 召回的文档列表（按排名排序）
            doc_scores: 文档相关性分数
            relevant_docs: 相关文档集合（用于计算排名指标）
            response_time: 响应时间
            
        Returns:
            完整的评测结果字典
        """
        results = {}
        
        # 1. 检索性能指标
        retrieval_results = self.retrieval_metrics.compute_metrics(
            question, retrieved_docs, relevant_docs
        )
        results["retrieval_metrics"] = retrieval_results
        
        # 2. 排名质量指标（如果提供了相关文档）
        if relevant_docs:
            # 构建文档相关性映射（简化：相关文档给1分，不相关给0分）
            doc_relevance = {doc: 1.0 if doc in relevant_docs else 0.0 
                           for doc in retrieved_docs}
            
            ranking_results = {
                "mrr": round(RankingMetrics.mrr(retrieved_docs, relevant_docs), 4),
                "ndcg@5": round(RankingMetrics.ndcg_at_k(retrieved_docs, doc_relevance, k=5), 4),
                "ndcg@10": round(RankingMetrics.ndcg_at_k(retrieved_docs, doc_relevance, k=10), 4),
                "precision@5": round(RankingMetrics.precision_at_k(retrieved_docs, relevant_docs, k=5), 4),
                "recall@5": round(RankingMetrics.recall_at_k(retrieved_docs, relevant_docs, k=5), 4),
                "ap": round(RankingMetrics.average_precision(retrieved_docs, relevant_docs), 4),
            }
            results["ranking_metrics"] = ranking_results
        
        # 3. 同基座模型评测
        if self.base_judge:
            judge_results = self.base_judge.evaluate(
                question, reference, answer, retrieved_docs
            )
            results["generation_metrics"] = judge_results
        
        # 4. 工程指标
        if response_time:
            results["efficiency_metrics"] = {
                "response_time": response_time,
                "num_retrieved_docs": len(retrieved_docs),
            }
        
        return results
    
    def evaluate_batch(self, predictions: List[Dict]) -> Dict:
        """
        批量评测并生成科研级报告
        
        Args:
            predictions: 预测结果列表，每个元素包含：
                {
                    "question": str,
                    "reference": str,
                    "answer": str,
                    "retrieved_docs": List[str],
                    "doc_scores": List[float],
                    "relevant_docs": List[str] (可选),
                    "response_time": float (可选)
                }
        """
        all_results = []
        
        for pred in predictions:
            result = self.evaluate_single(
                question=pred["question"],
                reference=pred["reference"],
                answer=pred["answer"],
                retrieved_docs=pred.get("retrieved_docs", []),
                doc_scores=pred.get("doc_scores", []),
                relevant_docs=pred.get("relevant_docs"),
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
            "retrieval_metrics": {},
            "ranking_metrics": {},
            "generation_metrics": {},
            "efficiency_metrics": {}
        }
        
        # 检索指标汇总
        retrieval_keys = ["semantic_recall_precision", "semantic_recall_diversity", 
                         "semantic_coverage", "semantic_recall", "mean_similarity"]
        for key in retrieval_keys:
            values = [r["retrieval_metrics"][key] for r in results 
                     if "retrieval_metrics" in r and key in r["retrieval_metrics"]]
            if values:
                summary["retrieval_metrics"][key] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "median": round(np.median(values), 4),
                    "min": round(np.min(values), 4),
                    "max": round(np.max(values), 4),
                }
        
        # 排名指标汇总
        ranking_keys = ["mrr", "ndcg@5", "ndcg@10", "precision@5", "recall@5", "ap"]
        for key in ranking_keys:
            values = [r["ranking_metrics"][key] for r in results 
                     if "ranking_metrics" in r and key in r["ranking_metrics"]]
            if values:
                summary["ranking_metrics"][key] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "median": round(np.median(values), 4),
                }
        
        # 生成指标汇总
        if self.base_judge:
            gen_keys = ["correctness", "completeness", "safety", "helpfulness", 
                       "groundedness", "overall"]
            for key in gen_keys:
                values = [r["generation_metrics"][key] for r in results 
                         if "generation_metrics" in r and key in r["generation_metrics"]]
                if values:
                    summary["generation_metrics"][key] = {
                        "mean": round(np.mean(values), 4),
                        "std": round(np.std(values), 4),
                    }
        
        return summary
    
    def compare_methods(self,
                       method_a_results: List[Dict],
                       method_b_results: List[Dict],
                       metric_key: str = "overall") -> Dict:
        """
        比较两种方法的性能差异（统计显著性检验）
        
        Args:
            method_a_results: 方法A的评测结果列表
            method_b_results: 方法B的评测结果列表
            metric_key: 要比较的指标键名
            
        Returns:
            统计检验结果
        """
        # 提取分数
        scores_a = [r["generation_metrics"][metric_key] 
                   for r in method_a_results if "generation_metrics" in r]
        scores_b = [r["generation_metrics"][metric_key] 
                   for r in method_b_results if "generation_metrics" in r]
        
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return {"error": "样本数量不匹配或不足"}
        
        # 配对t检验
        t_stat, p_value_t = self.stat_test.paired_t_test(scores_a, scores_b)
        
        # Wilcoxon检验
        w_stat, p_value_w = self.stat_test.wilcoxon_signed_rank_test(scores_a, scores_b)
        
        # 计算效应量 (Cohen's d)
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        pooled_std = np.sqrt((np.std(scores_a)**2 + np.std(scores_b)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            "metric": metric_key,
            "method_a_mean": round(np.mean(scores_a), 4),
            "method_b_mean": round(np.mean(scores_b), 4),
            "mean_difference": round(mean_diff, 4),
            "paired_t_test": {
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value_t, 4),
                "significance": self.stat_test.interpret_p_value(p_value_t)
            },
            "wilcoxon_test": {
                "statistic": round(w_stat, 4),
                "p_value": round(p_value_w, 4),
                "significance": self.stat_test.interpret_p_value(p_value_w)
            },
            "effect_size": {
                "cohens_d": round(cohens_d, 4),
                "interpretation": "大效应" if abs(cohens_d) > 0.8 else 
                                "中等效应" if abs(cohens_d) > 0.5 else 
                                "小效应" if abs(cohens_d) > 0.2 else "可忽略"
            }
        }


class BaseModelJudgeScientific:
    """
    科研级同基座模型评测器
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
    
    def evaluate(self, 
                question: str, 
                reference: str, 
                answer: str,
                retrieved_docs: List[str] = None) -> Dict:
        """使用同基座模型进行多维度评测"""
        
        prompt = self._build_prompt(question, reference, answer, retrieved_docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        judgment_text = response[len(prompt):].strip()
        
        return self._parse_judgment(judgment_text)
    
    def _build_prompt(self, question, reference, answer, retrieved_docs):
        """构建评测Prompt"""
        doc_context = ""
        if retrieved_docs:
            doc_context = "\n".join([f"[{i+1}] {doc[:200]}..." 
                                    for i, doc in enumerate(retrieved_docs[:3])])
        
        return f"""You are an expert medical AI evaluator. Evaluate the following AI-generated answer.

## Evaluation Criteria (Score 1-5):

1. **Correctness**: Medical accuracy and consistency with reference
2. **Completeness**: Coverage of key points from reference  
3. **Safety**: Medical safety and responsibility
4. **Helpfulness**: Direct usefulness for the user's question
5. **Groundedness**: Support from retrieved documents

## Input:

**Question**: {question}

**Reference**: {reference}

**AI Answer**: {answer}

**Retrieved Docs**: {doc_context if doc_context else "None"}

## Output (JSON format):
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

Evaluation:"""
    
    def _parse_judgment(self, text):
        """解析评测结果"""
        try:
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                for key in ["correctness", "completeness", "safety", "helpfulness", "groundedness"]:
                    result.setdefault(key, 3)
                result.setdefault("overall", 3.0)
                result.setdefault("reasoning", "Parsed")
                return result
        except:
            pass
        
        return {k: 3 for k in ["correctness", "completeness", "safety", "helpfulness", "groundedness"]} | \
               {"overall": 3.0, "reasoning": "Default"}


if __name__ == "__main__":
    # 测试示例
    print("=== 科研级评测系统测试 ===\n")
    
    # 测试排名指标
    print("1. 测试排名指标:")
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc5"]
    
    print(f"  MRR: {RankingMetrics.mrr(retrieved, relevant):.4f}")
    print(f"  P@3: {RankingMetrics.precision_at_k(retrieved, relevant, 3):.4f}")
    print(f"  R@3: {RankingMetrics.recall_at_k(retrieved, relevant, 3):.4f}")
    print(f"  AP: {RankingMetrics.average_precision(retrieved, relevant):.4f}")
    
    # 测试NDCG
    doc_rel = {"doc1": 3.0, "doc2": 2.0, "doc3": 3.0, "doc4": 0.0, "doc5": 1.0}
    print(f"  NDCG@5: {RankingMetrics.ndcg_at_k(retrieved, doc_rel, 5):.4f}")
    
    # 测试统计检验
    print("\n2. 测试统计显著性检验:")
    scores_a = [4.5, 4.2, 4.8, 4.3, 4.6]
    scores_b = [4.1, 3.9, 4.2, 4.0, 4.1]
    
    stat_test = StatisticalSignificanceTest()
    t_stat, p_val = stat_test.paired_t_test(scores_a, scores_b)
    print(f"  配对t检验: t={t_stat:.4f}, p={p_val:.4f}, {stat_test.interpret_p_value(p_val)}")
    
    print("\n=== 测试完成 ===")
