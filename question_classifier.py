# ============================================================
# question_classifier.py
# 基于NLI模型的问题类型分类器
# 用于Type-Aware检索策略路由
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re


class QuestionClassifier:
    """
    基于NLI（自然语言推理）模型的问题分类器
    将问题分类为：emergency, disease_definition, daily_advice, decision
    """
    
    # 问题类型定义和描述（用于NLI推理）
    QUESTION_TYPES = {
        "emergency": {
            "description": "紧急医疗情况，需要立即就医，如突发失明、剧烈眼痛、大量出血",
            "keywords": ["突然", "立即", "急诊", "失明", "剧痛", "出血", "紧急", "urgent", "sudden", "severe pain", "blind"]
        },
        "disease_definition": {
            "description": "疾病定义、病因、症状解释，如什么是糖尿病视网膜病变",
            "keywords": ["什么是", "定义", "原因", "机制", "症状", "什么是", "what is", "definition", "cause", "symptom"]
        },
        "daily_advice": {
            "description": "日常护理建议、预防措施、生活习惯，如高度近视日常注意事项",
            "keywords": ["日常", "注意", "预防", "饮食", "运动", "护理", "建议", "daily", "care", "prevent", "diet"]
        },
        "decision": {
            "description": "治疗决策、手术建议、方案选择，如是否建议手术治疗",
            "keywords": ["应该", "建议", "手术", "治疗", "选择", "需要", "是否", "should", "surgery", "treatment", "recommend"]
        }
    }
    
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        """
        初始化NLI分类器
        默认使用DeBERTa-v3-base-mnli-fever-anli，在NLI任务上表现优秀
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Classifier] Loading NLI model: {model_name}")
        print(f"[Classifier] Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 预定义的问题类型假设
        self.hypotheses = {
            qtype: f"This question is asking about {info['description']}"
            for qtype, info in self.QUESTION_TYPES.items()
        }
        
        print("[Classifier] NLI model loaded successfully")
    
    def classify(self, question: str) -> str:
        """
        对问题进行分类
        
        Args:
            question: 用户问题
            
        Returns:
            问题类型: emergency | disease_definition | daily_advice | decision
        """
        # 首先进行关键词快速匹配
        keyword_match = self._keyword_classify(question)
        if keyword_match:
            return keyword_match
        
        # 使用NLI模型进行精确分类
        return self._nli_classify(question)
    
    def _keyword_classify(self, question: str) -> str:
        """
        基于关键词的快速分类（作为NLI的辅助）
        """
        q_lower = question.lower()
        
        # 紧急关键词（最高优先级）
        emergency_keywords = ["突然", "立即", "急诊", "失明", "剧痛", "出血不止", 
                             "emergency", "sudden blindness", "severe pain", "urgent"]
        for kw in emergency_keywords:
            if kw in q_lower:
                return "emergency"
        
        return None
    
    def _nli_classify(self, question: str) -> str:
        """
        使用NLI模型进行分类
        通过判断问题与各类别描述的蕴含关系来确定类型
        """
        scores = {}
        
        for qtype, hypothesis in self.hypotheses.items():
            # NLI推理: premise=question, hypothesis=type_description
            inputs = self.tokenizer(
                question,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # DeBERTa MNLI模型输出: [contradiction, neutral, entailment]
                probs = torch.softmax(logits, dim=-1)
                
                # 使用entailment概率作为匹配分数
                entailment_score = probs[0][2].item()
                scores[qtype] = entailment_score
        
        # 返回蕴含分数最高的类型
        best_type = max(scores, key=scores.get)
        
        # 如果所有分数都很低，默认使用hybrid_rerank
        if scores[best_type] < 0.3:
            return "decision"  # 默认使用decision类型（hybrid_rerank）
        
        return best_type
    
    def get_type_confidence(self, question: str) -> dict:
        """
        获取问题分类的置信度分数
        
        Returns:
            dict: {type: confidence_score}
        """
        scores = {}
        
        for qtype, hypothesis in self.hypotheses.items():
            inputs = self.tokenizer(
                question,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                scores[qtype] = {
                    "entailment": probs[0][2].item(),
                    "neutral": probs[0][1].item(),
                    "contradiction": probs[0][0].item()
                }
        
        return scores


# 基于实验数据的最优策略映射
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


def get_optimal_strategy(question_type: str) -> dict:
    """
    根据问题类型获取最优检索策略
    
    Args:
        question_type: 问题类型
        
    Returns:
        dict: 包含retrieval_mode和reason的字典
    """
    return TYPE_AWARE_STRATEGY.get(
        question_type, 
        {"retrieval_mode": "hybrid_rerank", "reason": "默认使用最优策略"}
    )


if __name__ == "__main__":
    # 测试分类器
    classifier = QuestionClassifier()
    
    test_questions = [
        "我突然看不见了，应该怎么办？",  # emergency
        "什么是糖尿病视网膜病变？",  # disease_definition
        "高度近视患者平时应该注意什么？",  # daily_advice
        "我这种情况需要做手术吗？",  # decision
    ]
    
    print("\n=== 问题分类测试 ===")
    for q in test_questions:
        qtype = classifier.classify(q)
        strategy = get_optimal_strategy(qtype)
        confidence = classifier.get_type_confidence(q)
        
        print(f"\n问题: {q}")
        print(f"分类: {qtype}")
        print(f"策略: {strategy['retrieval_mode']}")
        print(f"原因: {strategy['reason']}")
        print(f"置信度: {confidence}")
