# ============================================================
# 眼底健康问答系统评测脚本（完整版 / 语义修正版）
# ============================================================
# 评测指标：
# 1. ROUGE-L（文本重合度）
# 2. BERTScore F1（语义相似度）
# 3. 响应时间（工程效率）
# 4. 关键词【语义】覆盖率（Embedding）
# 5. 医学 Checklist 覆盖度（语义触发）
# ============================================================

import json
import time
import statistics
import random
import numpy as np
import torch
import argparse

from rouge_score import rouge_scorer
from bert_score import score
from sentence_transformers import SentenceTransformer, util

from qa_system import get_eye_qa_system

# ============================================================
# 一、命令行参数解析
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='眼底健康问答系统评测脚本')
parser.add_argument('--retrieval_mode', type=str, default='hybrid_rerank', 
                    choices=['vector', 'hybrid', 'hybrid_rerank'],
                    help='检索模式')
parser.add_argument('--use_query_rewrite', type=str2bool, default=True,
                    help='是否启用查询改写')
parser.add_argument('--test_file', type=str, default='test_set_en.json',
                    help='测试文件路径')
parser.add_argument('--fast_mode_n', type=int, default=20,
                    help='快速模式测试样本数')
parser.add_argument('--max_new_tokens', type=int, default=200,
                    help='最大生成 tokens 数')
parser.add_argument('--save_results', type=str2bool, default=True,
                    help='是否保存结果')
parser.add_argument('--embedding_threshold', type=float, default=0.55,
                    help='关键词语义覆盖率阈值')
args = parser.parse_args()

# ============================================================
# 二、基础配置
# ============================================================

TEST_FILE = args.test_file
FAST_MODE_N = args.fast_mode_n
MAX_NEW_TOKENS = args.max_new_tokens

SAVE_RESULTS = args.save_results
# 根据配置生成不同的结果文件名
SAVE_FILE = f"eval_{args.retrieval_mode}_rewrite_{args.use_query_rewrite}.json"

BERT_MODEL_EN = "roberta-large"
BERT_MODEL_ZH = "bert-base-chinese"

USE_EMBEDDING_KEYWORD = True

# ⚠️ 核心改动 1：阈值下调（理由后面解释）
EMBEDDING_THRESHOLD = args.embedding_threshold

USE_CHECKLIST = True

# 打印命令行参数，确认解析正确
print(f"\n=== 命令行参数 ===")
print(f"retrieval_mode: {args.retrieval_mode}")
print(f"use_query_rewrite: {args.use_query_rewrite}")
print(f"test_file: {args.test_file}")
print(f"fast_mode_n: {args.fast_mode_n}")
print(f"max_new_tokens: {args.max_new_tokens}")
print(f"save_results: {args.save_results}")
print(f"embedding_threshold: {args.embedding_threshold}")

# 初始化 QA 系统
print(f"\n=== 初始化 QA 系统 ===")
print(f"使用检索模式: {args.retrieval_mode}")
print(f"使用查询改写: {args.use_query_rewrite}")
eye_qa = get_eye_qa_system(
    retrieval_mode=args.retrieval_mode,
    use_query_rewrite=args.use_query_rewrite,
    stream_output=False
)

print(f"\n=== 评测配置 ===")
print(f"检索模式: {args.retrieval_mode}")
print(f"查询改写: {args.use_query_rewrite}")
print(f"测试文件: {TEST_FILE}")
print(f"测试样本数: {FAST_MODE_N}")
print(f"结果保存到: {SAVE_FILE}")
print(f"================\n")

# ============================================================
# 三、随机种子（保证可复现）
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

# ============================================================
# 四、Embedding 模型
# ============================================================

print("Loading embedding model (all-MiniLM-L6-v2)...")
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# 五、Checklist（语义触发版）
# ============================================================

CHECKLIST_TRIGGERS = {
    "condition": [
        "eye condition", "eye disease", "vision problem", "this condition"
    ],
    "cause": [
        "cause", "risk factor", "due to", "can be caused by"
    ],
    "symptom": [
        "symptom", "sign", "experience", "may feel", "can cause"
    ],
    "advice": [
        "recommend", "should", "suggest", "advised to"
    ],
    "medical_help": [
        "see a doctor", "consult", "seek medical", "eye specialist"
    ],
    "disclaimer": [
        "not a diagnosis", "not medical advice", "informational purposes"
    ]
}

# ============================================================
# 六、加载测试集
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

if FAST_MODE_N:
    test_data = test_data[:FAST_MODE_N]

print(f"\nLoaded {len(test_data)} test questions.\n")

# ============================================================
# 七、评测工具初始化
# ============================================================

rouge_eval = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rouge_ls = []
bert_f1s = []
response_times = []
keyword_covs = []
checklist_covs = []

eval_details = []

# ============================================================
# 八、辅助函数
# ============================================================

def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fa5" for c in text)


def embedding_coverage(items, answer, threshold):
    """
    关键词语义覆盖率（修正版）：
    - keyword 与 answer 使用 embedding 相似度
    - 取 max similarity（而不是平均）
    """
    if not items or not answer:
        return None

    answer_emb = emb_model.encode(answer, convert_to_tensor=True)
    hit = 0

    for item in items:
        item_emb = emb_model.encode(item, convert_to_tensor=True)
        sim = util.cos_sim(item_emb, answer_emb).max().item()
        if sim >= threshold:
            hit += 1

    return hit / len(items)


def checklist_trigger_coverage(triggers, answer):
    """
    Checklist 只要求「至少一次语义触发」
    """
    answer = answer.lower()
    hit = 0
    for _, phrases in triggers.items():
        if any(p in answer for p in phrases):
            hit += 1
    return hit / len(triggers)

# ============================================================
# 九、开始评测
# ============================================================

for idx, item in enumerate(test_data, 1):
    question = item["question"]
    reference = item["reference_answer"]
    keywords = item.get("expected_keywords", [])

    print(f"\n[{idx}/{len(test_data)}] Question:")
    print(question)

    start = time.time()
    system_answer = eye_qa(
        question,
        max_new_tokens=MAX_NEW_TOKENS
    )
    rt = time.time() - start
    response_times.append(rt)

    print(f"→ Answer generated ({rt:.2f}s)")

    # ---------- ROUGE ----------
    rouge_l = rouge_eval.score(
        reference,
        system_answer
    )["rougeL"].fmeasure
    rouge_ls.append(rouge_l)

    # ---------- BERTScore ----------
    is_zh = contains_chinese(reference)
    bert_model = BERT_MODEL_ZH if is_zh else BERT_MODEL_EN
    lang = "zh" if is_zh else "en"

    _, _, F1 = score(
        [system_answer],
        [reference],
        lang=lang,
        model_type=bert_model,
        verbose=False
    )
    bert_f1s.append(F1.mean().item())

    # ---------- Keyword coverage（修正版） ----------
    if USE_EMBEDDING_KEYWORD:
        cov = embedding_coverage(
            keywords,
            system_answer,
            EMBEDDING_THRESHOLD
        )
        if cov is not None:
            keyword_covs.append(cov)

    # ---------- Checklist coverage ----------
    if USE_CHECKLIST:
        cc = checklist_trigger_coverage(
            CHECKLIST_TRIGGERS,
            system_answer
        )
        checklist_covs.append(cc)

    eval_details.append({
        "id": item.get("id", idx),
        "question": question,
        "system_answer": system_answer,
        "reference_answer": reference,
        "rouge_l": rouge_l,
        "bert_f1": bert_f1s[-1],
        "response_time": rt,
        "keyword_coverage": cov if USE_EMBEDDING_KEYWORD else None,
        "checklist_coverage": checklist_covs[-1]
    })

    print("→ Metrics computed")

# ============================================================
# 十、统计结果
# ============================================================

def mean(x):
    return statistics.mean(x) if x else 0.0

print("\n================ Evaluation Summary ================\n")
print(f"ROUGE-L: {mean(rouge_ls):.3f}")
print(f"BERTScore F1: {mean(bert_f1s):.3f}")
print(f"Avg response time: {mean(response_times):.2f}s")
print(f"Keyword coverage (embed): {mean(keyword_covs):.3f}")
print(f"Checklist coverage: {mean(checklist_covs):.3f}")

# ============================================================
# 十一、保存结果
# ============================================================

if SAVE_RESULTS:
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "rouge_l": mean(rouge_ls),
                    "bert_f1": mean(bert_f1s),
                    "avg_response_time": mean(response_times),
                    "keyword_coverage": mean(keyword_covs),
                    "checklist_coverage": mean(checklist_covs),
                },
                "details": eval_details
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nResults saved to {SAVE_FILE}")
