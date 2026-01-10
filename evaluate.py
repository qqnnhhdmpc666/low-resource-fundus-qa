import json
import time
import statistics
import string
import random
import numpy as np
import torch

from rouge_score import rouge_scorer
from bert_score import score
from qa_system import eye_qa   # 接口保持不变

# ===========================
# 一、基础配置（你主要改这里）
# ===========================
TEST_FILE = "test_set.json"        # 你的测试集
FAST_MODE_N = 20                  # 你现在就是 20 条
MAX_NEW_TOKENS = 200

USE_EMBEDDING_KEYWORD = True      # 是否使用语义关键词
EMBEDDING_THRESHOLD = 0.8

SAVE_RESULTS = True               # 是否保存结果
SAVE_FILE = "eval_baseline.json"

BERT_MODEL = "roberta-large"      # 固定 BERTScore 模型（重要）

# ===========================
# 二、固定随机种子（保证可复现）
# ===========================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

# ===========================
# 三、Embedding 模型（可选）
# ===========================
if USE_EMBEDDING_KEYWORD:
    from sentence_transformers import SentenceTransformer, util
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===========================
# 四、加载测试集
# ===========================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

if FAST_MODE_N:
    test_data = test_data[:FAST_MODE_N]

print(f"Loaded {len(test_data)} test questions.\n")

# ===========================
# 五、评测工具初始化
# ===========================
rouge_eval = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

keyword_cov_string = []
keyword_cov_embed = []
rouge_ls = []
bert_f1s = []
response_times = []

eval_details = []

# ===========================
# 六、关键词覆盖函数
# ===========================
def keyword_coverage_string(keywords, answer):
    """最传统的字符串关键词匹配"""
    if not keywords or not answer:
        return None

    answer_proc = answer.lower().translate(
        str.maketrans("", "", string.punctuation)
    )

    hit = 0
    for kw in keywords:
        kw_proc = kw.lower().translate(
            str.maketrans("", "", string.punctuation)
        )
        if kw_proc in answer_proc:
            hit += 1

    return hit / len(keywords)


def keyword_coverage_embedding(keywords, answer, threshold=EMBEDDING_THRESHOLD):
    """基于语义相似度的关键词覆盖"""
    if not keywords or not answer:
        return None

    answer_emb = emb_model.encode(answer, convert_to_tensor=True)
    hit = 0

    for kw in keywords:
        kw_emb = emb_model.encode(kw, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(kw_emb, answer_emb).item()
        if sim >= threshold:
            hit += 1

    return hit / len(keywords)

# ===========================
# 七、开始评测
# ===========================
for idx, item in enumerate(test_data, 1):
    question = item["question"]
    reference = item["reference_answer"]
    keywords = item.get("expected_keywords", [])

    print(f"[{idx}/{len(test_data)}] Q: {question}")

    # ---- 调用 QA 系统（只调用一次！） ----
    start = time.time()
    system_answer = eye_qa(question, max_new_tokens=MAX_NEW_TOKENS)
    rt = time.time() - start

    response_times.append(rt)

    # ---- 关键词覆盖 ----
    cov_str = keyword_coverage_string(keywords, system_answer)
    cov_emb = None
    if USE_EMBEDDING_KEYWORD:
        cov_emb = keyword_coverage_embedding(keywords, system_answer)

    if cov_str is not None:
        keyword_cov_string.append(cov_str)
    if cov_emb is not None:
        keyword_cov_embed.append(cov_emb)

    # ---- ROUGE-L ----
    rouge_l = rouge_eval.score(reference, system_answer)["rougeL"].fmeasure
    rouge_ls.append(rouge_l)

    # ---- BERTScore ----
    _, _, F1 = score(
        [system_answer],
        [reference],
        lang="en",
        model_type=BERT_MODEL,
        verbose=False
    )
    bert_f1 = F1.mean().item()
    bert_f1s.append(bert_f1)

    # ---- 保存详细结果 ----
    if SAVE_RESULTS:
        eval_details.append({
            "id": item.get("id", idx),
            "question": question,
            "reference_answer": reference,
            "system_answer": system_answer,
            "keyword_coverage_string": cov_str,
            "keyword_coverage_embedding": cov_emb,
            "rouge_l": rouge_l,
            "bert_f1": bert_f1,
            "response_time_sec": rt
        })

# ===========================
# 八、统计 & 输出
# ===========================
print("\n================ Evaluation Summary ================\n")

def mean_std(x):
    return statistics.mean(x), statistics.stdev(x) if len(x) > 1 else 0.0

if keyword_cov_string:
    m, s = mean_std(keyword_cov_string)
    print(f"Keyword Coverage (string): {m:.3f} ± {s:.3f}")

if keyword_cov_embed:
    m, s = mean_std(keyword_cov_embed)
    print(f"Keyword Coverage (embedding): {m:.3f} ± {s:.3f}")

m, s = mean_std(rouge_ls)
print(f"ROUGE-L: {m:.3f} ± {s:.3f}")

m, s = mean_std(bert_f1s)
print(f"BERTScore F1: {m:.3f} ± {s:.3f}")

print(f"Avg response time (s): {statistics.mean(response_times):.2f}")

# ===========================
# 九、保存结果（可选）
# ===========================
if SAVE_RESULTS:
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "keyword_string": statistics.mean(keyword_cov_string) if keyword_cov_string else None,
                "keyword_embedding": statistics.mean(keyword_cov_embed) if keyword_cov_embed else None,
                "rouge_l": statistics.mean(rouge_ls),
                "bert_f1": statistics.mean(bert_f1s),
                "response_time": statistics.mean(response_times)
            },
            "details": eval_details
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to {SAVE_FILE}")
