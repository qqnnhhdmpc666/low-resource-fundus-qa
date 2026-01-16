import json
import time
import statistics
import string
import random
import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score
from qa_system import eye_qa  # 接口保持不变
from sentence_transformers import SentenceTransformer, util  # 用于语义关键词和 Judge 辅助

# ===========================
# 一、基础配置（你主要改这里）
# ===========================
TEST_FILE = "test_set.json"  # 你的测试集
FAST_MODE_N = 20  # 你现在就是 20 条
MAX_NEW_TOKENS = 200
USE_EMBEDDING_KEYWORD = True  # 是否使用语义关键词
EMBEDDING_THRESHOLD = 0.8
SAVE_RESULTS = True  # 是否保存结果
SAVE_FILE = "eval_baseline.json"
BERT_MODEL = "roberta-large"  # 固定 BERTScore 模型（重要）
USE_JUDGE = True  # 是否使用 LLM-as-a-Judge
USE_CHECKLIST = True  # 是否使用 Checklist 评估
JUDGE_MAX_TOKENS = 400  # Judge 生成上限
CHECKLIST_ITEMS = [  # 医疗 QA Checklist 项目
    "是否明确疾病名称和定义",
    "是否提及主要病因或危险因素",
    "是否给出诊断或检查建议",
    "是否列出治疗方案",
    "是否提及注意事项或生活指导",
    "是否提示及时就医",
    "是否有免责声明",
    "是否避免绝对化表述（如‘一定能治好’）",
]

# ===========================
# 二、固定随机种子（保证可复现）
# ===========================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

# ===========================
# 三、Embedding 模型（用于语义关键词/Checklist/Judge 辅助）
# ===========================
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
judge_scores = []  # 新增 Judge 总体质量分数
checklist_coverages = []  # 新增 Checklist 覆盖率
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
# 七、LLM-as-a-Judge 函数（新增）
# ===========================
def llm_judge(reference, system_answer):
    judge_prompt = f"""
你是一位资深眼科主任医师，现在评估下面AI回答的质量。
参考答案：{reference}
AI回答：{system_answer}

请从以下维度打分（1-10分，满分10），并给出简短理由：
1. 专业准确性（是否符合医学事实，无错误）
2. 完整性（是否覆盖核心诊断、治疗、注意事项、就医建议）
3. 安全性（是否有误导风险、是否提醒就医）
4. 通俗易懂度（患者能否看懂，非专业术语过多）
5. 总体质量

格式示例：
专业准确性：8/10 理由：...
完整性：...
...
总体质量：...
总分：...
"""
    # 用你的 model/tokenizer 打分
    judge_inputs = tokenizer(judge_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        judge_output = model.generate(
            **judge_inputs,
            max_new_tokens=JUDGE_MAX_TOKENS,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    judge_text = tokenizer.decode(judge_output[0], skip_special_tokens=True)
    # 简单解析总分（可优化为正则）
    try:
        total_score = float(judge_text.split("总分：")[-1].strip().split("/")[0])
    except:
        total_score = 0.0
    return total_score, judge_text

# ===========================
# 八、Checklist 评估函数（新增）
# ===========================
def checklist_coverage(checklist, answer, threshold=0.7):
    if not checklist or not answer:
        return None
    answer_emb = emb_model.encode(answer, convert_to_tensor=True)
    hit = 0
    for item in checklist:
        item_emb = emb_model.encode(item, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(item_emb, answer_emb).item()
        if sim >= threshold:
            hit += 1
    return hit / len(checklist)

# ===========================
# 九、开始评测
# ===========================
for idx, item in enumerate(test_data, 1):
    question = item["question"]
    reference = item["reference_answer"]
    keywords = item.get("expected_keywords", [])
    print(f"[{idx}/{len(test_data)}] Q: {question}")

    # ---- 调用 QA 系统 ----
    start = time.time()
    system_answer = eye_qa(question, max_new_tokens=MAX_NEW_TOKENS)
    rt = time.time() - start
    response_times.append(rt)

    # ---- 关键词覆盖 ----
    cov_str = keyword_coverage_string(keywords, system_answer)
    cov_emb = keyword_coverage_embedding(keywords, system_answer) if USE_EMBEDDING_KEYWORD else None
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

    # ---- LLM-as-a-Judge (新增) ----
    if USE_JUDGE:
        judge_score, judge_reason = llm_judge(reference, system_answer)
        judge_scores.append(judge_score)

    # ---- Checklist 覆盖率 (新增) ----
    if USE_CHECKLIST:
        checklist_cov = checklist_coverage(CHECKLIST_ITEMS, system_answer)
        if checklist_cov is not None:
            checklist_coverages.append(checklist_cov)

    # ---- 保存详细结果 ----
    if SAVE_RESULTS:
        detail = {
            "id": item.get("id", idx),
            "question": question,
            "reference_answer": reference,
            "system_answer": system_answer,
            "keyword_coverage_string": cov_str,
            "keyword_coverage_embedding": cov_emb,
            "rouge_l": rouge_l,
            "bert_f1": bert_f1,
            "response_time_sec": rt
        }
        if USE_JUDGE:
            detail["judge_score"] = judge_score
            detail["judge_reason"] = judge_reason
        if USE_CHECKLIST:
            detail["checklist_coverage"] = checklist_cov
        eval_details.append(detail)

# ===========================
# 十、统计 & 输出
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

if USE_JUDGE and judge_scores:
    m, s = mean_std(judge_scores)
    print(f"LLM-as-a-Judge Total Score: {m:.2f} ± {s:.2f}")

if USE_CHECKLIST and checklist_coverages:
    m, s = mean_std(checklist_coverages)
    print(f"Checklist Coverage: {m:.3f} ± {s:.3f}")

# ===========================
# 十一、保存结果（可选）
# ===========================
if SAVE_RESULTS:
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "keyword_string": statistics.mean(keyword_cov_string) if keyword_cov_string else None,
                "keyword_embedding": statistics.mean(keyword_cov_embed) if keyword_cov_embed else None,
                "rouge_l": statistics.mean(rouge_ls),
                "bert_f1": statistics.mean(bert_f1s),
                "response_time": statistics.mean(response_times),
                "judge_score": statistics.mean(judge_scores) if USE_JUDGE and judge_scores else None,
                "checklist_coverage": statistics.mean(checklist_coverages) if USE_CHECKLIST and checklist_coverages else None
            },
            "details": eval_details
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {SAVE_FILE}")