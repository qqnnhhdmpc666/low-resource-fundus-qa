# 测试集说明文档

---

## 1. 测试集构成

### 1.1 测试集概述

| 项目 | 说明 |
|-----|------|
| 总题数 | 50题 |
| 语言分布 | 中文20题 + 英文30题 |
| 问题类型 | 4大类 |
| 来源 | EYE-QA-PLUS数据集 + 人工扩展 |

### 1.2 问题类型分布

| 问题类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| daily_advice | 15 | 30% | 日常建议类问题 |
| disease_definition | 12 | 24% | 疾病定义类问题 |
| decision | 13 | 26% | 决策建议类问题 |
| emergency | 10 | 20% | 紧急情况类问题 |

---

## 2. 测试集内容

### 2.1 中文测试集（20题，test_set.json）

#### daily_advice类（5题）
1. 高度近视日常要注意什么？
2. 孩子近视加深快该怎么办？
3. 视疲劳怎么缓解？
4. 高度近视需要多久检查一次眼底？
5. 干眼症怎么治疗？

#### disease_definition类（5题）
1. 高度近视会引起视网膜脱离吗？
2. 糖尿病会引起眼底出血吗？
3. 眼底照片里有黄色斑点是什么意思？
4. 视盘边界不清是什么问题？
5. 近视眼底会出现什么变化？

#### decision类（5题）
1. 眼底激光后要注意什么？
2. 白内障手术后视力又模糊了怎么办？
3. 玻璃体出血怎么办？
4. 高度近视可以做激光近视手术吗？
5. 糖尿病眼底病变怎么预防？

#### emergency类（5题）
1. 飞蚊症突然多了怎么办？
2. 孕妇高度近视能顺产吗？
3. （其他紧急问题）

### 2.2 英文测试集（30题，text_set_hard.json）

#### daily_advice类（10题）
1. What should I do if my eyes feel dry after working on a computer for a long time?
2. How often should I get an eye check-up?
3. How can I reduce eye strain while reading?
4. Can diet affect my eye health?
5. Is it normal for eyes to feel tired after long phone use?
6. How can I tell if I need reading glasses?
7. Are sunglasses really necessary?
8. Can eye exercises improve vision?
9. How should I care for my eyes during allergy season?
10. Can sleeping well help my eye health?

#### disease_definition类（7题）
1. What are the signs of cataracts?
2. What is glaucoma, and how is it detected?
3. Can stress affect my eyesight?
4. How do I know if my child needs an eye exam?
5. Is it safe to wear makeup near the eyes?
6. Can wearing glasses prevent my eyesight from getting worse?
7. Can wearing contact lenses cause infections?

#### decision类（8题）
1. What should I do if I accidentally get dust in my eye?
2. What should I do if I see flashes of light in my vision?
3. What can I do to prevent myopia from worsening?

#### emergency类（5题）
1. （紧急情况相关问题）

---

## 3. 评测指标说明

### 3.1 文本生成指标

| 指标 | 说明 | 范围 |
|-----|------|------|
| ROUGE-L | 最长公共子序列重叠度 | [0, 1] |
| BERT-F1 | 基于BERT的语义相似度 | [0, 1] |

### 3.2 LLM-as-a-Judge指标

| 指标 | 说明 | 范围 |
|-----|------|------|
| Correctness | 医学准确性 | [1, 5] |
| Completeness | 完整性 | [1, 5] |
| Safety | 安全性 | [1, 5] |
| Helpfulness | 有用性 | [1, 5] |

### 3.3 覆盖率指标

| 指标 | 说明 | 范围 |
|-----|------|------|
| Keyword Coverage | 关键词覆盖率 | [0, 1] |
| Checklist Coverage | 检查清单覆盖率 | [0, 1] |

### 3.4 效率指标

| 指标 | 说明 | 单位 |
|-----|------|------|
| Avg Response Time | 平均响应时间 | 秒 |

---

## 4. 参考答案示例

### 4.1 中文参考答案（test_set.json）

| 题号 | 问题 | 参考答案关键词 |
|-----|------|--------------|
| 1 | 高度近视会引起视网膜脱离吗？ | 高度近视、眼轴延长、视网膜变薄、格子样变性、裂孔、闪光感、飞蚊症 |
| 2 | 糖尿病会引起眼底出血吗？ | 糖尿病视网膜病变、新生血管、眼底出血、玻璃体积血、控制血糖 |
| 3 | 干眼症怎么治疗？ | 热敷、人工泪液、眨眼、加湿器、Omega-3、环孢素 |
| 4 | 飞蚊症突然多了怎么办？ | 闪光感、玻璃体后脱离、裂孔、散瞳查眼底 |
| 5 | 高度近视日常要注意什么？ | 剧烈运动、揉眼睛、闪光、黑影、查眼底 |

### 4.2 英文参考答案（text_set_hard.json）

| 题号 | 问题 | 参考答案关键词 |
|-----|------|--------------|
| 1 | What should I do if my eyes feel dry...? | breaks, blink, artificial tears, 20-20-20 rule |
| 2 | How often should I get an eye check-up? | 1-2 years, diabetes, high myopia, regular exams |
| 3 | What are the signs of cataracts? | cloudy vision, light sensitivity, faded colors, double vision |

---

## 5. 数据文件说明

| 文件名 | 说明 | 格式 |
|-------|------|------|
| test_set.json | 中文测试集（20题） | JSON |
| text_set_hard.json | 英文测试集（30题） | JSON |
| eval_*.json | 评测结果文件 | JSON |
| llm_scores_*.json | LLM评分文件 | JSON |
| comprehensive_scores_summary.json | 综合评分汇总 | JSON |

---

## 6. 测试集使用说明

### 6.1 运行评测

```bash
# 评测中文测试集
python evaluate.py --test_file test_set.json --output results/eval_chinese.json

# 评测英文测试集
python evaluate.py --test_file text_set_hard.json --output results/eval_english.json
```

### 6.2 LLM评测

```bash
# 对评测结果进行LLM评分
python llm_judge_new.py --input eval_hybrid_rerank_rewrite_True.json --output llm_scores.json
```

---

## 7. 测试集统计信息

### 7.1 问题长度统计

| 统计量 | 中文问题 | 英文问题 |
|-------|---------|---------|
| 平均长度 | 15字 | 15词 |
| 最短 | 8字 | 6词 |
| 最长 | 30字 | 35词 |

### 7.2 答案长度统计

| 统计量 | 参考答案 | 系统答案 |
|-------|---------|---------|
| 平均长度 | 200字 | 300字 |
| 最短 | 100字 | 150字 |
| 最长 | 500字 | 600字 |

---

**测试集构建时间**：2024年
**最后更新**：与项目代码同步更新
