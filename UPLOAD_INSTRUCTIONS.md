# 手动上传到 GitHub 指南

由于环境权限限制，无法直接执行 git 命令。请按照以下步骤手动上传项目：

## 步骤 1：下载完整项目

1. 进入项目目录：`/root/autodl-tmp/low-resource-fundus-qa-master/`
2. 选择所有文件和文件夹
3. 压缩成一个 ZIP 文件（例如：`low-resource-fundus-qa-master.zip`）

## 步骤 2：直接上传到 GitHub

### 方法 A：使用 GitHub 网页界面

1. 访问你的 GitHub 仓库：https://github.com/qqnnhhdmpc666/low-resource-fundus-qa

2. 点击右上角的 **Add file** → **Upload files**

3. 拖放 ZIP 文件到上传区域，或点击 **choose your files** 选择文件

4. 在 **Commit changes** 部分：
   - 填写 Commit message：
     ```
     feat: 重大更新 - Type-Aware路由 + 科研级评测体系
     
     新增功能：
     - Type-Aware智能路由机制（NLI分类器）
     - 科研级评测体系（MRR/NDCG/MAP + 统计检验）
     - QA对Chunk策略优化（5000条知识库）
     - 完整实验数据文档
     
     更新文档：
     - README.md完整重构
     - 技术实现细节说明书
     - 论文实验数据汇总
     - 测试集说明文档
     
     适配：
     - A4000 16GB低算力环境
     ```
   - 选择 **Commit directly to the master branch**
   - 点击 **Commit changes**

### 方法 B：使用 GitHub Desktop

1. 下载并安装 GitHub Desktop：https://desktop.github.com/

2. 克隆你的仓库：
   - File → Clone repository
   - URL: https://github.com/qqnnhhdmpc666/low-resource-fundus-qa
   - 选择本地存储位置

3. 解压 `low-resource-fundus-qa-master.zip` 到克隆的仓库目录

4. 打开 GitHub Desktop，你会看到所有更改

5. 填写 Commit message，然后点击 **Commit to master**

6. 点击 **Push origin** 推送到 GitHub

### 方法 C：使用 Git Bash（Windows）或 Terminal（Mac/Linux）

```bash
# 克隆仓库
git clone https://github.com/qqnnhhdmpc666/low-resource-fundus-qa
cd low-resource-fundus-qa

# 解压项目文件（替换为实际路径）
unzip /path/to/low-resource-fundus-qa-master.zip -d .

# 添加所有更改
git add .

# 提交
git commit -m "feat: 重大更新 - Type-Aware路由 + 科研级评测体系"

# 推送
git push origin master
```

## 步骤 3：验证上传

1. 访问：https://github.com/qqnnhhdmpc666/low-resource-fundus-qa
2. 点击 **README.md** 查看是否显示新内容
3. 检查文件列表是否包含新增的文件：
   - `question_classifier.py`
   - `evaluator_scientific.py`
   - `论文实验数据汇总.md`
   - `测试集说明文档.md`
   - `实验数据文件清单.md`

## 步骤 4：设置 GitHub 仓库

1. 点击 **Settings** → **General**
2. 在 **Description** 中填写：
   ```
   低资源眼底病智能问答系统（RAG + QLoRA + Type-Aware）
   ```
3. 在 **Topics** 中添加：
   ```
   rag qlora medical-qa fundus nlp pytorch
   ```
4. 保存更改

## 注意事项

- **不要上传大文件**：`fundus_lora/`、`hf_cache/`、`EYE_QA_PLUS_LOCAL/` 等目录
- **确保所有 .md 文档都已上传**
- **README.md 应该显示完整的项目信息**

## 成功标志

上传成功后，README 页面应该显示：
- 🌟 项目亮点（Type-Aware 智能路由等）
- 📊 实验结果速览（四种策略对比表）
- 🏗️ 系统架构（流程图和核心模块）
- 📁 项目结构（完整目录树）
- 🚀 快速开始（安装和运行指南）

---

如果遇到任何问题，请告诉我！
