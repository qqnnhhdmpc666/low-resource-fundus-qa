# GitHub 上传指南

## 上传到 GitHub 的步骤

### 1. 在本地终端执行以下命令

```bash
# 进入项目目录
cd /path/to/low-resource-fundus-qa-master

# 初始化 git（如果还没有）
git init

# 添加所有文件到暂存区
git add .

# 提交更改
git commit -m "feat: 更新项目到最新版本

- 新增 Type-Aware 智能路由机制（NLI分类器）
- 新增科研级评测体系（MRR/NDCG/MAP + 统计检验）
- 优化 QA对 Chunk 策略（5000条知识库）
- 更新 README.md 完整文档
- 新增论文实验数据汇总
- 适配 A4000 16GB 低算力环境"

# 添加远程仓库（替换为你的GitHub仓库地址）
git remote add origin https://github.com/qqnnhhdmpc666/low-resource-fundus-qa.git

# 推送到 GitHub
git push -u origin main
# 或如果是 master 分支
git push -u origin master
```

### 2. 如果已有远程仓库，直接推送

```bash
# 添加所有更改
git add .

# 提交
git commit -m "feat: 更新项目到最新版本

- 新增 Type-Aware 智能路由机制
- 新增科研级评测体系
- 优化知识库构建策略
- 更新完整文档"

# 推送
git push origin main
```

### 3. 处理大文件（如果模型文件太大）

```bash
# 安装 git-lfs
sudo apt-get install git-lfs
git lfs install

# 追踪大文件
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.pkl"

# 添加 .gitattributes
git add .gitattributes
git commit -m "chore: 配置 git-lfs"
```

### 4. 排除不需要上传的文件

确保 `.gitignore` 文件包含：

```
# 模型文件（太大，建议通过其他方式分享）
hf_cache/
*.safetensors
*.bin
fundus_lora/*.safetensors

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
fundus_env/

# 数据（可选，如果太大）
EYE_QA_PLUS_LOCAL/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/

# 日志
*.log
```

### 5. 验证上传

```bash
# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 查看分支
git branch -a
```

### 6. GitHub 页面设置建议

上传后，建议在 GitHub 页面设置：

1. **About 部分**：
   - Description: 低资源眼底病智能问答系统（RAG + QLoRA + Type-Aware）
   - Topics: `rag`, `qlora`, `medical-qa`, `fundus`, `nlp`, `pytorch`
   - Website: （如果有演示页面）

2. **README 展示**：
   - 确保 README.md 正确渲染
   - 检查 badges 显示正常

3. **Releases**（可选）：
   - 创建 Release 版本
   - 上传预训练模型到 Release Assets

### 7. 分享预训练模型（可选）

如果模型文件太大无法上传到 GitHub：

**方案1：Hugging Face Hub**
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/fundus-lora ./fundus_lora
```

**方案2：百度网盘/阿里云盘**
- 打包 `fundus_lora/` 目录
- 上传到网盘并分享链接
- 在 README 中添加下载链接

**方案3：GitHub Releases**
- 在 Release 页面上传模型文件（单个文件最大 2GB）

### 8. 常见问题

**Q: 推送被拒绝？**
```bash
# 先拉取远程更改
git pull origin main --rebase
# 然后再推送
git push origin main
```

**Q: 提交信息写错了？**
```bash
# 修改最后一次提交
git commit --amend -m "新的提交信息"
# 强制推送（谨慎使用）
git push origin main --force
```

**Q: 忘记添加文件？**
```bash
# 添加遗漏的文件
git add 遗漏的文件
git commit --amend --no-edit
git push origin main --force-with-lease
```

---

## 上传后的检查清单

- [ ] 代码文件已上传
- [ ] README.md 正确显示
- [ ] 文档文件已上传（技术实现细节说明书.md、论文实验数据汇总.md等）
- [ ] .gitignore 配置正确
- [ ] 大文件已处理（git-lfs 或其他方式）
- [ ] GitHub About 信息已完善
- [ ] Topics 标签已添加

---

**提示**：如果当前环境无法执行 git 命令，请在本地终端执行上述命令。
