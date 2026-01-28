#!/usr/bin/env python3
# ============================================================
# 自动运行所有实验组合脚本 (Hard 测试集版本)
# ============================================================
# 运行所有检索模式和查询改写的组合
# 共3×2=6个实验，串行运行
# 使用 text_set_hard.json 作为测试文件
# ============================================================

import subprocess
import time

# 定义实验参数组合
retrieval_modes = ["vector", "hybrid", "hybrid_rerank"]
query_rewrite_options = [True, False]
fast_mode_n = 20  # Hard 测试集有20个样本

test_file = "text_set_hard.json"

# 虚拟环境设置
venv_path = "/root/autodl-tmp/fundus_env"
venv_activate = f"{venv_path}/bin/activate"

print("开始运行所有实验组合 (Hard 测试集)...")
print(f"每个实验将使用 {fast_mode_n} 个测试样本")
print(f"测试文件: {test_file}")
print(f"使用环境: {venv_path}")
print(f"环境激活脚本: {venv_activate}")
print("运行模式: 串行运行（一个接一个）")
print("=" * 80)

# 运行剩余的实验组合
# 已完成的实验：
# 1. vector, True
# 2. vector, False
# 3. hybrid, True
# 4. hybrid, False
# 5. hybrid_rerank, True
# 剩余实验：
# 6. hybrid_rerank, False

all_experiments = [
    ("hybrid_rerank", False)
]

for mode, rewrite in all_experiments:
    # 构建命令，先激活虚拟环境，然后运行 evaluate.py
    cmd = f"source {venv_activate} && python evaluate.py "
    cmd += f"--retrieval_mode {mode} "
    cmd += f"--use_query_rewrite {str(rewrite).lower()} "
    cmd += f"--test_file {test_file} "
    cmd += f"--fast_mode_n {fast_mode_n}"
    
    print(f"\n运行实验: 检索模式={mode}, 查询改写={rewrite}")
    print(f"命令: {cmd}")
    print("-" * 80)
    
    # 执行命令，使用 bash shell，直接输出到终端
    start_time = time.time()
    # 移除 capture_output=True，让输出直接显示在终端
    result = subprocess.run(cmd, shell=True, executable="/bin/bash")
    end_time = time.time()
    
    # 输出结果
    print(f"实验完成，用时: {end_time - start_time:.2f}秒")
    print(f"返回码: {result.returncode}")
    
    # 检查是否有错误
    if result.returncode != 0:
        print("实验失败，请查看上面的错误信息")
    else:
        print("实验成功完成！")
    
    print("-" * 80)

print("\n所有实验组合运行完成！")
print("请查看生成的 eval_*.json 文件获取实验结果。")
