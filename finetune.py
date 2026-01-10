import os
os.environ["DISABLE_BF16"] = "1"  # CUDA 11.8 必须

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer


# =========================
# 1. 基本配置
# =========================
model_name = "Qwen/Qwen2.5-7B-Instruct"
data_path = "fundus_finetune.jsonl"
output_dir = "./fundus_lora"

MAX_LEN = 1024


# =========================
# 2. 加载数据
# =========================
dataset = load_dataset(
    "json",
    data_files=data_path,
    split="train"
)


# =========================
# 3. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = MAX_LEN  # ⚠️ 老 TRL 只能这样控长度


# =========================
# 4. QLoRA 4-bit 量化配置
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# =========================
# 5. 加载模型（4bit）
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)


# =========================
# 6. LoRA 配置（QLoRA 核心）
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# 7. 数据格式化函数（⚠️ 必须返回 list）
# =========================
def formatting_func(example):
    text = (
        "### Question:\n"
        f"{example['question']}\n\n"
        "### Answer:\n"
        f"{example['answer']}"
    )
    return [text]   # ⚠️ TRL 0.9.x 强制要求 list


# =========================
# 8. TrainingArguments
# =========================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,      # 12G 显存安全值
    gradient_accumulation_steps=8,      # 等效 batch = 16
    learning_rate=2e-4,                 # QLoRA 推荐
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",            # 省显存关键
    report_to="none",
    dataloader_num_workers=4,
)


# =========================
# 9. SFTTrainer（老版本稳定写法）
# =========================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,          # ⚠️ 0.9.6 仍然需要
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)


# =========================
# 10. 开始训练
# =========================
print("🚀 开始 QLoRA 微调 ...")
trainer.train()


# =========================
# 11. 保存 LoRA 权重
# =========================
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ 微调完成，LoRA 权重已保存至 {output_dir}")
