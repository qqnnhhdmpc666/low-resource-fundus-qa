import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ======================
# 1. 基础配置
# ======================
model_name = "Qwen/Qwen2.5-7B-Instruct"
data_path = "fundus_finetune.jsonl"
output_dir = "./fundus_sft_out"

max_seq_length = 1024

# ======================
# 2. Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================
# 3. Model
# ======================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ======================
# 4. Dataset
# ======================
dataset = load_dataset(
    "json",
    data_files=data_path,
    split="train"
)

# ======================
# 5. formatting_func（完全匹配你的数据）
# ======================
def formatting_func(example):
    return (
        "### Question:\n"
        f"{example['question']}\n\n"
        "### Answer:\n"
        f"{example['answer']}"
    )

# ======================
# 6. SFTConfig（⚠️ 关键：API 正确写法）
# ======================
sft_config = SFTConfig(
    max_seq_length=max_seq_length,
    packing=False
)

# ======================
# 7. TrainingArguments
# ======================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

# ======================
# 8. Trainer
# ======================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args,
    sft_config=sft_config,
)

# ======================
# 9. Train
# ======================
trainer.train()

# ======================
# 10. Save
# ======================
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

