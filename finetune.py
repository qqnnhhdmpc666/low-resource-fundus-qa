import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 加载你的微调数据
dataset = load_dataset("json", data_files="fundus_finetune.jsonl", split="train")

# 2. 4-bit量化配置（低显存关键）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 3. 加载模型和tokenizer（第一次会下载7B模型，约7-8GB硬盘）
model_name = "Qwen/Qwen2.5-7B-Instruct"  # 中文强，指令跟随好
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 修Qwen常见坑，避免loss异常

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配GPU
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA配置（只训少量参数）
lora_config = LoraConfig(
    r=8,  # 秩，低显存
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen注意力层，更稳
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. 训练参数（适配低显存）
training_args = TrainingArguments(
    output_dir="./fundus_model",
    per_device_train_batch_size=2,  # 保守设置，避免显存溢出
    gradient_accumulation_steps=4,  # 模拟大batch
    learning_rate=2e-4,
    num_train_epochs=3,  # 3轮足够
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # 混合精度，省显存
    report_to="none"
)

# 6. 数据格式化（指令格式）
def format_func(example):
    return f"### Question: {example['question']}\n### Answer: {example['answer']}"

# 7. 启动训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    formatting_func=format_func,
    max_seq_length=512
)

print("开始微调，请耐心等待...")
trainer.train()

# 8. 保存LoRA权重（只有几十MB）
trainer.save_model("./fundus_lora")
print("✅ 微调完成！LoRA权重保存到 ./fundus_lora")
