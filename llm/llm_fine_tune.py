import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig, Trainer, TrainingArguments, \
    DataCollatorForSeq2Seq
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model

import config
import llm_fine_tune_data

model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=False
)

login(token=config.hugging_face_token)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=config.llm_cache, token=config.hugging_face_token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=quantization_config,
    cache_dir=config.llm_cache,
    token=config.hugging_face_token
)

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

datasets = Dataset.from_list(llm_fine_tune_data.prompt)
tokenized_id = datasets.map(process_func, remove_columns=datasets.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

lora_model = get_peft_model(model, config)

# 確保所有參數設置為可求導
for param in model.parameters():
    param.requires_grad = True

args = TrainingArguments(
    output_dir=config.fine_tune_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=lora_model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

model.save_pretrained(config.fine_tune_path)
tokenizer.save_pretrained(config.fine_tune_path)
