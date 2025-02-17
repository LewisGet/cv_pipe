from transformers import AutoTokenizer, AutoModelForCausalLM

import config

# 載入 tokenizer 和模型結構
tokenizer = AutoTokenizer.from_pretrained(config.fine_tune_path)
model = AutoModelForCausalLM.from_pretrained(config.fine_tune_path)
model.to("cuda")

model.eval()
