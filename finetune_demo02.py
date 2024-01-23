#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM

'''
# 量化微调后的模型(Q-LoRA模型不需要量化，因为它本身就是量化过的)
# 使用auto_gptq去量化模型
pip install auto-gptq optimum

#1、准备校准集
#2、量化模型
python run_gptq.py \
    --model_name_or_path $YOUR_LORA_MODEL_PATH \
    --data_path $DATA \
    --out_path $OUTPUT_PATH \
    --bits 4 # 4 for int4; 8 for int8
'''

# 3、测试模型
tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/your/model",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
print(response)



