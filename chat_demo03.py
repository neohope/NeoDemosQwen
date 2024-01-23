#!/usr/bin/env python3
# -*- coding utf-8 -*-

from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

'''
使用Qwen-chat进行推理，使用modelscope
pip install modelscope

# for windows
set PATH=%PYTHON_PATH%;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python chat_test03.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python chat_test03.py
'''

# 直接使用modelscope加载模型
# 可选的模型包括: "qwen/Qwen-7B-Chat", "qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) 

# 现在模型就可以使用了
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=history) 
print(response)
response, history = model.chat(tokenizer, "它有什么好玩的景点", history=history)
print(response)
