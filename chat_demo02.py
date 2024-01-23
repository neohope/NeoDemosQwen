#!/usr/bin/env python3
# -*- coding utf-8 -*-

from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
使用Qwen-chat进行推理，使用本地模型
用代码直接从modelscope下载模型
pip install modelscope

# for windows
set PATH=%PYTHON_PATH%;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python chat_test02.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python chat_test02.py
'''

# 从HuggingFace下载模型，并放到qwen文件夹
# model_dir = snapshot_download('qwen/Qwen-7B')
# model_dir = snapshot_download('qwen/Qwen-7B-Chat')
# model_dir = snapshot_download('qwen/Qwen-14B')
model_dir = snapshot_download('qwen/Qwen-14B-Chat')

# 从本地加载模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()

# 现在模型就可以使用了
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=history) 
print(response)
response, history = model.chat(tokenizer, "它有什么好玩的景点", history=history)
print(response)
