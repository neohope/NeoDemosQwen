#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

'''
使用Qwen-Audio-Chat进行语音处理

# for windows
set PATH=%PYTHON_PATH%;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python audio_demo01.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python audio_demo01.py
'''

# 设置随机种子
torch.manual_seed(1234)

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# 音频转文字
query = tokenizer.from_list_format([
    {'audio': 'audio/1272-128104-0000.flac'}, # Either a local path or an url
    {'text': 'what does the person say?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
'''
# The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".
'''

# 查找某个关键词在音频中出现的时间范围
response, history = model.chat(tokenizer, 'Find the start time and end time of the word "middle classes"', history=history)
print(response)
'''
# The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
'''
