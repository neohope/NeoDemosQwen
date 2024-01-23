#!/usr/bin/env python3
# -*- coding utf-8 -*-

from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

'''
从modelscope下载模型

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

# 从modelscope下载模型
model_id = 'qwen/Qwen-Audio-Chat'
revision = 'master'
model_dir = snapshot_download(model_id, revision=revision)

# 设置随机种子
torch.manual_seed(1234)

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir

# 加载模型
# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# 音频转文字
query = tokenizer.from_list_format([
    {'audio': 'audio/1272-128104-0000.flac'}, # Either a local path or an url
    {'text': 'what does the person say?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
'''
The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".
'''

# 查找某个关键词在音频中出现的时间范围
response, history = model.chat(tokenizer, 'Find the start time and end time of the word "middle classes"', history=history)
print(response)
'''
The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
'''

'''
# 官方自带Web UI Demo
pip install -r requirements_web_demo.txt
python web_demo_audio.py
'''