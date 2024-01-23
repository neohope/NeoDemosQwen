#!/usr/bin/env python3
# -*- coding utf-8 -*-

from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
下载模型

# for windows
set PATH=%PYTHON_PATH%;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python audio_demo03.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python audio_demo03.py
'''

# 下载模型
model_id = 'qwen/Qwen-Audio-Chat'
revision = 'master'
model_dir = snapshot_download(model_id, revision=revision)

# 加载本地模型
# trust_remote_code is still set as True since we still load codes from local dir instead of transformers
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    trust_remote_code=True
).eval()
