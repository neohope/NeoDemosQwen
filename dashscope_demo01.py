#!/usr/bin/env python3
# -*- coding utf-8 -*-

import random
from http import HTTPStatus
from dashscope import Generation

'''
# 使用阿里云灵积服务dashscope进行推理，与openai的接口很像
# 配置DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
# 安装dashscope库
pip install dashscope

# for windows
set PATH=%PYTHON_PATH%;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python dashscope_demo01.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python dashscope_demo01.py
'''

def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做西红柿鸡蛋？'}]
    gen = Generation()
    response = gen.call(
        Generation.Models.qwen_turbo,
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message" format.
    )
    return response

if __name__ == '__main__':
    response = call_with_messages()
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
