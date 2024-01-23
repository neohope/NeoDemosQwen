#!/usr/bin/env python3
# -*- coding utf-8 -*-

# 从https://github.com/QwenLM/Qwen下载vllm_wrapper.py
from vllm_wrapper import vLLMWrapper

'''
# 使用vLLM进行部署并加速推理

# 该方法安装较快，但官方版本不支持量化模型
pip install vllm  

# 该面方法支持int4量化 (int8量化模型支持将近期更新)，但安装更慢 (约~10分钟)。
git clone https://github.com/QwenLM/vllm-gptq
cd vllm-gptq
pip install -e .
'''


# 用vLLM进行接口封装，并进行交互
model = vLLMWrapper('Qwen/Qwen-7B-Chat', tensor_parallel_size=1)
response, history = model.chat(query="你好", history=None)
print(response)
response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(query="给这个故事起一个标题", history=history)
print(response)

'''
# 网页Demo
# 你可以使用FastChat去搭建一个网页Demo或类OpenAI API服务器。首先，请安装FastChat：
pip install "fschat[model_worker,webui]"

# 使用vLLM和FastChat运行Qwen之前，首先启动一个controller：
python -m fastchat.serve.controller

# 然后启动model worker读取模型。如使用单卡推理，运行如下命令：
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --dtype bfloat16
# 运行int4模型
# python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --dtype float16 

# 然而，如果你希望使用多GPU加速推理或者增大显存，你可以使用vLLM支持的模型并行机制。假设你需要在4张GPU上运行你的模型，命令如下所示：
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4 --dtype bfloat16
# 运行int4模型
# python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4 --dtype float16

# 启动model worker后，你可以启动一个：Web UI Demo
python -m fastchat.serve.gradio_web_server

# OpenAI API
使用OpenAI API前，请阅读我们的API章节配置好环境，然后运行如下命令：
python -m fastchat.serve.openai_api_server --host localhost --port 8000
'''

'''
# 官方自带Web UI Demo
pip install -r requirements_web_demo.txt
python web_demo.py

# 官方自带交互式Demo
python cli_demo.py

# 官方自带OpenAI API Demo
pip install fastapi uvicorn "openai<1.0" pydantic sse_starlette
python openai_api.py
'''
