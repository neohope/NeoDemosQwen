#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer

'''
# 使用量化模型
# 安装AutoGPTQ，安装之前一定要核对好torch和CUDA的版本
# torch==2.1 auto-gptq>=0.5.1 transformers>=4.35.0 optimum>=1.14.0 peft>=0.6.1
# torch>=2.0,<2.1 auto-gptq<0.5.0 transformers<4.35.0 optimum<1.14.0 peft>=0.5.0,<0.6.0
pip install auto-gptq optimum
'''

# 调用量化模型
# 可选模型包括："Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)

# 调用量化模型
# 同时启用KV cache量化，不支持与flash attention同时开启
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
     device_map="auto",
     trust_remote_code=True,
     use_cache_quantization=True,
     use_cache_kernel=True,
     use_flash_attn=False
)
response, history = model.chat(tokenizer, "Hi", history=None)

'''
# 开启了KV cache量化之后，模型在推理的时候可以开启更大的batch size (bs)
# 开启了KV cache量化之后，模型在推理时可在生成更长的序列（sl，生成的token数）时，节约更多的显存
# 开启KV cache量化后，模型在推理时会将原始存进layer-past的float格式的key/value转换成int8格式，同时存储量化部分的参数

# quantize_cache_v和dequantize_cache_torch函数，需要在huggingface下载模型：Qwen/Qwen-7B-Chat-Int4
# 将key/value进行量化操作
qv,scale,zero_point=quantize_cache_v(v)

# 存入量化格式的layer-past
layer_past=((q_key,key_scale,key_zero_point), (q_value,value_scale,value_zero_point))

# 存入原始格式的layer-past
layer_past=(key,value)

# 将layer-past中存好的key，value直接取出使用，可以使用反量化操作将Int8格式的key/value转回float格式
v=dequantize_cache_torch(qv,scale,zero_point)

'''