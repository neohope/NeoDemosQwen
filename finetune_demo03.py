#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

'''
# 使用finetune.py脚本进行微调，video

# 安装所需组件
pip install peft deepspeed

# 准备微调数据
finetune_demo03.json

# 全参数训练，经测试，在微调阶段不更新ViT的参数会取得更好的表现
# 只支持分布式训练
sh finetune/finetune_ds.sh

# LORA训练
# 只更新adapter层的参数而无需更新原有语言模型的参数。这种方法允许用户用更低的显存开销来训练模型，也意味着更小的计算开销
# 单卡训练
sh finetune/finetune_qlora_single_gpu.sh
# 分布式训练
sh finetune/finetune_qlora_ds.sh

# Q-LoRA训练，必须使用4比特量化模型，同时采用paged attention等技术实现更小的显存开销
# 如果依然遇到显存不足的问题，可以考虑使用Q-LoRA 。该方法使用4比特量化模型以及paged attention等技术实现更小的显存开销
# 单卡：pip install mpi4py
# 单卡训练
bash finetune/finetune_qlora_single_gpu.sh
# 分布式训练
bash finetune/finetune_qlora_ds.sh
'''

path_to_adapter = "/path/to/output/directory"
merged_model_directory = "/path/to/merged/model"

# 微调后，可以读取训练后的模型
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

# 微调后，也可以合并，然后存储模型
# 合并并存储模型（LoRA支持合并，Q-LoRA不支持），再用常规方式读取你的新模型
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
merged_model = model.merge_and_unload()

# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(merged_model_directory, max_shard_size="2048MB", safe_serialization=True)

# 保存tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
tokenizer.save_pretrained(merged_model_directory)
