What is this project about
=========
This is just a project of Qwen demos. 


How to build
============
1. install python 3.10+


2. update pip
```shell
    python -m pip install --upgrade pip
```

3. install packages
```shell
    # 必须：安装requirements
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

    # 可选：安装flash-attention提升计算效率（显卡要支持）
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention && pip install .
    # 下方安装可选，安装可能比较缓慢
    # pip install csrc/layer_norm
    # 如果flash-attn版本高于2.1.1，下方无需安装
    # pip install csrc/rotary
```

4. 要匹配cuda版本及pytorch版本
```shell
# 查询cuda版本及pytorch版本
https://pytorch.org/get-started/locally/
https://developer.nvidia.com/cuda-toolkit-archive

# 下载所需cuda，比如118
# 
# 安装对应版本的pytorch
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 测试一下
import torch
print(torch.cuda.is_available())
```

5. run
```shell
    python -m venv venv
    python chat_demo01.py
```

Reference
=========
