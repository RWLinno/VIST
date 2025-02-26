
# RAST
This code is a PyTorch implementation of our paper "RAST: A Retrieval-Augmented Transformer for Time Series Forecasting".

## Requirements
We implement the experiments on a Linux Server with CUDA 12.2 equipped with 4x A6000 GPUs. For convenience, execute the following command.
```
# Install Python
conda create -n RAST python==3.11
conda activate RAST

# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install other dependencies
pip install -r requirements.txt
```

### Localizing the RAG knowledge base
https://zhuanlan.zhihu.com/p/20818407889
使用Ollama本地部署大语言模型，模型选择看个人喜好
```
ollama run deepseek-r1:7b # you can 
ollama pull nomic-embed-text #

```

## Dataset
You can download the dataset from [LargeST](https://github.com/liuxu77/LargeST) or [BasicTS](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/getting_started.md). Unzip the files to the datasets/ directory:

## Quick Start
```bash
python experiments/train.py -c examples/regular_config.py -g 0

python experiments/train.py -c RAST/train_SD.py -g 0
```