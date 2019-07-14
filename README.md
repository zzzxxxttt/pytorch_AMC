# Pytorch simple classification baselines

This repository is the pytorch implementation of [channel pruning](https://arxiv.org/abs/1707.06168) and [AMC](https://arxiv.org/pdf/1802.03494.pdf), the code is inspired by the tensorflow implementation.      

 
## Requirements:
- python>=3.5
- pytorch>=0.4.1
- sklearn
- tensorboardX(optional)

## Usage 

### Train a baseline network
* ```python3 cifar_train_eval.py```

### Searching pruning ratio for each layer using AMC 
* ```python3 cifar_search.py```

### Prune and finetune using searched pruning ratio 
* ```python3 cifar_finetune.py```

## Plain-20 50%FLOPs on CIFAR-10:

### Reward curve
<img src="https://github.com/zzzxxxttt/pytorch_AMC/blob/master/figs/plain20_agent_outputs.png" width="500" />

### Agent outputs
<img src="https://github.com/zzzxxxttt/pytorch_AMC/blob/master/figs/plain20_search.png" width="500" />

### Compression results
Model|Acc.|Acc. after pruning|Acc. after finetune|
:---:|:---:|:---:|:---:|
Plain-20|91.22%|80.01%|89.86%|
