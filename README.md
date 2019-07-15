# Pytorch AMC

This repository is the pytorch implementation of [*Channel Pruning for Accelerating Very Deep Neural Networks*](https://arxiv.org/abs/1707.06168) and [*AMC: AutoML for Model Compression and Acceleration on Mobile Devices*](https://arxiv.org/abs/1802.03494), the code is inspired by the [tensorflow implementation](https://pocketflow.github.io/).      

 
## Requirements:
- python>=3.5
- pytorch>=0.4.1
- sklearn
- tensorboardX(optional)

## Usage 

### Train a baseline network
```
python3 cifar_train_eval.py \\
        --model plain20
        --log_name plain20_baseline \\
```

### Searching pruning ratio using AMC 
```
python3 cifar_search.py \\
        --model plain20 \\
        --pretrain_name plain20_baseline\\
        --log_name search_plain20_flops0.5 \\
        --method channel_pruning \\
        --lim_type flops \\
        --lim_ratio 0.5 \\
        --max_steps 500
```

### Prune and finetune 
```
python3 cifar_finetune.py \\
        --search_name search_plain20_flops0.5 \\
        --pretrain_name plain20_baseline \\
        --model plain20 \\
        --method channel_pruning \\
        --lim_type flops \\
        --lim_ratio 0.5 \\
        --lr 0.01 \\
        --max_epochs 100
```

## Plain-20 50%FLOPs on CIFAR-10:

### Reward curve
<img src="https://github.com/zzzxxxttt/pytorch_AMC/blob/master/figs/plain20_agent_outputs.png" width="500" />

### Agent outputs
<img src="https://github.com/zzzxxxttt/pytorch_AMC/blob/master/figs/plain20_search.png" width="500" />

### Compression results
Model|Acc.|Acc. after pruning|Acc. after finetune|
:---:|:---:|:---:|:---:|
Plain-20|91.22%|80.01%|89.86%|
