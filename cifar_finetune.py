import os
import time
import pickle
import argparse

# 3~5 is enough, otherwise sklearn will try to use all the CPUs
os.environ["OPENBLAS_NUM_THREADS"] = '3'
os.environ["MKL_NUM_THREADS"] = '3'

import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = True

from nets.cifar_vgg import vgg16
from nets.cifar_plain import plain20
from nets.cifar_resnet import resnet56

from nets.cifar_vgg_proxy import vgg16_proxy
from nets.cifar_plain_proxy import plain20_proxy
from nets.cifar_resnet_proxy import resnet56_proxy

from learners.channel_pruning import channelPruner

from utils.io import save_model
from utils.dataset import CIFAR10_split
from utils.summary import SummaryWriter
from utils.preprocessing import cifar_transform

# Training settings
parser = argparse.ArgumentParser(description='CIFAR_finetune')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--search_name', type=str, default='RL_plain20_flops0.5')
parser.add_argument('--pretrain_name', type=str, default='plain20_baseline')

parser.add_argument('--model', type=str, default='plain20')

parser.add_argument('--method', type=str, default='channel_pruning')
parser.add_argument('--lim_type', type=str, default='flops')
parser.add_argument('--lim_ratio', type=float, default=0.5)
parser.add_argument('--min_action', type=float, default=0.2)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.search_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name + '.t7')
cfg.search_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.search_name + '.pickle')
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.search_name + '_finetune.t7')

os.makedirs(cfg.log_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus


def main():
  print('==> Preparing data ...')
  train_dataset = CIFAR10_split(root=cfg.data_dir, split='train', split_size=50000,
                                transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.train_batch_size,
                                             shuffle=True,
                                             num_workers=cfg.num_workers)

  test_dataset = CIFAR10_split(root=cfg.data_dir, split='test',
                               transform=cifar_transform(is_training=False))
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.eval_batch_size,
                                            shuffle=False,
                                            num_workers=cfg.num_workers)

  print('==> Building Model ...')
  if cfg.model == 'plain20':
    network_fn, proxy_fn = plain20, plain20_proxy
  elif cfg.model == 'vgg16':
    network_fn, proxy_fn = vgg16, vgg16_proxy
  elif cfg.model == 'resnet56':
    network_fn, proxy_fn = resnet56, resnet56_proxy
  else:
    raise NotImplementedError

  with open(cfg.search_dir, 'rb') as handle:
    ckpt = pickle.load(handle)
  actions = ckpt['real_actions'][np.argmax(ckpt['rewards'])]

  learner = channelPruner(method=cfg.method,
                          model=network_fn(),
                          proxy=proxy_fn(lim_type=cfg.lim_type,
                                         ratio=cfg.lim_ratio,
                                         min_action=cfg.min_action,
                                         lower_bound=True),
                          pretrain_dir=cfg.pretrain_dir,
                          num_sample_points=10,
                          data_dir=cfg.data_dir,
                          debug=False)

  _, _, done, _ = learner.init_episode()
  for action in actions:
    _, _, done, score = learner.compress(action)
    print('layer: %s ratio: %.2f score: %.5f' % (learner.proxy.layer_now, action, score[0]))
  print('acc after pruning: ', learner.evaluate())

  model = learner.model.cuda()
  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedulr = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
  criterion = torch.nn.CrossEntropyLoss().cuda()

  summary_writer = SummaryWriter(cfg.log_dir)

  # Training
  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      inputs, targets = inputs.cuda(), targets.cuda()

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        duration = time.time() - start_time
        print('epoch: %d step: %d loss: %.5f (%d imgs/sec, %.2f sec/batch)' %
              (epoch, batch_idx, loss.item(),
               cfg.log_interval * cfg.train_batch_size / duration,
               duration / cfg.log_interval))
        start_time = time.time()

        step = epoch * len(train_loader) + batch_idx
        summary_writer.add_scalar('loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch):
    # pass
    with torch.no_grad():
      model.eval()
      correct = 0
      for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

      acc = 100. * correct / len(test_dataset)
      print('-' * 30 + 'test set acc = %.2f%%' % acc)
      summary_writer.add_scalar('test_acc', acc, global_step=epoch)

  for epoch in range(cfg.max_epochs):
    train(epoch)
    test(epoch)
    lr_schedulr.step(epoch)
    save_model(model, optimizer, cfg.ckpt_dir)

  summary_writer.close()


if __name__ == '__main__':
  main()
