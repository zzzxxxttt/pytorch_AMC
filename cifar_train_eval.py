import os
import time
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = True

from nets.cifar_vgg import vgg16
from nets.cifar_plain import plain20
from nets.cifar_resnet import resnet56

from utils.io import save_model
from utils.dataset import CIFAR10_split
from utils.summary import SummaryWriter
from utils.preprocessing import cifar_transform

# Training settings
parser = argparse.ArgumentParser(description='CIFAR_train_eval')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--log_name', type=str, default='plain20_baseline')

parser.add_argument('--model', type=str, default='plain20')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=200)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name + '.t7')

os.makedirs(cfg.log_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus


def main():
  print('==> Preparing data ...')
  train_dataset = CIFAR10_split(root=cfg.data_dir, split='train', split_size=45000,
                                transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.train_batch_size,
                                             shuffle=True,
                                             num_workers=cfg.num_workers)

  val_dataset = CIFAR10_split(root=cfg.data_dir, split='val', split_size=5000,
                              transform=cifar_transform(is_training=False))
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.eval_batch_size,
                                           shuffle=False,
                                           num_workers=cfg.num_workers)

  test_dataset = CIFAR10_split(root=cfg.data_dir, split='test',
                               transform=cifar_transform(is_training=False))
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.eval_batch_size,
                                            shuffle=False,
                                            num_workers=cfg.num_workers)

  print('==> Building Model ...')
  if cfg.model == 'plain20':
    network = plain20
  elif cfg.model == 'vgg16':
    network = vgg16
  elif cfg.model == 'resnet56':
    network = resnet56
  else:
    raise NotImplementedError

  model = network().cuda()

  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedulr = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
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

  def val(epoch):
    with torch.no_grad():
      model.eval()
      correct = 0
      for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

      acc = 100. * correct / len(val_dataset)
      print('-' * 30 + 'validation set acc = %.2f%%' % acc)
      summary_writer.add_scalar('val_acc', acc, global_step=epoch)

  def test(epoch):
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
    val(epoch)
    test(epoch)
    lr_schedulr.step(epoch)
    save_model(model, optimizer, cfg.ckpt_dir)

  summary_writer.close()


if __name__ == '__main__':
  main()
