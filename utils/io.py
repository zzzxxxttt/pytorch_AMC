import os
import torch


def load_pretrain_v2(model, pretrain_dir):
  if os.path.isfile(pretrain_dir):
    pretrained_dict = torch.load(pretrain_dir)
    if 'state_dict' in pretrained_dict:
      pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    for k in pretrained_dict.keys():
      if k not in model_dict:
        print('%s not in model dict !' % k)
    for k in model_dict.keys():
      if k not in pretrained_dict:
        print('%s not in pretrained dict !' % k)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '%s'" % pretrain_dir)
  else:
    print("=> no checkpoint found at '%s' !" % pretrain_dir)


def save_model(net, optimizer, save_dir):
  torch.save({'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict(), }, save_dir)
  print('model saved in %s' % save_dir)


def print_mask(masks):
  print('[', end=' ')
  for m_ in masks:
    for m in m_:
      if m is None:
        print(m, end=', ')
      else:
        print('%d' % m.sum().item(), end=', ')
  print(']')
