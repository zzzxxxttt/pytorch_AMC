import pickle
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class PreActBlock(nn.Module):
  def __init__(self, in_planes, out_planes, stride=1):
    super(PreActBlock, self).__init__()
    self.bn0 = nn.BatchNorm2d(in_planes)
    self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.register_buffer('mask0', torch.ones(1, out_planes, 1, 1))
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.register_buffer('mask1', torch.ones(1, out_planes, 1, 1))

    self.skip_conv = None
    if stride != 1:
      self.skip_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      self.skip_bn = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    out = F.relu(self.bn0(x))

    shortcut_out = None
    if self.skip_conv is not None:
      shortcut = self.skip_conv(out)
      shortcut_out = shortcut
      shortcut = self.skip_bn(shortcut)
    else:
      shortcut = x

    conv0_input = out
    out = self.conv0(out)
    conv0_out = out
    out = F.relu(self.bn1(out))
    out = out * self.mask0
    conv1_input = out
    out = self.conv1(out)
    out = out * self.mask1
    out += shortcut
    conv1_out = out
    return out, (conv0_input, conv1_input), (conv0_out, conv1_out, shortcut_out), shortcut


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, num_classes):
    super(PreActResNet, self).__init__()

    self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    self.name_to_ind = OrderedDict()
    self.ind_to_name = OrderedDict()
    self.meta_data = OrderedDict()
    self.name_to_next_name = OrderedDict()

    self.layers = nn.ModuleList()
    last_layer = 'conv0'
    last_n = 16
    fsize = 32
    strides = [1] * num_units[0] + \
              [2] + [1] * (num_units[1] - 1) + \
              [2] + [1] * (num_units[2] - 1)
    out_planes = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
    for i, (stride, n) in enumerate(zip(strides, out_planes)):
      self.layers.append(block(last_n, n, stride))
      if stride != 1:
        fsize /= 2

      for j in range(2):
        name = 'conv%d_%d' % (i, j)
        self.name_to_ind[name] = (i, j)
        self.ind_to_name[(i, j)] = name
        self.meta_data[name] = {'n': n,
                                'c': last_n,
                                'ksize': 3,
                                'padding': 1,
                                'fsize': fsize,
                                'stride': stride}
        self.name_to_next_name[last_layer] = name
        last_layer = name
        last_n = n
        stride = 1

    self.bn = nn.BatchNorm2d(64)
    self.logit = nn.Linear(64, num_classes)

    # Initialize weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')

  def forward(self, x):
    out = self.conv0(x)
    for layer in self.layers:
      out = layer(out)[0]

    out = self.bn(out)
    out = out.mean(2).mean(2)
    out = self.logit(out)
    return out

  def get_layer_names(self):
    return list(self.name_to_ind.keys())

  def get_conv_weight(self, layer_name):
    ind = self.name_to_ind[layer_name]
    weight = eval('self.layers[ind[0]].conv%d.weight' % ind[1])
    return weight.detach().cpu().numpy()

  def apply_1layer_mask(self, keep_inds, layer_name):
    ind = self.name_to_ind[layer_name]
    # set the keep_inds of the mask to 1.0, otherwise 0.0
    exec('self.layers[ind[0]].mask%d.fill_(0.0)' % ind[1])
    exec('self.layers[ind[0]].mask%d[0, keep_inds, 0, 0] = 1.0' % ind[1])
    return

  def apply_1layer_weight(self, layer_name, keep_inds, weight):
    ind = self.name_to_ind[layer_name]
    if keep_inds is not None:
      exec('self.layers[ind[0]].conv%d.weight.data[:, keep_inds, :, :] = weight[0]' % ind[1])
      # discard_inds = [i for i in range(self.meta_data[layer_name]['c']) if i not in keep_inds]
      # exec('self.layers[ind[0]].conv%d.weight.data[:, discard_inds, :, :].fill_(0.0)' % ind[1])
      if weight[1] is not None:
        exec('self.layers[ind[0]].skip_conv.weight.data[:, keep_inds, :, :] = weight[1]')
    else:
      exec('self.layers[ind[0]].conv%d.weight.data = weight[0]' % ind[1])
      if weight[1] is not None:
        exec('self.layers[ind[0]].skip_conv.weight.data = weight[1]')
    return

  def conv_input(self, x, layer_name, sample_inds=None):
    self.eval()
    # get sampled input of the layer with the specified layer name
    ind = self.name_to_ind[layer_name]
    x = self.conv0(x)
    for i, layer in enumerate(self.layers):
      x = layer(x)

      if i != ind[0]:
        x = x[0]
      else:
        need_identity = ind[1] == 1
        need_shortcut = x[2][2] is not None and ind[1] == 0

        conv_out = x[1][ind[1]]
        if sample_inds is not None:
          unfold = nn.Unfold(kernel_size=self.meta_data[layer_name]['ksize'],
                             padding=self.meta_data[layer_name]['padding'],
                             stride=self.meta_data[layer_name]['stride'])
          # extract image patches correspond to each output point
          # shape=[B, C x K x K, H x W]
          patches = unfold(conv_out.detach())
          # extract patches correspond to the sampled points
          # shape=[B, C x K x K, len(sample_inds)]
          # and reshape to [B, C, K, K, len(sample_inds)]
          patches = patches[:, :, sample_inds].view(conv_out.shape[0],
                                                    conv_out.shape[1],
                                                    self.meta_data[layer_name]['ksize'],
                                                    self.meta_data[layer_name]['ksize'],
                                                    -1)
          # transpose patches to [B, len(x_inds), C, K, K]
          patches = patches.permute([0, 4, 1, 2, 3]).contiguous()
          # reshape patches to [B x len(x_inds), C, K, K]
          patches = patches.view(-1,
                                 conv_out.shape[1],
                                 self.meta_data[layer_name]['ksize'],
                                 self.meta_data[layer_name]['ksize']).cpu().numpy()

          if need_shortcut:
            shortcut_unfold = nn.Unfold(kernel_size=1,
                                        padding=0,
                                        stride=self.meta_data[layer_name]['stride'])
            # extract image patches correspond to each output point
            # shape=[B, C x 1 x 1, H x W]
            shortcut_patches = shortcut_unfold(conv_out.detach())
            # extract patches correspond to the sampled points
            # shape=[B, C x 1 x 1, len(sample_inds)]
            # and reshape to [B, C, 1, 1, len(sample_inds)]
            shortcut_patches = \
              shortcut_patches[:, :, sample_inds].view(conv_out.shape[0],
                                                       conv_out.shape[1],
                                                       1, 1, -1)
            # transpose patches to [B, len(x_inds), C, 1, 1]
            shortcut_patches = shortcut_patches.permute([0, 4, 1, 2, 3]).contiguous()
            # reshape patches to [B x len(x_inds), C, 1, 1]
            shortcut_patches = shortcut_patches.view(-1, conv_out.shape[1], 1, 1).cpu().numpy()
          else:
            shortcut_patches = None

          if need_identity:
            identity = x[3]
            # layer output with shape [B, C, H, W], reshpae to [B, C, H x W]
            identity = identity.view(identity.shape[0], identity.shape[1], -1)
            # sample output with shape [B, C, len(sample_ind)]
            # and transpose to [B, len(sample_ind), C]
            identity = identity[:, :, sample_inds].permute([0, 2, 1]).contiguous()
            # reshape to [B x len(sample_ind), C]
            identity = identity.view(-1, identity.shape[-1]).cpu().numpy()
          else:
            identity = None
        else:
          patches = conv_out.detach().cpu().numpy()
          if need_shortcut:
            shortcut_patches = conv_out.detach().cpu().numpy()
          else:
            shortcut_patches = None
          if need_identity:
            identity = x[3].cpu().numpy()
          else:
            identity = None

        prev_prunable = ind[1] == 1
        return patches, prev_prunable, identity, shortcut_patches

  def conv_outputs(self, x, layers_sample_inds=None):
    self.eval()
    # return sampled output of all the layers
    # the layer_sample_inds should be a dict that contains all the layer names
    if layers_sample_inds is not None:
      assert layers_sample_inds.keys() == self.name_to_ind.keys()

    outputs = OrderedDict()
    x = self.conv0(x)
    for i, layer in enumerate(self.layers):
      x = layer(x)

      for j in range(2):
        conv_out = x[2][j]
        if layers_sample_inds is not None:
          # layer output with shape [B, C, H, W], reshpae to [B, C, H x W]
          conv_out = conv_out.view(x[2][j].shape[0], x[2][j].shape[1], -1)
          # sample output with shape [B, C, len(sample_ind)]
          conv_out = conv_out[:, :, layers_sample_inds[self.ind_to_name[(i, j)]]]
          # transpose to [B, len(sample_ind), C]
          conv_out = conv_out.permute([0, 2, 1]).contiguous()
          # reshape to [B x len(sample_ind), C]
          conv_out = conv_out.view(-1, conv_out.shape[-1])

        outputs[self.ind_to_name[(i, j)]] = conv_out.detach().cpu().numpy()

      if x[2][2] is not None:
        shortcut_out = x[2][2]
        if layers_sample_inds is not None:
          # layer output with shape [B, C, H, W], reshpae to [B, C, H x W]
          shortcut_out = shortcut_out.view(shortcut_out.shape[0],
                                           shortcut_out.shape[1],
                                           -1)
          # sample output with shape [B, C, len(sample_ind)]
          # and transpose to [B, len(sample_ind), C]
          sample_inds = layers_sample_inds[self.ind_to_name[(i, 0)]]
          shortcut_out = shortcut_out[:, :, sample_inds].permute([0, 2, 1]).contiguous()
          # reshape to [B x len(sample_ind), C]
          shortcut_out = shortcut_out.view(-1, shortcut_out.shape[-1])
        outputs[self.ind_to_name[(i, 0)] + '_shortcut'] = shortcut_out.detach().cpu().numpy()

      x = x[0]
    return outputs


def resnet20():
  return PreActResNet(PreActBlock, [3, 3, 3], num_classes=10)


def resnet56():
  return PreActResNet(PreActBlock, [9, 9, 9], num_classes=10)


if __name__ == '__main__x':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)

  net = resnet20()

  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  layer_names = net.get_layer_names()
  img = torch.randn(1, 3, 32, 32)
  outs = net.conv_outputs(img, layers_sample_inds={k: [3, 4, 5] for k in layer_names})
  inp = net.conv_input(img, layer_name=layer_names[6], sample_inds=[3, 4, 5])

  net.apply_1layer_action(action=[0.5], layer_name=layer_names[0])
  net.apply_1layer_weight(layer_name=layer_names[0],
                          keep_inds=np.arange(8),
                          weight=torch.zeros(16, 8, 3, 3))

  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())

