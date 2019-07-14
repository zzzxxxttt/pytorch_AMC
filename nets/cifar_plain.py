from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class standard_block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(standard_block, self).__init__()
    self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.register_buffer('mask', torch.ones(1, out_channels, 1, 1))

  def forward(self, x):
    out = self.conv2d(x)
    conv_out = out
    out = self.bn(out)
    out = self.relu(out)
    if self.mask is not None:
      out = out * self.mask
    return out, conv_out


class PLAIN(nn.Module):
  def __init__(self, conv_config, num_classes):
    super(PLAIN, self).__init__()
    self.layers = nn.ModuleList()

    self.name_to_ind = OrderedDict()
    self.ind_to_name = OrderedDict()
    self.meta_data = OrderedDict()
    self.name_to_next_name = OrderedDict()

    self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
    self.bn0 = nn.BatchNorm2d(16)

    last_layer = 'input'
    last_n = 16
    fsize = 32
    for i, n in enumerate(conv_config):
      if n == 'M':
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        fsize /= 2
      else:
        self.layers.append(standard_block(last_n, n))
        name = 'conv%d' % len(self.name_to_ind)
        self.name_to_ind[name] = i
        self.ind_to_name[i] = name
        self.meta_data[name] = {'n': n, 'c': last_n, 'ksize': 3, 'fsize': fsize}
        self.name_to_next_name[last_layer] = name
        last_layer = name
        last_n = n

    self.fc = nn.Linear(last_n, num_classes)

    # Initialize weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
        # m.bias.data.zero_()

  def forward(self, x):
    x = F.relu(self.bn0(self.conv0(x)))

    for layer in self.layers:
      x = layer(x)
      if isinstance(layer, standard_block):
        x = x[0]

    x = x.mean(2).mean(2)
    x = self.fc(x)
    return x

  def get_layer_names(self):
    return list(self.name_to_ind.keys())

  def get_conv_weight(self, layer_name):
    weight = self.layers[self.name_to_ind[layer_name]].conv2d.weight
    return weight.detach().cpu().numpy()

  def apply_layer_mask(self, keep_inds, layer_name):
    # set the keep_inds of the mask to 1.0, otherwise 0.0
    self.layers[self.name_to_ind[layer_name]].mask.fill_(0.0)
    self.layers[self.name_to_ind[layer_name]].mask[0, keep_inds, 0, 0] = 1.0
    return

  def apply_layer_weight(self, layer_name, keep_inds, weight):
    if keep_inds is not None:
      self.layers[self.name_to_ind[layer_name]].conv2d.weight.data[:, keep_inds, :, :] = weight[0]
    else:
      self.layers[self.name_to_ind[layer_name]].conv2d.weight.data = weight[0]
    return

  def conv_input(self, x, layer_name, sample_inds=None):
    self.eval()
    x = F.relu(self.bn0(self.conv0(x)))

    # get sampled input of a layer
    for ind, layer in enumerate(self.layers):
      if not isinstance(layer, standard_block):
        x = layer(x)
      else:
        if ind < self.name_to_ind[layer_name]:
          x = layer(x)[0]
        else:
          if sample_inds is not None:
            unfold = nn.Unfold(kernel_size=layer.conv2d.kernel_size,
                               padding=layer.conv2d.padding,
                               stride=layer.conv2d.stride)
            # extract image patches wrt to each output point
            # shape=[B, C x K x K, H x W]
            patches = unfold(x.detach())
            # extract patches wrt to the sampled points
            # shape=[B, C x K x K, len(sample_inds)]
            # and reshape to [B, C, K, K, len(sample_inds)]
            patches = patches[:, :, sample_inds].view(x.shape[0],
                                                      x.shape[1],
                                                      layer.conv2d.kernel_size[0],
                                                      layer.conv2d.kernel_size[1],
                                                      -1)
            # transpose patches to [B, len(x_inds), C, K, K]
            patches = patches.permute([0, 4, 1, 2, 3]).contiguous()
            # reshape patches to [B x len(x_inds), C, K, K]
            patches = patches.view(-1,
                                   x.shape[1],
                                   layer.conv2d.kernel_size[0],
                                   layer.conv2d.kernel_size[1])
          else:
            patches = x.detach()

          return patches.cpu().numpy(), True, None, None

  def conv_outputs(self, x, layers_sample_inds=None):
    self.eval()
    # return sampled output of all the layers
    # the layer_sample_inds should be a dict that contains all the layer names
    if layers_sample_inds is not None:
      assert layers_sample_inds.keys() == self.name_to_ind.keys()
    outputs = OrderedDict()

    x = F.relu(self.bn0(self.conv0(x)))

    for i, layer in enumerate(self.layers):
      x = layer(x)
      if isinstance(layer, standard_block):
        out = x[1]
        if layers_sample_inds is not None:
          # layer output with shape [B, C, H, W], reshpae to [B, C, H x W]
          out = out.view(x[1].shape[0], x[1].shape[1], -1)
          # sample output with shape [B, C, len(sample_ind)]
          out = out[:, :, layers_sample_inds[self.ind_to_name[i]]]
          # transpose to [B, len(sample_ind), C]
          out = out.permute([0, 2, 1]).contiguous()
          # reshape to [B x len(sample_ind), C]
          out = out.view(-1, out.shape[-1])
        outputs[self.ind_to_name[i]] = out.detach().cpu().numpy()
        x = x[0]
    return outputs


def plain20(num_classes=10):
  return PLAIN(conv_config=[16, 16, 16, 16, 16, 16, 'M',
                            32, 32, 32, 32, 32, 32, 'M',
                            64, 64, 64, 64, 64, 64, 'M'],
               num_classes=num_classes)


def plain8(num_classes=10):
  return PLAIN(conv_config=[16, 16, 'M', 32, 32, 'M', 64, 64, 'M'],
               num_classes=num_classes)

# if __name__ == '__main__':
#
#   def hook(self, input, output):
#     print(output.data.cpu().numpy().shape)
#
#   net = plain20()
#
#   for m in net.modules():
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#       m.register_forward_hook(hook)
#
#   layer_names = net.get_layer_names()
#
#   img = torch.randn(1, 3, 32, 32)
#   outs = net.conv_outputs(img, layers_sample_inds={k: [1, 2, 3] for k in layer_names})
#   inp = net.conv_input(img, layer_name=layer_names[0], sample_inds=[1, 2, 3, 4, 5])
#
#   y = net(torch.randn(1, 3, 32, 32))
#   print(y)
