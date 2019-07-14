import numpy as np
from collections import OrderedDict

odic = OrderedDict


class VGG_Proxy:
  def __init__(self, conv_config, lim_type, ratio, min_action, lower_bound):
    # the minimum action for each layer
    self.lim_type = lim_type
    self.ratio = ratio
    self.lower_bound = lower_bound  # todo perform test without this
    self.min_action = min_action

    self.names = []
    self.name_to_ind = OrderedDict()
    self.ind_to_name = OrderedDict()
    self.meta_data = OrderedDict()
    self.name_to_next_name = OrderedDict()
    self.orig_paras = OrderedDict()
    self.orig_flops = OrderedDict()
    self.min_paras = OrderedDict()
    self.min_flops = OrderedDict()

    last_layer = 'input'
    last_n = 3
    last_min_n = 3
    fsize = 32
    for i, n in enumerate(conv_config):
      if n == 'M':
        fsize /= 2
      else:
        name = 'conv%d' % len(self.name_to_ind)
        self.names.append(name)
        self.name_to_ind[name] = i
        self.ind_to_name[i] = name
        self.meta_data[name] = {'n': n, 'c': last_n, 'ksize': 3, 'fsize': fsize}
        self.name_to_next_name[last_layer] = name
        self.orig_paras[name] = 3 * 3 * n * last_n / 1e6
        self.orig_flops[name] = 3 * 3 * n * last_n * fsize ** 2 / 1e6
        self.min_paras[name] = 3 * 3 * np.round(n * min_action) * last_min_n / 1e6
        self.min_flops[name] = 3 * 3 * np.round(n * min_action) * last_min_n * fsize ** 2 / 1e6
        last_layer = name
        last_n = n
        last_min_n = np.round(n * min_action)

    # do not compress the last conv layer
    last_last_min_n = np.round(self.meta_data[last_layer]['c'] * min_action)
    last_fsize = self.meta_data[last_layer]['fsize']
    self.min_paras[last_layer] = 3 * 3 * last_n * last_last_min_n / 1e6
    self.min_flops[last_layer] = 3 * 3 * last_n * last_last_min_n * last_fsize ** 2 / 1e6

    self.total_para = sum(self.orig_paras.values())
    self.total_flops = sum(self.orig_flops.values())
    self.para_upper_bound = self.total_para * ratio
    self.para_lower_bound = self.total_para * (ratio - 0.05)
    self.flops_upper_bound = self.total_flops * ratio
    self.flops_lower_bound = self.total_flops * (ratio - 0.05)

    # these values are for state normalization
    self.max_num_filters = np.amax([self.meta_data[l]['n'] for l in self.names])
    self.max_layer_para = np.amax(list(self.orig_paras.values()))
    self.max_layer_flops = np.amax(list(self.orig_flops.values()))

    self.layer_now = None  # the name of the current layer
    self.filters_now = None  # the num of filters in the generated layers
    self.states_now = None  # the list of all previous states
    self.actions_now = None  # the list of generated actions
    self.paras_now = None  # the total num of parameters until the current layer
    self.flops_now = None  # the total flops until the current layer

  def init_episode(self):
    name = self.name_to_next_name['input']
    self.layer_now = name
    self.filters_now = [3]
    self.paras_now = []
    self.flops_now = []
    self.actions_now = []

    if self.lim_type == 'para':
      value1 = sum(self.paras_now) / self.para_upper_bound
      value2 = self.orig_paras[name] / self.max_layer_para
    else:
      value1 = sum(self.flops_now) / self.flops_upper_bound
      value2 = self.orig_flops[name] / self.max_layer_flops

    self.states_now = \
      [(self.names.index(name),  # layer index
        self.meta_data[name]['n'] / self.max_num_filters,  # layer filters
        self.meta_data[name]['c'] / self.max_num_filters,  # layer channels
        self.meta_data[name]['fsize'] / 32,  # layer input size
        value1,  # total para till current state
        value2)]  # para of current layer
    return self.states_now[-1], 0, False

  # return cliped action of the current state
  def compress(self, action):

    # channels of current layer(num of filters of the previous layer)
    prev_n = self.filters_now[-1]
    # original num of filters of the current layer
    current_n = self.meta_data[self.layer_now]['n']

    # the minimum para and flops #after# the next layer
    future_min_para = 0
    future_max_para = 0
    future_min_flops = 0
    future_max_flops = 0
    next_name = self.name_to_next_name[self.layer_now]
    while next_name in self.name_to_next_name:
      future_min_para += self.min_paras[self.name_to_next_name[next_name]]
      future_max_para += self.orig_paras[self.name_to_next_name[next_name]]
      future_min_flops += self.min_flops[self.name_to_next_name[next_name]]
      future_max_flops += self.orig_flops[self.name_to_next_name[next_name]]
      next_name = self.name_to_next_name[next_name]

    next_name = self.name_to_next_name[self.layer_now]
    fs = self.meta_data[self.layer_now]['fsize']
    fs_next = self.meta_data[next_name]['fsize']
    # if next layer is not the last layer, the minimum num of filters is n x min_action,
    # otherwise the minimum num of filters is n
    if next_name in self.name_to_next_name:
      next_n_min = np.round(self.meta_data[next_name]['n'] * self.min_action)
    else:
      next_n_min = self.meta_data[next_name]['n']

    # maximum num of filters of the next layer
    next_n_max = self.meta_data[next_name]['n']

    # maximum and minimum num of filters of the current layer
    if self.lim_type == 'para':
      max_n = (self.para_upper_bound - sum(self.paras_now) - future_min_para) \
              * 1e6 / 9 / (prev_n + next_n_min)
      min_n = (self.para_lower_bound - sum(self.paras_now) - future_max_para) \
              * 1e6 / 9 / (prev_n + next_n_max)
    else:
      max_n = (self.flops_upper_bound - sum(self.flops_now) - future_min_flops) \
              * 1e6 / 9 / (prev_n * fs ** 2 + next_n_min * fs_next ** 2)
      min_n = (self.flops_lower_bound - sum(self.flops_now) - future_max_flops) \
              * 1e6 / 9 / (prev_n * fs ** 2 + next_n_max * fs_next ** 2)

    max_a = np.clip(max_n / current_n, self.min_action, 1)  # maximum action of current layer
    min_a = np.clip(min_n / current_n, self.min_action, 1)  # minimum action of current layer

    if self.lower_bound:
      real_a = np.clip(action, min_a, max_a)
    else:
      real_a = np.clip(action, None, max_a)

    real_n = np.round(real_a * current_n)
    real_n = np.maximum(real_n, 1)  # at least keep 1 filter in a layer

    self.actions_now.append(real_a)
    self.filters_now.append(real_n)
    self.paras_now.append(3 * 3 * prev_n * real_n / 1e6)
    self.flops_now.append(3 * 3 * prev_n * real_n * fs ** 2 / 1e6)

    # move state to the next layer
    self.layer_now = next_name
    # don't prune the last conv layer
    done = next_name not in self.name_to_next_name
    if not done:
      if self.lim_type == 'para':
        value1 = sum(self.paras_now) / self.para_upper_bound
        value2 = self.orig_paras[next_name] / self.max_layer_para
      else:
        value1 = sum(self.flops_now) / self.flops_upper_bound
        value2 = self.orig_flops[next_name] / self.max_layer_flops
      self.states_now.append((self.names.index(next_name) / (len(self.names) - 1),
                              self.meta_data[next_name]['n'] / self.max_num_filters,
                              self.meta_data[next_name]['c'] / self.max_num_filters,
                              self.meta_data[next_name]['fsize'] / 32,
                              value1,
                              value2))
    else:
      self.paras_now.append(3 * 3 * real_n * next_n_max / 1e6)
      self.flops_now.append(3 * 3 * real_n * next_n_max * fs_next ** 2 / 1e6)

    return self.states_now[-1], real_a, done


def vgg16_proxy(lim_type, ratio, min_action, lower_bound=True):
  return VGG_Proxy(conv_config=[64, 64, 'M',
                                128, 128, 'M',
                                256, 256, 256, 'M',
                                512, 512, 512, 'M',
                                512, 512, 512, 'M'],
                   lim_type=lim_type,
                   ratio=ratio,
                   min_action=min_action,
                   lower_bound=lower_bound)


def vgg16s_proxy(lim_type, ratio, min_action, lower_bound=True):
  return VGG_Proxy(conv_config=[16, 16, 'M',
                                32, 32, 'M',
                                64, 64, 64, 'M',
                                64, 64, 64, 'M',
                                64, 64, 64, 'M'],
                   lim_type=lim_type,
                   ratio=ratio,
                   min_action=min_action,
                   lower_bound=lower_bound)


# if __name__ == '__main__':
#   import time
#   from utils.io import *
#
#   proxy = vgg16_proxy(lim_type='flops',
#                       ratio=0.5,
#                       min_action=0.1,
#                       lower_bound=False)
#   print(proxy.total_para)
#   print(proxy.total_flops)
#
#   i = 1
#   state, _, done = proxy.init_episode()
#   while not done:
#     next_state, action, done = proxy.compress(1.0)
#     print('{}\t {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(i, *state), action)
#     state = next_state
#     i += 1
#
#   print(np.sum(proxy.paras_now), np.sum(proxy.paras_now) / proxy.total_para)
#   print(np.sum(proxy.flops_now), np.sum(proxy.flops_now) / proxy.total_flops)
