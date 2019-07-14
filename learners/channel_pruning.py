import time
from collections import OrderedDict

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, LassoLars, LinearRegression

import torch

from utils.dataset import CIFAR10_split
from utils.preprocessing import cifar_transform

np.random.seed(12345)

class channelPruner:
  def __init__(self, method, model, proxy,
               pretrain_dir, num_sample_points,
               data_dir, batch_size=300, num_loader_workers=0,
               gpu_id=0, debug=False):

    assert method in ['abs_value', 'channel_pruning']
    self.method = method
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_sample_points = num_sample_points
    self.num_loader_workers = num_loader_workers
    self.pretrain_dir = pretrain_dir
    self.save_dir = None  # the filename of generated pruned model
    self.gpu_id = gpu_id
    self.device = torch.device('cuda:%d' % gpu_id)
    self.debug = debug

    self.model = model
    self.proxy = proxy

    # put images to RAM for fast inference
    self.train_data_cache = None
    self.train_target_cache = None
    self.val_data_cache = None
    self.val_target_cache = None

    # the original state dict before pruning
    self.pretrain_dict = None

    # the sample points for layer reconstruction
    self.sample_inds = OrderedDict()
    for layer in self.proxy.names:
      fs = self.proxy.meta_data[layer]['fsize']
      self.sample_inds[layer] = \
        np.random.choice(int(fs ** 2), int(min(num_sample_points, fs ** 2)), replace=False)

    # the cache of layer outputs
    self.conv_outputs = None

  # this function should NOT be called inside the __init__
  def regenerate_train_data_cache(self):
    with torch.cuda.device(self.gpu_id):
      # data for lasso regression and layer reconstruction
      train_dataset = CIFAR10_split(root=self.data_dir,
                                    split='train',
                                    split_size=2000,
                                    transform=cifar_transform(is_training=False))
      train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_loader_workers)

      self.train_data_cache = []
      self.train_target_cache = []
      for inputs, targets in train_loader:
        self.train_data_cache.append(inputs)
        self.train_target_cache.append(targets)

      self.train_data_cache = torch.cat(self.train_data_cache, dim=0).to(self.device)
      self.train_target_cache = torch.cat(self.train_target_cache, dim=0).to(self.device)
      return

  # this function should NOT be called inside the __init__
  def regenerate_val_data_cache(self):
    with torch.cuda.device(self.gpu_id):
      # data for evaluate pruning
      val_dataset = CIFAR10_split(root=self.data_dir,
                                  split='val',
                                  split_size=5000,
                                  transform=cifar_transform(is_training=False))
      val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False,
                                               num_workers=self.num_loader_workers)

      self.val_data_cache = []
      self.val_target_cache = []
      for inputs, targets in val_loader:
        self.val_data_cache.append(inputs)
        self.val_target_cache.append(targets)

      self.val_data_cache = torch.cat(self.val_data_cache, dim=0).to(self.device)
      self.val_target_cache = torch.cat(self.val_target_cache, dim=0).to(self.device)
      return

  # this function should NOT be called inside the __init__
  def regenerate_network_outputs(self):
    with torch.cuda.device(self.gpu_id):
      # the cache of layer outputs
      self.conv_outputs = {}
      with torch.no_grad():
        num_batches = int(np.ceil(self.train_data_cache.shape[0] / self.batch_size))
        for i in range(num_batches):
          inputs = self.train_data_cache[i * self.batch_size:(i + 1) * self.batch_size]
          outputs = self.model.conv_outputs(inputs.cuda(), self.sample_inds)
          for l in outputs.keys():
            if l not in self.conv_outputs:
              self.conv_outputs[l] = [outputs[l]]
            else:
              self.conv_outputs[l].append(outputs[l])

      self.conv_outputs = {k: np.vstack(v) for k, v in self.conv_outputs.items()}
      return

  def regenerate_sample_inds(self):
    for layer in self.proxy.names:
      fs = self.proxy.meta_data[layer]['fsize']
      self.sample_inds[layer] = \
        np.random.choice(int(fs ** 2), int(min(self.num_sample_points, fs ** 2)), replace=False)

    # the output need to be regenerated due to the sample index change
    self.regenerate_network_outputs()
    return

  # evaluate pruned network on validation set
  def evaluate(self):
    with torch.cuda.device(self.gpu_id):
      if self.val_data_cache is None or self.val_target_cache is None:
        self.regenerate_val_data_cache()

      self.model.eval()
      with torch.no_grad():
        correct = 0
        num_batches = int(np.ceil(self.val_data_cache.shape[0] / self.batch_size))
        for i in range(num_batches):
          inputs = self.val_data_cache[i * self.batch_size:(i + 1) * self.batch_size]
          targets = self.val_target_cache[i * self.batch_size:(i + 1) * self.batch_size]

          outputs = self.model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          correct += predicted.eq(targets.data).cpu().sum().item()
        acc = 100. * correct / self.val_data_cache.shape[0]
      return acc

  def init_episode(self, save_dir=None):
    with torch.cuda.device(self.gpu_id):
      if self.pretrain_dict is None:
        # load network that need to be compressed
        pretrain_dict = torch.load(self.pretrain_dir)['state_dict']
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        for k in pretrain_dict.keys():
          if k not in model_dict:
            print('%s not in model dict !' % k)
        for k in model_dict.keys():
          if k not in pretrain_dict:
            print('%s not in pretrained dict !' % k)

        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        self.pretrain_dict = {k: model_dict[k] for k in model_dict if 'fc' not in k}

      model_dict = self.model.state_dict()
      model_dict.update(self.pretrain_dict)
      self.model.load_state_dict(model_dict)
      # self.model.load_state_dict(self.pretrain_dict)
      self.save_dir = save_dir

      self.model.to(self.device)

      if self.train_data_cache is None:
        self.regenerate_train_data_cache()

      if self.conv_outputs is None:
        self.regenerate_network_outputs()

      state, action, done = self.proxy.init_episode()
    return state, action, done, 0

  def compress(self, action):
    with torch.cuda.device(self.gpu_id):
      layer_now = self.proxy.layer_now
      layer_next = self.proxy.name_to_next_name[layer_now]
      # generate action according to current state
      next_state, real_action, num_channels, done = self.proxy.compress(action)

      # ----------------------------------- step-1 -----------------------------------
      # get the input of next conv layer
      tic = time.perf_counter()
      conv_inputs = []
      shortcut_inputs = []
      identities = []
      prev_prunable = True
      with torch.no_grad():
        num_batches = int(np.ceil(self.train_data_cache.shape[0] / self.batch_size))
        for i in range(num_batches):
          inputs = self.train_data_cache[i * self.batch_size:(i + 1) * self.batch_size]
          patches, prev_prunable, identity, shortcut_patches = \
            self.model.conv_input(inputs, layer_next, self.sample_inds[layer_next])
          conv_inputs.append(patches)
          identities.append(identity)
          shortcut_inputs.append(shortcut_patches)

      conv_inputs = np.concatenate(conv_inputs, axis=0)
      if identities[0] is not None:
        identities = np.concatenate(identities, axis=0)
      if shortcut_inputs[0] is not None:
        shortcut_inputs = np.concatenate(shortcut_inputs, axis=0)
        shortcut_inputs = shortcut_inputs.reshape([shortcut_inputs.shape[0], -1])
      if self.debug:
        print('gen input time: %.3f' % (time.perf_counter() - tic))

      # ----------------------------------- step-2 -----------------------------------
      # get the original output of next conv layer
      conv_outputs = self.conv_outputs[layer_next]
      if identities[0] is not None:
        # a': original indentity branch
        # b': original conv output
        # a : new identity branch
        # b : new conv ouput
        # we want a + b = a' + b' , but we can only control b,
        # so we need to find b such that b = a' + b' - a
        conv_outputs = conv_outputs - identities

      shortcut_outputs = None
      if shortcut_inputs[0] is not None:
        shortcut_outputs = self.conv_outputs[layer_next + '_shortcut']

      # ----------------------------------- step-3 -----------------------------------
      # generate indexes of the keeped channels from action and mask out the pruned filters
      if self.method == 'abs_value':
        weight = self.model.get_conv_weight(layer_name=layer_next)
        keep_inds = self.get_index_by_abs_value(weight, num_channels)
      else:
        weight = self.model.get_conv_weight(layer_name=layer_next)
        keep_inds, real_action = self.get_index_by_channel_pruning(X=conv_inputs,
                                                                   Y=conv_outputs,
                                                                   W=weight,
                                                                   c_new=num_channels,
                                                                   debug=self.debug)

      self.model.apply_layer_mask(keep_inds, layer_name=layer_now)

      # ----------------------------------- step-4 -----------------------------------
      if num_channels < self.proxy.meta_data[layer_next]['c']:
        # performing regression using the inputs and the outputs
        tic = time.perf_counter()
        if prev_prunable:
          # get the input of next layer after pruned and reshape to [B, new_channel x K x K]
          conv_inputs = conv_inputs[:, keep_inds, :, :].reshape([conv_inputs.shape[0], -1])
        else:
          conv_inputs = conv_inputs.reshape([conv_inputs.shape[0], -1])

        conv_regresser = \
          LinearRegression(fit_intercept=False).fit(conv_inputs, conv_outputs)
        if self.debug:
          print('regression time: %.3f' % (time.perf_counter() - tic))

        regression_score = [r2_score(conv_outputs,
                                     conv_regresser.predict(conv_inputs),
                                     sample_weight=None,
                                     multioutput='variance_weighted') - 1]
        # get the regressed new weight and assign to the network
        n = self.proxy.meta_data[layer_next]['n']
        k = self.proxy.meta_data[layer_next]['ksize']
        conv_weight = torch.from_numpy(conv_regresser.coef_.reshape([n, -1, k, k])).float().cuda()

        if shortcut_outputs is not None:
          tic = time.perf_counter()
          shortcut_regresser = \
            LinearRegression(fit_intercept=False).fit(shortcut_inputs, shortcut_outputs)
          if self.debug:
            print('regression time: %.3f' % (time.perf_counter() - tic))

          regression_score.append(r2_score(shortcut_outputs,
                                           shortcut_regresser.predict(shortcut_inputs),
                                           sample_weight=None,
                                           multioutput='variance_weighted') - 1)
          # get the regressed new weight and assign to the network
          n = self.proxy.meta_data[layer_next]['n']
          shortcut_weight = torch.from_numpy(shortcut_regresser.coef_.reshape([n, -1, 1, 1])).float().cuda()
        else:
          shortcut_weight = None

        self.model.apply_layer_weight(layer_next,
                                      keep_inds=keep_inds if prev_prunable else None,
                                      weight=(conv_weight, shortcut_weight))
      else:
        regression_score = [0]

      if done and self.save_dir is not None:
        tic = time.perf_counter()
        torch.save(self.model.state_dict(), self.save_dir)
        if self.debug:
          print('end episode time: %.3f' % (time.perf_counter() - tic))

      return next_state, real_action, done, regression_score

  @staticmethod
  def get_index_by_abs_value(W, c_new):
    keep_inds = []
    # pruning weight according to the squred value of filters
    weight = np.sum(W ** 2, axis=(0, 2, 3))
    weight_inds = np.argsort(weight)
    if c_new > 0:
      keep_inds.append(weight_inds[-c_new:])
    if len(keep_inds) > 0:
      keep_inds = np.concatenate(keep_inds)
    else:
      keep_inds = np.array([0])
    return keep_inds

  @staticmethod
  def get_index_by_channel_pruning(X, Y, W, c_new, alpha=1e-4, tolerance=0.02, debug=False):
    # X shape: [B, c_in, 3, 3]
    # Y shape: [B, c_out]
    # W shape: [c_out, c_in, 3, 3]
    num_samples = X.shape[0]  # num of training samples
    c_in = W.shape[1]  # num of input channels
    c_out = W.shape[0]  # num of output channels
    # select subset of training samples
    # subset_inds = np.random.choice(num_samples, min(400, num_samples // 20))
    subset_inds = np.random.choice(num_samples, 400)
    # sample and reshape X to [c_in, subset_size, 9]
    reshape_X = X.reshape([num_samples, c_in, -1])[subset_inds].transpose([1, 0, 2])
    # reshape W to [c_in, 9, c_out]
    reshape_W = W.reshape((c_out, c_in, -1)).transpose([1, 2, 0])
    # reshape Y to [subset_size x c_out]
    reshape_Y = Y[subset_inds].reshape(-1)

    # product has size [subset_size x c_out, c_in]
    product = np.matmul(reshape_X, reshape_W).reshape((c_in, -1)).T

    # use LassoLars because it's more robust than Lasso
    solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)

    # solver = Lasso(alpha=alpha, fit_intercept=False,
    #                max_iter=3000, warm_start=True, selection='random')

    def solve(alpha):
      """ Solve the Lasso"""
      solver.alpha = alpha
      solver.fit(product, reshape_Y)
      nonzero_inds = np.where(solver.coef_ != 0.)[0]
      nonzero_num = sum(solver.coef_ != 0.)
      return nonzero_inds, nonzero_num, solver.coef_

    tic = time.perf_counter()

    if c_new == c_in:
      keep_inds = np.arange(c_new)
      keep_num = c_new
    elif c_new == 0:
      keep_inds = np.array([0])
      keep_num = 1
    else:
      left = 0  # minimum alpha is 0, which means don't use lasso regularizer at all
      right = alpha

      # the left bound of num of selected channels
      lbound = np.clip(c_new - tolerance * c_in / 2, 1, None)
      # the right bound of num of selected channels
      rbound = c_new + tolerance * c_in / 2

      # increase alpha until the lasso can find a selection with size < c_new
      while True:
        _, keep_num, coef = solve(right)
        if keep_num < c_new:
          break
        else:
          right *= 2
          if debug:
            print("relax right to %.3f" % right)
            print("expected %d channels, but got %d channels" % (c_new, keep_num))

      # shrink the alpha for less aggressive lasso regularization
      # if the selected num of channels is less than the lbound
      while True:
        keep_inds, keep_num, coef = solve(alpha)
        # print loss
        loss = 1 / (2 * float(product.shape[0])) * \
               np.sqrt(np.sum((reshape_Y - np.matmul(product, coef)) ** 2, axis=0)) + \
               alpha * np.sum(np.fabs(coef))

        if debug:
          print('loss: %.3f, alpha: %.3f, feature nums: %d, '
                'left: %.3f, right: %.3f, left_bound: %.3f, right_bound: %.3f' %
                (loss, alpha, keep_num, left, right, lbound, rbound))

        if lbound <= keep_num <= rbound:
          break
        elif abs(left - right) <= right * 0.1:
          if lbound > 1:
            lbound = lbound - 1
          if rbound < c_in:
            rbound = rbound + 1
          left = left / 1.2
          right = right * 1.2
        elif keep_num > rbound:
          left = left + (alpha - left) / 2
        else:
          right = right - (right - alpha) / 2

        if alpha < 1e-10:
          break

        alpha = (left + right) / 2

    # if debug:
    # print('LASSO Regression time: %.2f s' % (time.perf_counter() - tic))
    # print(c_new, keep_num)
    return keep_inds, [keep_num / c_in]

# if __name__ == '__main__':
#   from nets.cifar_vgg import *
#   from nets.cifar_vgg_proxy import *
#   from nets.cifar_resnet import *
#   from nets.cifar_resnet_proxy import *
#   from nets.cifar_plain import *
#   from nets.cifar_plain_proxy import *
#
#   learner = channelPruner(method='abs_value',
#                           model=plain20(),
#                           proxy=plain20_proxy(lim_type='flops',
#                                               ratio=0.5,
#                                               min_action=0.2,
#                                               lower_bound=False),
#                           pretrain_dir='../ckpt/plain20_baseline.t7',
#                           num_sample_points=10,
#                           data_dir='../data',
#                           gpu_id=0,
#                           debug=False)
#
#   tic = time.perf_counter()
#   _, _, done, _ = learner.init_episode()
#   # learner.regenerate_sample_inds()
#   while not done:
#     _, _, done, score = learner.compress([0.3 * i] * 8)
#     print(i, '\t', score)
#   print('acc= ', learner.evaluate(), 'flops= ', np.sum(learner.proxy.flops_now))
