import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import pickle
import numpy as np
from torchvision.datasets.folder import *


class CIFAR10_split(data.Dataset):
  """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
  """
  base_folder = 'cifar-10-batches-py'
  train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
  ]

  test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
  ]

  def __init__(self, root, split, split_size=10000, transform=None, target_transform=None):
    assert split in ['train', 'val', 'test']
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.split = split  # training set or test set

    # now load the picked numpy arrays
    if self.split == 'test':
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()

      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      self.data = []
      self.labels = []
      for fentry in self.train_list:
        f = fentry[0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        entry = pickle.load(fo, encoding='latin1')
        self.data.append(entry['data'])
        if 'labels' in entry:
          self.labels += entry['labels']
        else:
          self.labels += entry['fine_labels']
        fo.close()

      self.data = np.concatenate(self.data)
      self.data = self.data.reshape((10000 * len(self.train_list), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      if self.split == 'train':
        self.data = self.data[:split_size]
        self.labels = self.labels[:split_size]
      elif self.split == 'val':
        self.data = self.data[-split_size:]
        self.labels = self.labels[-split_size:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]
    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data)


if __name__ == '__main__':
  from tqdm import tqdm

  dataset = CIFAR10_split('../data', split='val', split_size=1000)
  for img, label in tqdm(dataset):
    pass
