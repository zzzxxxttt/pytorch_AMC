import numpy as np
from collections import deque


class Memory(object):
  def __init__(self, batch_size, memory_size=10000):
    self.batch_size = batch_size
    self.memory = deque(maxlen=memory_size)
    self.memory_size = memory_size

  def __len__(self):
    return len(self.memory)

  def append(self, item):
    self.memory.append(item)

  def sample_batch(self):
    idx = np.random.permutation(len(self.memory))[:self.batch_size]
    return [self.memory[i] for i in idx]
