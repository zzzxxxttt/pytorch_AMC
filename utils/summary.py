import os
import pickle
from datetime import datetime

# return a fake summarywriter if tensorbaordX is not installed

try:
  from tensorboardX import SummaryWriter
except ImportError:
  class SummaryWriter:
    def __init__(self, log_dir=None, comment='', **kwargs):
      print('\nunable to import tensorboardX, log will be recorded in pickle format!\n')
      self.log_dir = log_dir if log_dir is not None else './logs'
      os.makedirs('./logs', exist_ok=True)
      self.logs = {'comment': comment}
      return

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
      if tag in self.logs:
        self.logs[tag].append((scalar_value, global_step, walltime))
      else:
        self.logs[tag] = [(scalar_value, global_step, walltime)]
      return

    def close(self):
      timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_')
      with open(os.path.join(self.log_dir, 'log_%s.pickle' % timestamp), 'wb') as handle:
        pickle.dump(self.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
      return

# if __name__ == '__main__':
#   sw = SummaryWriter()
#   sw.close()
