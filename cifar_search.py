import os

# 3~5 is enough, otherwise sklearn will try to use all the CPUs
os.environ["OPENBLAS_NUM_THREADS"] = '3'
os.environ["MKL_NUM_THREADS"] = '3'

import pickle
import argparse
from datetime import datetime

from nets.cifar_vgg import vgg16
from nets.cifar_plain import plain20
from nets.cifar_resnet import resnet56

from nets.cifar_vgg_proxy import vgg16_proxy
from nets.cifar_plain_proxy import plain20_proxy
from nets.cifar_resnet_proxy import resnet56_proxy

from agents.ddpg import Agent

from learners.channel_pruning import channelPruner

parser = argparse.ArgumentParser(description='pytorch_AMC')
parser.add_argument('--log_name', type=str, default='RL_plain20_flops0.5')

parser.add_argument('--model', type=str, default='plain20')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/plain20_baseline.t7')

parser.add_argument('--method', type=str, default='channel_pruning')
parser.add_argument('--num_sample_points', type=int, default=10)
parser.add_argument('--lim_type', type=str, default='flops')
parser.add_argument('--lim_ratio', type=float, default=0.5)
parser.add_argument('--min_action', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--buffer_size', type=int, default=2000)
parser.add_argument('--min_buffer_size', type=int, default=1000)
parser.add_argument('--max_steps', type=int, default=500)

cfg = parser.parse_args()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def main():
  if cfg.model == 'plain20':
    netowrk_fn, proxy_fn = plain20, plain20_proxy
  elif cfg.model == 'vgg16':
    netowrk_fn, proxy_fn = vgg16, vgg16_proxy
  elif cfg.model == 'resnet56':
    netowrk_fn, proxy_fn = resnet56, resnet56_proxy
  else:
    raise NotImplementedError

  agent = Agent(state_size=6,
                action_size=1,
                max_epsiodes=cfg.max_steps,
                action_min=cfg.min_action,
                buffer_size=cfg.buffer_size,
                min_buffer_size=cfg.min_buffer_size,
                batch_size=cfg.batch_size)

  model = netowrk_fn()

  proxy = proxy_fn(lim_type=cfg.lim_type,
                   ratio=cfg.lim_ratio,
                   min_action=cfg.min_action,
                   lower_bound=True)

  learner = channelPruner(method=cfg.method,
                          model=model,
                          proxy=proxy,
                          pretrain_dir=cfg.pretrain_dir,
                          num_sample_points=cfg.num_sample_points,
                          data_dir='./data')

  checkpoint = {'rewards': [], 'avg_rewards': [], 'accs': [],
                'actions': [], 'real_actions': [], 'scores': []}

  print('search start at %s !' % datetime.now())
  total_start_time = time.perf_counter()
  start_time = time.perf_counter()
  step = 0
  while step < cfg.max_steps:
    # generate an episode and evaluate
    states, actions, real_actions, scores, dones = [], [], [], [], []
    agent.episode_start()
    state, _, done, _ = learner.init_episode()
    while not done:
      action = agent.get_action(torch.tensor(state).float()[None, :])[0]
      next_state, real_action, done, score = learner.compress(action)

      states.append(state)
      actions.append(action[0])
      real_actions.append(real_action[0])
      scores.append(score)
      dones.append(done)

      state = next_state
      c_loss, a_loss, epsilon = agent.train()
    states.append(state)  # fix a hidden bug!

    acc = learner.evaluate()

    reward = acc / 100
    rewards = [reward for _ in actions]

    checkpoint['accs'].append(acc)
    checkpoint['rewards'].append(reward)
    checkpoint['scores'].append(scores)
    checkpoint['actions'].append(actions)
    checkpoint['real_actions'].append(real_actions)
    checkpoint['avg_rewards'].append(np.mean(checkpoint['rewards']) if step < 10 else
                                     0.95 * checkpoint['avg_rewards'][-1] + 0.05 * reward)

    agent.episode_end(reward)
    for transition in zip(states[:-1], real_actions, rewards, states[1:], dones):
      agent.record(transition)

    if step % 1 == 0:
      duration = time.perf_counter() - start_time
      print('step: %d reward: %.3f avg_reward: %.3f, acc:%.3f (%.2f sec/step)' %
            (step, reward, checkpoint['avg_rewards'][-1], acc, duration), end='\t')
      print('actor loss: %.5f critic loss: %.5f epsilon: %.3f' %
            (a_loss, c_loss, epsilon), end='\t')
      print(('actions: [' + '{:.2f}, ' * len(real_actions) + '] %s' % datetime.now()).
            format(*real_actions))
      start_time = time.perf_counter()

    step += 1

  print('search finished at %s' % datetime.now())
  print('total search time: %s' % (time.perf_counter() - total_start_time))
  print('max reward: %.5f' % checkpoint['rewards'][np.argmax(checkpoint['rewards'])])
  print(('max reward actions: [' + '{:.2f}, ' * len(real_actions) + ']').
        format(*checkpoint['real_actions'][np.argmax(checkpoint['rewards'])]))
  with open('./ckpt/' + cfg.log_name + '.pickle', 'wb') as handle:
    pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  main()
