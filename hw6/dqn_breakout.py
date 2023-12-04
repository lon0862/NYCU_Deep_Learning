'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari
import warnings
warnings.filterwarnings("ignore")

class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity, device):
        c,h,w = 5,84,84
        self.capacity = capacity
        self.states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.device = device

    def push(self, state, action, reward, done):
        """Saves a transition."""
        state = state.squeeze(0)
        self.states[self.position] = torch.tensor(state) # 5,84,84
        self.actions[self.position,0] = action
        self.rewards[self.position,0] = reward
        self.dones[self.position,0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        i = torch.randint(0, high=self.size, size=(batch_size,))
        state = self.states[i, :4].to(self.device)
        next_state = self.states[i, 1:].to(self.device)
        action = self.actions[i].to(self.device)
        reward = self.rewards[i].to(self.device).float()
        done = self.dones[i].to(self.device).float()

        return state, action, reward, next_state, done

    def __len__(self):
        # return self.size
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        ## TODO ##
        """Initialize replay buffer"""
        self._memory = ReplayMemory(capacity=args.capacity, device=args.device)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##
        if random.random() <= epsilon:
            action = action_space.sample()
        else:
            with torch.no_grad():
                action_values = self._behavior_net(state)
                _, action = torch.max(action_values, 1)
                action = action.item()

        return action

    def append(self, state, action, reward, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        self._memory.push(state, action, reward, int(done))

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size)

        ## TODO ##
        action_values = self._behavior_net(state)
        q_value = torch.gather(action_values, dim=1, index=action.long())
        with torch.no_grad():
            q_next = self._target_net(next_state)
            q_target = reward + (gamma * torch.max(q_next, dim=1)[0].view(-1, 1)) * (1 - done)

        # criterion = nn.MSELoss()
        criterion = F.smooth_l1_loss
        loss = criterion(q_value, q_target)

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()
        
    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

def train(args, agent, writer):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life=True, clip_rewards=True, frame_stack=False)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    q = deque(maxlen=5)
    for i in range(5):
        empty_frame = torch.from_numpy(np.zeros((1, 84, 84)))
        q.append(empty_frame)

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        state, reward, done, _ = env.step(1) # fire first !!!
        n_frame = torch.from_numpy(state)
        q.append(n_frame.squeeze(-1).unsqueeze(0))
        
        for t in itertools.count(start=1):
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                # select action
                state = torch.cat(list(q))[1:].unsqueeze(0).to(args.device)
                action = agent.select_action(state, epsilon, action_space)
                # decay epsilon
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            # execute action
            n_frame, reward, done, _ = env.step(action)

            ## TODO ##
            # store transition
            n_frame = torch.from_numpy(n_frame)
            q.append(n_frame.squeeze(-1).unsqueeze(0))
            agent.append(torch.cat(list(q)).unsqueeze(0), action, reward, done)
            
            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward
            
            if total_steps % args.eval_freq == 0:
                """You can write another evaluate function, or just call the test function."""
                test(args, agent, writer)
                agent.save(args.model + "dqn_breakout_" + str(total_steps) + ".pt")

            total_steps += 1

            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    env.close()


def test(args, agent, writer):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life=False, clip_rewards=False, frame_stack=False)
    action_space = env.action_space
    e_rewards = []
    q = deque(maxlen=5)
    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        done = False
        
        for j in range(10):
            n_frame, _, _, _ = env.step(0)
            n_frame = torch.from_numpy(n_frame)
            q.append(n_frame.squeeze(-1).unsqueeze(0))

        while not done:
            time.sleep(0.01)
            # env.render()
            state = torch.cat(list(q))[1:].unsqueeze(0).to(args.device)
            action = agent.select_action(state, args.test_epsilon, action_space)
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            q.append(state.squeeze(-1).unsqueeze(0))
            e_reward += reward

        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ckpt/dqn_breakout/')
    parser.add_argument('--logdir', default='log/dqn_breakout')
    # train
    parser.add_argument('--warmup', default=20000, type=int) # origin 20000
    parser.add_argument('--episode', default=50000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=1000000, type=float) # origin 1000000
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int) # origin 10000
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='ckpt/dqn_breakout/dqn_breakout_10600000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
    else:
        train(args, agent, writer)

if __name__ == '__main__':
    main()
