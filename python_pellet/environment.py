from __future__ import division
import gym
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import deque
from gym.spaces.box import Box
import gym.wrappers.frame_stack as frame_stacking
# from skimage.color import rgb2gray
# from cv2 import resize
# from skimage.transform import resize
# from scipy.misc import imresize as resize
import random

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize((84, 84)),
                             transforms.Grayscale(num_output_channels=1)])

def create_atari_env(env_id):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    env = gym.make(env_id)
    env = frame_stacking.FrameStack(env, 4)
    env = EpisodicLifeEnv(env)
    '''if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)'''
    env = AtariRescale(env, device)
    return env


def process_frame(frame, device):
    preprocess = []
    for __frame in frame:
        resize_state = np.array(resize(__frame[34:34 + 160, :160]))
        resize_state = torch.tensor(resize_state, dtype=torch.uint8, device=device).unsqueeze(0)
        preprocess.append(resize_state)
    frame = torch.cat(preprocess).unsqueeze(0).to(device)

    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, device):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 1, 84, 84])
        self.device = device

    def observation(self, observation):
        return process_frame(observation, self.device)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return torch.from_numpy((observation - unbiased_mean) / (unbiased_std + 1e-8)).float().unsqueeze(0)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = self.env.ale.lives()
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        soft_terminating_state = done
        if info['ale.lives'] < self.lives:
            #reward -= 1
            self.lives = info['ale.lives']
            soft_terminating_state = True
        return obs, reward, soft_terminating_state, done

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        obs = self.env.reset(**kwargs)

        self.lives = self.env.ale.lives()
        return obs

