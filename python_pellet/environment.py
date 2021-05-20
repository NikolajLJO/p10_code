from __future__ import division
import gym
import numpy as np
import torch
import torchvision.transforms as transforms
from gym.spaces.box import Box

resize = transforms.Compose([transforms.ToPILImage(),transforms.Resize((84, 84)),transforms.Grayscale(num_output_channels=1)])


def create_atari_env(env_id, device):

	env = gym.make(env_id)
	env = EpisodicLifeEnv(env)
	if 'FIRE' in env.unwrapped.get_action_meanings():
		env = FireResetEnv(env)
	env = AtariRescale(env, device)
	return env


def process_frame(frame, device):
	frame = frame[34:34 + 160, :160]
	frame = np.array(resize(frame), dtype=np.uint8)
	frame = torch.tensor(frame, dtype=torch.uint8, device=device).unsqueeze(0)
	return frame


class AtariRescale(gym.ObservationWrapper):
	def __init__(self, env, device):
		gym.ObservationWrapper.__init__(self, env)
		self.observation_space = Box(0.0, 1.0, [1, 1, 84, 84])
		self.device = device

	def observation(self, observation):
		return process_frame(observation, self.device).unsqueeze(0)


class NormalizedEnv(gym.ObservationWrapper):
	def __init__(self, env=None):
		gym.ObservationWrapper.__init__(self, env)
		self.state_mean = 0
		self.state_std = 0
		self.alpha = 0.9999
		self.num_steps = 0

	def observation(self, observation):
		self.num_steps += 1
		self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
		self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

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
		self.lives = 0
		self.was_real_done = True

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done
		# check current lives, make loss of life terminal,
		# then update lives to handle bonus lives
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives and lives > 0:
			# for Qbert sometimes we stay in lives == 0 condtion for a few frames
			# so its important to keep lives > 0, so that we only reset once
			# the environment advertises done.
			done = True
		self.lives = lives
		return obs, reward, done, self.was_real_done

	def reset(self, **kwargs):
		"""Reset only when lives are exhausted.
		This way all states are still reachable even though lives are episodic,
		and the learner need not know about any of this behind-the-scenes.
		"""
		if self.was_real_done:
			obs = self.env.reset(**kwargs)
		else:
			# no-op step to advance from terminal/lost life state
			obs, _, _, _ = self.env.step(0)
		self.lives = self.env.unwrapped.ale.lives()
		return obs

