import numpy as np
import math
import copy
import torch


class ReplayMemory:
	def __init__(self, batch_size=32, max_memory_size=750000):
		self.memory = []
		self.batch_size = batch_size
		self.memory_refrence_pointer = 0
		self.MAX_MEMORY_SIZE = max_memory_size
		self.EE_TIME_SEP_CONSTANT_M = 100
		self.pellet_discount = 0.99
		self.ee_beta = 1
		self.maximum_pellet_reward = torch.tensor([[0.1]*100])

	def save(self, episode_buffer):
		full_pellet_reward = 0
		mc_reward = 0
		for i, transition in enumerate(reversed(episode_buffer)):
			post_visit = torch.min(torch.stack([self.maximum_pellet_reward[0].unsqueeze(0), copy.deepcopy(transition[7])], dim=1),dim=1)[0]
			pre_visit = torch.min(torch.stack([self.maximum_pellet_reward[0].unsqueeze(0), copy.deepcopy(transition[2])], dim=1),dim=1)[0]
			pellet_reward = torch.sum(post_visit, 1) - torch.sum(pre_visit, 1)
			full_pellet_reward =  pellet_reward + self.pellet_discount * full_pellet_reward
			mc_reward =  transition[4] + 0.99 * mc_reward
			transition.append(mc_reward + full_pellet_reward)
			transition.append(i)

		for transition in episode_buffer:
			self.memory.append(transition)
			self.memory = self.memory[-self.MAX_MEMORY_SIZE:]

	def sample(self, forced_batch_size=None, should_pop=False):
		batch = []
		if forced_batch_size is not None:
			batch_size = forced_batch_size
		else:
			batch_size = self.batch_size

		for i in range(0, batch_size):
			state_index = np.random.randint(0, (len(self.memory)))
			if should_pop:
				element = self.memory.pop(state_index)
			else:
				element = self.memory[state_index]
			batch.append(copy.deepcopy(element))
		return batch

	def sample_ee_minibatch(self, forced_batch_size=None):
		batch = []
		if forced_batch_size is not None:
			batch_size = forced_batch_size
		else:
			batch_size = self.batch_size

		for i in range(0, batch_size):
			state_index = np.random.randint(0, (len(self.memory)))
			while self.memory[state_index][-1] <= 2:
				state_index = np.random.randint(0, (len(self.memory)))
			offset = np.random.randint(1, self.memory[state_index][-1])
			offset = min(offset, self.EE_TIME_SEP_CONSTANT_M)
			state_prime_index = (state_index + offset) % self.MAX_MEMORY_SIZE

			aux = []
			for j in range(0, offset):
				auxiliary_reward = self.memory[((state_index + j) % self.MAX_MEMORY_SIZE)][3]
				aux.append(auxiliary_reward)
			batch.append([self.memory[state_index][0], self.memory[state_prime_index][0], self.memory[state_index][-4], aux])

		return batch

	def calc_pellet_reward(self, visits):
		return self.ee_beta / math.sqrt(max(1, visits))
