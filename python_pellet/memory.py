import numpy as np
import math
import copy


class ReplayMemory:
	def __init__(self, batch_size=32, max_memory_size=1000000):
		self.memory = []
		self.batch_size = batch_size
		self.memory_refrence_pointer = 0
		self.MAX_MEMORY_SIZE = max_memory_size
		self.EE_TIME_SEP_CONSTANT_M = 100
		self.pellet_discount = 0.99
		self.ee_beta = 1

	def save(self, episode_buffer):
		y = 0
		for i, transition in enumerate(episode_buffer):
			state_index = i
			mc_reward = 0
			terminating = episode_buffer[state_index][5]
			t = (len(episode_buffer) - 1)
			mc_reward = episode_buffer[t][4]
			j = 0
			while not terminating:
				if len(episode_buffer[2]) < len(episode_buffer[7]):
					pellet_reward = self.calc_pellet_reward(episode_buffer[7][-1][1])
					mc_reward = mc_reward + pellet_reward * (self.pellet_discount ** j)
				state_index += 1
				j += 1
				terminating = episode_buffer[state_index][5]

			transition.append(mc_reward)
			transition.append(len(episode_buffer)-y)
			y += 1

			if len(self.memory) < self.MAX_MEMORY_SIZE:
				self.memory.append(transition)
			else:
				self.memory[self.memory_refrence_pointer] = transition
			self.memory_refrence_pointer = (self.memory_refrence_pointer + 1) % self.MAX_MEMORY_SIZE

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
