
import tools
from pathlib import Path
import datetime
import logging
import numpy as np
import sys
import math


class ReplayMemory:
    def __init__(self, batch_size=32, max_memory_size=100000):
        self.memory = []
        self.batch_size = batch_size
        self.memory_refrence_pointer = 0
        self.MAX_MEMORY_SIZE = max_memory_size
        self.EE_TIME_SEP_CONSTANT_M = 100
        self.pellet_discount = 0.99
        self.ee_beta = 1

    def save(self, episode_buffer):
        for i, transition in enumerate(episode_buffer):
            transition.append(len(episode_buffer)-i)
            if len(self.memory) < self.MAX_MEMORY_SIZE:
                self.memory.append(transition)
            else:
                self.memory[self.memory_refrence_pointer] = transition
            self.memory_refrence_pointer = (self.memory_refrence_pointer + 1) % self.MAX_MEMORY_SIZE

    def sample(self, forced_batch_size=None):
        if forced_batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = forced_batch_size
        batch = []
        for i in range(batch_size):
            state_index = np.random.randint(0, (len(self.memory)))

            index = state_index
            terminating = self.memory[state_index][5]
            mc_reward = self.memory[(state_index + self.memory[state_index][8]-1) % self.MAX_MEMORY_SIZE][4]
            j = 0
            while not terminating:
                transition = self.memory[index]
                if len(transition[2]) < len(transition[7]):
                    pellet_reward = self.calc_pellet_reward(transition[7][-1][1])
                    mc_reward = mc_reward + pellet_reward * (self.pellet_discount ** j)
                index += 1
                j += 1
                terminating = self.memory[index][5]
            batch.append([self.memory[state_index][0], self.memory[state_index][1],
                          self.memory[state_index][2], self.memory[state_index][4],
                          self.memory[state_index][5], self.memory[state_index][6],
                          self.memory[state_index][7], mc_reward])

        return batch

    def sample_ee_minibatch(self):
        batch = []
        for i in range(self.batch_size):
            state_index = np.random.randint(0, (len(self.memory)))
            if len(self.memory[state_index]) == 9:
                self.memory[state_index].pop()
            while self.memory[state_index][-1] <= 2:
                state_index = np.random.randint(0, (len(self.memory)))
            offset = np.random.randint(1, self.memory[state_index][-1])
            offset = min(offset, self.EE_TIME_SEP_CONSTANT_M)

            state_prime_index = (state_index + offset) % self.MAX_MEMORY_SIZE

            aux = []
            for j in range(offset):
                auxiliary_reward = self.memory[(state_index + j) % self.MAX_MEMORY_SIZE][3]
                aux.append(auxiliary_reward)
            batch.append(
                [self.memory[state_index][0], self.memory[state_prime_index][0], self.memory[state_index][-3], aux])

        return batch

    def calc_pellet_reward(self, visits):
            return self.ee_beta / math.sqrt(max(1, visits))
