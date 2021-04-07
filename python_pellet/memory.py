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
            state_index = i
            mc_reward = 0
            terminating = episode_buffer[state_index][5]
            try:
                t = (len(episode_buffer) - 1)
                mc_reward = episode_buffer[t][4]
            except IndexError:
                logging.info("state_index: " + str(state_index) + " mem: " + str(len(episode_buffer)) + " total:" + str(t))
            j = 0
            while not terminating:
                if len(episode_buffer[2]) < len(episode_buffer[7]):
                    pellet_reward = self.calc_pellet_reward(episode_buffer[7][-1][1])
                    mc_reward = mc_reward + pellet_reward * (self.pellet_discount ** j)
                state_index += 1
                j += 1
                terminating = episode_buffer[state_index][5]

            transition.append(mc_reward)

            if len(self.memory) < self.MAX_MEMORY_SIZE:
                self.memory.append(transition)
            else:
                self.memory[self.memory_refrence_pointer] = transition
            self.memory_refrence_pointer = (self.memory_refrence_pointer + 1) % self.MAX_MEMORY_SIZE

    def sample(self, forced_batch_size=None):
        batch = []
        if forced_batch_size is not None:
            batch_size = forced_batch_size
        else:
            batch_size = self.batch_size

        for i in range(batch_size):
            state_index = np.random.randint(0, (len(self.memory)))

            element = self.memory.pop(state_index)
            logging.info("element pop len: " + str(len(element)))
            try:
                logging.info(state_index)
                batch.append([element[0], element[1],
                          element[2], element[4],
                          element[5], element[6],
                          element[7], element[8]])  # TODO ASK LARS HERE WHY NOT ALL ELEMENTS
            except IndexError:
                logging.info("index error statee_index: " + str(state_index))
                logging.info("mem size: " + str(len(self.memory)))
                logging.info("Is Term: " + str(element[5]))
                exit()

        return batch

    def sample_ee_minibatch(self):
        batch = []
        for i in range(self.batch_size):
            state_index = np.random.randint(0, (len(self.memory)))
            while self.memory[state_index][-1] <= 2:
                state_index = np.random.randint(0, (len(self.memory)))
            offset = np.random.randint(1, self.memory[state_index][-1])
            offset = min(offset, self.EE_TIME_SEP_CONSTANT_M)
            
            state_prime_index = (state_index + offset) % self.MAX_MEMORY_SIZE

            aux = []
            for j in range(offset):
                auxiliary_reward = self.memory[(state_index + j) % self.MAX_MEMORY_SIZE][3]
                aux.append(auxiliary_reward)
            batch.append([self.memory[state_index][0],
                          self.memory[state_prime_index][0],
                          self.memory[state_index][-3],
                          aux])

        return batch

    def calc_pellet_reward(self, visits):
            return self.ee_beta / math.sqrt(max(1, visits))
