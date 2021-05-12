'''
The class includes the replay memory and
functions to query it.
'''
import math
import torch
import numpy as np

class ReplayMemory:
    def __init__(self, batch_size=32, max_memory_size=70000):
        self.memory = []
        self.batch_size = batch_size
        self.memory_refrence_pointer = 0
        self.MAX_MEMORY_SIZE = max_memory_size
        self.EE_TIME_SEP_CONSTANT_M = 100
        self.pellet_discount = 0.99
        self.ee_beta = 1

    def save(self, episode_buffer):
        '''
        This functions saves a buffers content to memory while adding the mc reward and the count to a terminating state
        Input: episode_bffer a list of transaction
        '''
        full_pellet_reward = 0
        mc_reward = 0
        for i, transition in enumerate(reversed(episode_buffer)):
            pellet_reward = torch.sum(transition[7], 1) - torch.sum(transition[2], 1)
            full_pellet_reward =  pellet_reward + self.pellet_discount * full_pellet_reward
            mc_reward =  transition[4].item() + 0.99 * mc_reward
            transition.append(full_pellet_reward + mc_reward)
            transition.append(i)

        for transition in episode_buffer:
            if len(self.memory) < self.MAX_MEMORY_SIZE:
                self.memory.append(transition)
            else:
                self.memory[self.memory_refrence_pointer] = transition
            self.memory_refrence_pointer = (self.memory_refrence_pointer + 1) % self.MAX_MEMORY_SIZE

    def sample(self):
        '''
        This function samples a batch used for training
        Output: batch a list of transaction
        '''
        # TODO Document what a batch returned from this method contains.
        batch = []
        for i in range(self.batch_size):
            state_index = np.random.randint(0, (len(self.memory)))
            
            # self.memory[state_index][0] = state at state_index
            # self.memory[state_index][1] = the action done at state
            # self.memory[state_index][2] = list of visited partitions before the transition
            # self.memory[state_index][4] = reward for taking the action at state
            # self.memory[state_index][5] = boolean if the transition leads to termination
            # self.memory[state_index][6] = state_prime the state following the state and action pair
            # self.memory[state_index][7] = list of visited partitions after the transition
            batch.append([self.memory[state_index][0], self.memory[state_index][1],
                          self.memory[state_index][2], self.memory[state_index][4],
                          self.memory[state_index][5], self.memory[state_index][6],
                          self.memory[state_index][7], self.memory[state_index][8]])

        return batch

    def sample_ee_minibatch(self):
        '''
        This function samples a batch used for training
        Output: batch a list of transaction
        '''
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
            # self.memory[state_index][0] = state at state_index
            # self.memory[state_prime_index][0] = the state that we want to find the distance to
            # self.memory[state_index][6]
            # aux the auxilary reward form state to state prime in a list
            batch.append([self.memory[state_index][0],
                          self.memory[state_prime_index][0],
                          self.memory[state_index][6],
                          aux])

        return batch

    def calc_pellet_reward(self, visits):
        '''
        the function that calculatets the pellet reward
        Input: number of times a partition has been visited
        '''
        return self.ee_beta / math.sqrt(max(1, visits))
