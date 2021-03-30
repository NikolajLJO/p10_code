'''
The class includes the replay memory and
functions to query it.
'''
import random

class ReplayMemory:
    def __init__(self, batch_size=32, max_memory_size=10000):
        self.memory = []
        self.batch_size = batch_size
        self.memory_refrence_pointer = 0
        self.MAX_MEMORY_SIZE = max_memory_size
        self.EE_TIME_SEP_CONSTANT_M = 100
    
    def save(self, episode_buffer):
        for i, transition in enumerate(episode_buffer):
            transition.append(len(episode_buffer)-i)
            if len(self.memory) < self.MAX_MEMORY_SIZE:
                self.memory.append(transition)
            else:
                self.memory[self.memory_refrence_pointer] = transition
            self.memory_refrence_pointer = (self.memory_refrence_pointer + 1) %  self.MAX_MEMORY_SIZE

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def sampleEEminibatch(self):
        batch = []
        for i in range(self.batch_size):
            state_index = np.random.randint(0, (len(self.memory)-1))
            offset = np.random.randint(1, self.memory[state_index][-1])
            offset = min(offset, self.EE_TIME_SEP_CONSTANT_M)
            
            state_prime_index = state_index + offset % self.MAX_MEMORY_SIZE

            aux = []
            for i in range(offset):
                auxiliary_reward = self.memory[index + i % self.MAX_MEMORY_SIZE][3]
                aux.append(auxiliary_reward)

            batch.append([self.memory[state_index][0],self.memory[state_prime_index][0], self.memory[state_index][-3], aux])

        return batch
            
def sampleEEminibatch(memory, batch_size, memory_replace_pointer):
    resbatch = []
    batch = sample(list(enumerate(memory)), batch_size)

    # this loop takes the elements in the batch and goes k elements forward to give the auxilaray calculations on what actions has been used
    # between state s_0 and state s_0+k
    for element in batch:
        if len(memory) >= element[0] + k and (memory_replace_pointer < element[0] or memory_replace_pointer >= element[0] + k):
            auxs = [element[1][3]]
            for i in range(1, k):
                aux = memory[element[0] + i][3]
                auxs.append(aux)
            resbatch.append([element[1][0], memory[element[0] + i][0], memory[element[0]+1][0], auxs])

    return resbatch
        
