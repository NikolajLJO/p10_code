'''
The class includes the replay memory and
functions to query it.
'''


class ReplayMemory:
    def __init__(self, batch_size=32, max_memory_size=10000):
        self.memory = []
        self.batch_size = batch_size
        self.memory_refrence_pointer = 0
        self.memory_size = max_memory_size

    def save(self, state, action, visited, reward, terminating, state_prime, visited_prime):
        if len(self.memory) < self.memory_size:
            self.memory.append([state, action, reward, terminating, state_prime])
        else:
            self.memory[self.memory_refrence_pointer] = [state, action, reward, terminating, state_prime]
            self.memory_refrence_pointer = (self.memory_refrence_pointer + 1) % self.memory_size

    def sample(self):
        return self.memory.sample(self.replay_batch_size)

    def sampleEEminibatch(self):
        raise NotImplementedError
