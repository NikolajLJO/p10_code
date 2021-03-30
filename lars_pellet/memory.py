'''
The class includes the replay memory and
functions to query it.
'''


class ReplayMemory():
    def __init__(self, batch_size = 32, max_memory_size = 10000):
        self.memory = []
        self.batch_size = batch_size
        self.memory_refrence_pointer = 0
        self.MAX_MEMORY_SIZE = max_memory_size
    
    def save(self, state, action, reward, terminating, state_prime): 
        if len(memory) < memory_size:
            self.memory.append([state, action, reward, terminating, state_prime])
        else:
            memory[memory_replace_pointer] = transition
            self.memory_replace_pointer = (self.memory_replace_pointer + 1) % memory_size

    def sample(self):
        return = self.memory.sample(self.replay_batch_size)

    def sampleEEminibatch():
        raise NotImplementedError


    else:
        memory[memory_replace_pointer] = transition
        memory_replace_pointer = (memory_replace_pointer + 1) % memory_size
    
    return memory, memory_replace_pointer