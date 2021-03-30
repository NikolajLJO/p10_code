'''
The class includes the replay memory and
functions to query it.
'''


class ReplayMemory():
    def __init__(self, batch_size = 32):
        self.memory = []
        self.batch_size = batch_size
    
    def save(self, state, action, reward, terminating, state_prime):
        self.memory.append([state, action, reward, terminating, state_prime])

    def sample(self):
        return = self.memory.sample(self.replay_batch_size)

    def sampleEEminibatch():
        raise NotImplementedError

