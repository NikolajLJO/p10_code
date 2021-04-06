from memory import ReplayMemory


class MemoryManager:
    def __init__(self, replay_que, partition_que, learner_replay_que, learner_que_max_size):
        self.replay_memory = ReplayMemory()
        self.partition_memory = []
        self.replay_que = replay_que
        self.partition_que = partition_que
        self.manage(learner_replay_que, learner_que_max_size)

    def manage(self, learner_replay_que, learner_que_max_size):
        while True:
            optional_replay = self.replay_que.get()
            if optional_replay is not None:
                self.replay_memory.save(optional_replay)
            optional_partition = self.partition_que.get()
            if optional_partition is not None:
                self.partition_memory.append(optional_partition)

            if not learner_replay_que.full():
                self.fill_learner_replay_que(learner_replay_que, learner_que_max_size)

    def fill_learner_replay_que(self, learner_replay_que, learner_que_max_size):
        batch = self.replay_memory.sample(forced_batch_size=learner_que_max_size)
        learner_replay_que.put(batch)
