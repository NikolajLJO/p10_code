import tools
from pathlib import Path
import datetime
import logging
import sys
import copy

from memory import ReplayMemory


class MemoryManager:
    def __init__(self, replay_que, partition_que, learner_replay_que, learner_que_max_size):
        path = Path(__file__).parent
        Path(path / 'logs').mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            filename=(str(path) + now_but_text + "-manager" + "-log.txt"),
                            filemode='w')
        logger = tools.get_writer()
        sys.stdout = logger
        self.replay_memory = ReplayMemory()
        self.partition_memory = []
        self.manage(learner_replay_que, learner_que_max_size, replay_que, partition_que)

    def manage(self, learner_replay_que, learner_que_max_size, replay_que, partition_que):
        i = 0
        j = 0
        while True:
            if not replay_que.empty():
                i += 1
                optional_replay = replay_que.get()
                self.replay_memory.save(optional_replay)
                if i % 100 == 0:
                    logging.info("I've saved 100 elements to manager replay memory")  # TODO set back to 1000
                    for e in self.replay_memory.memory:
                        logging.info(len(e))
            if not partition_que.empty():
                j += 1
                optional_partition = partition_que.get()
                self.partition_memory.append(optional_partition)
                if j % 1000 == 0:
                    logging.info("I've saved 1000 elements to manager partition memory")
            if (not learner_replay_que.full()) and len(self.replay_memory.memory) > learner_que_max_size:
                logging.info("refilled learner replay mem @ |" + str(datetime.datetime.now()) + "| with |" + str(learner_que_max_size) + "| elements")
                self.fill_learner_replay_que(learner_replay_que, learner_que_max_size)

    def fill_learner_replay_que(self, learner_replay_que, learner_que_max_size):
        batch = self.replay_memory.sample(forced_batch_size=learner_que_max_size)
        for item in batch:
            learner_replay_que.put(copy.deepcopy(item))
