import tools
from pathlib import Path
import datetime
import logging
import sys

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

        while True:
            if not replay_que.empty():

                optional_replay = replay_que.get()
                self.replay_memory.save(optional_replay)

            if not partition_que.empty():

                optional_partition = partition_que.get()
                self.partition_memory.append(optional_partition)

            if not learner_replay_que.full() and len(self.replay_memory.memory) > learner_que_max_size:

                self.fill_learner_replay_que(learner_replay_que, learner_que_max_size)

    def fill_learner_replay_que(self, learner_replay_que, learner_que_max_size):
        batch = self.replay_memory.sample(forced_batch_size=learner_que_max_size)
        for item in batch:
            learner_replay_que.put(item)
