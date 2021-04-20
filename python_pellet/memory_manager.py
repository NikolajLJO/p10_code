import tools
from pathlib import Path
import datetime
import logging
import sys
import copy

from memory import ReplayMemory


class MemoryManager:
    def __init__(self, replay_que, learner_replay_que, learner_que_max_size, learner_ee_que, learner_ee_que_max_size):
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
        self.manage(learner_replay_que, learner_que_max_size, replay_que, learner_ee_que, learner_ee_que_max_size)

    def manage(self, learner_replay_que, learner_que_max_size, replay_que, learner_ee_que, learner_ee_que_max_size):
        while True:
            while not replay_que.empty() and len(self.replay_memory.memory) < learner_ee_que_max_size + learner_que_max_size:
                optional_replay = replay_que.get()
                process_local_optional_replay = copy.deepcopy(optional_replay)
                self.replay_memory.save(process_local_optional_replay)
                del optional_replay

            if (not learner_replay_que.full()) and len(self.replay_memory.memory) > learner_que_max_size:
                self.fill_learner_replay_que(learner_replay_que, learner_que_max_size)
                logging.info("refilled learner replay mem @ |" + str(datetime.datetime.now()) + "| with |" + str(learner_que_max_size) + "| elements")

            if (not learner_ee_que.full()) and len(self.replay_memory.memory) > learner_ee_que_max_size:
                self.fill_learner_ee_que(learner_ee_que, learner_ee_que_max_size)
                logging.info("refilled learner ee mem @ |" + str(datetime.datetime.now()) + "| with |" + str(learner_ee_que_max_size) + "| elements")

    def fill_learner_replay_que(self, learner_replay_que, learner_que_max_size):
        batch = self.replay_memory.sample(forced_batch_size=learner_que_max_size)
        for item in batch:
            learner_replay_que.put(copy.deepcopy(item))

    def fill_learner_ee_que(self, learner_ee_que, learner_ee_que_max_size):
        batch = self.replay_memory.sample_ee_minibatch(forced_batch_size=learner_ee_que_max_size)
        for item in batch:
            learner_ee_que.put(copy.deepcopy(item))
