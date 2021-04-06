import tools
from pathlib import Path
import datetime
import logging
import numpy as np
from memory import ReplayMemory
import sys
from init import setup_agent
from init import setup
import torch


class Learner:
    def __init__(self, args, replay_que, partition_que, learner_replay_que, learner_que_max_size):
        path = Path(__file__).parent
        Path(path / 'logs').mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            filename=(str(path) + now_but_text + "-learner" + "-log.txt"),
                            filemode='w')
        logger = tools.get_writer()
        sys.stdout = logger

        self.partition_candidate = None
        self.terminating = False
        self.dmax = np.NINF
        self.distance = np.NINF
        self.agent = setup_agent(args[1])
        self.partition_memory = []
        self.replay_memory = ReplayMemory()
        self. learner_que_max_size = learner_que_max_size
        self.replay_que = replay_que
        self.partition_que = partition_que
        self.update_memory_break_point = self.learner_que_max_size / 10

        self.learn(learner_replay_que)

    def learn(self, learner_replay_que):
        logging.info("Started")
        i = 0
        while True:
            i += 1

            # while we have more than 10% replay memory, learn
            while len(self.replay_memory.memory) >= self.update_memory_break_point:
                self.agent.update(self.replay_memory)
                logging.info("learned!")
                if i % 1000 == 0:
                    self.agent.update_targets()
                    i = 0

            # when rpelay memory is almost empty, wait until the que has a full memory size
            while learner_replay_que.qsize() < self.learner_que_max_size:
                pass

            # then when it does, update it
            for _ in range(self.learner_que_max_size):
                logging.info("added trans to memory")
                transition = learner_replay_que.get()
                self.replay_memory.memory.append(transition)

            logging.info("i updated my memory que")
