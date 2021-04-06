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
        self.learner_replay_que = learner_replay_que
        self.replay_que = replay_que
        self.partition_que = partition_que
        self.update_memory_break_point = self.learner_que_max_size / 10

        self.learn()

    def learn(self):
        i = 0
        while True:
            i += 1

            # if replay memory que
            if self.learner_que_max_size > self.learner_replay_que.qsize():
                pass
            else:
                for _ in range(self.learner_que_max_size):
                    transition = self.learner_replay_que.get()
                    self.replay_memory.memory.append(transition)


            while len(self.replay_memory.memory) >= self.update_memory_break_point:


            if i % 1000 == 0:
                agent.update_targets()
                # TODO update partition memory
                i = 0


