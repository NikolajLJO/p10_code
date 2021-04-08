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
    def __init__(self, args, learner_replay_que, learner_que_max_size, q_network_que, e_network_que, q_t_network_que, e_t_network_que):

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

        self.e_t_network_que = e_t_network_que
        self.q_t_network_que = q_t_network_que
        self.e_network_que = e_network_que
        self.q_network_que = q_network_que
        self.partition_candidate = None
        self.terminating = False
        self.dmax = np.NINF
        self.distance = np.NINF
        self.agent = setup_agent(args[1])
        self.partition_memory = []
        self.replay_memory = ReplayMemory()
        self. learner_que_max_size = learner_que_max_size
        self.update_memory_break_point = self.learner_que_max_size / 10

        self.learn(learner_replay_que)

    def learn(self, learner_replay_que):
        i = 0
        while True:
            logging.info("Started with empty memory")

            # when rpelay memory is almost empty, wait until the que has a full memory size
            while learner_replay_que.qsize() < int(self.learner_que_max_size * 0.9):  # TODO remove this from testing using less than full mem due to uncertainty in qsize
                pass

            # then when it does, update it
            for _ in range(int(self.learner_que_max_size * 0.75)):  # TODO remove this from testing using less than full mem due to uncertainty in qsize
                transition = learner_replay_que.get()
                self.replay_memory.memory.append(transition)

            logging.info("Refilled memory")

            # while we have more than 10% replay memory, learn
            while len(self.replay_memory.memory) >= self.update_memory_break_point:
                self.agent.update(self.replay_memory)
                logging.info("learned! ")

            logging.info("I processed 90% of que")

            self.q_network_que.put(self.agent.Qnet.state_dict())
            self.e_network_que.put(self.agent.Qnet.state_dict())
            self.q_t_network_que.put(self.agent.Qnet.state_dict())
            self.e_t_network_que.put(self.agent.Qnet.state_dict())
            logging.info("Pushed networks")




            i += 1
