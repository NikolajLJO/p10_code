import copy

import tools
from pathlib import Path
import datetime
import logging
import numpy as np
from memory import ReplayMemory
import sys
from init import setup_agent
import traceback
import torch


class Learner:
    def __init__(self, args,
                 learner_replay_que,
                 learner_que_max_size,
                 q_network_que,
                 e_network_que,
                 q_t_network_que,
                 e_t_network_que,
                 learner_ee_que,
                 learner_ee_que_max_size,
                 from_actor_partition_que,
                 to_actor_partition_que,
                 actor_count):
        torch.multiprocessing.set_sharing_strategy('file_system')
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
        self.agent = setup_agent()
        self.partition_memory = []
        self.replay_memory = ReplayMemory()
        self.ee_memory = []
        self. learner_que_max_size = learner_que_max_size
        self.learner_ee_que_max_size = learner_ee_que_max_size
        self.update_memory_break_point = self.learner_que_max_size / 10
        self.update_ee_memory_break_point = self.learner_ee_que_max_size / 10
        try:
            self.learn(learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que, actor_count)
        except Exception as err:
            logging.info(err)
            logging.info(traceback.format_exc())

    def learn(self, learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que,  actor_count):
        logging.info("Started with empty memory")
        while True:
            # when rpelay memory is almost empty, wait until the que has a full memory size
            while learner_replay_que.qsize() < self.learner_que_max_size:
                pass

            # then when it does, update it
            for _ in range(int(self.learner_que_max_size)):
                transition = learner_replay_que.get()
                process_local_transition = copy.deepcopy(transition)
                self.replay_memory.memory.append(process_local_transition)
                del transition

            logging.info("Refilled replay memory")

            while learner_ee_que.qsize() < self.learner_ee_que_max_size:
                pass

            if not learner_ee_que.empty():
                for _ in range(int(self.learner_ee_que_max_size)):
                    transition = learner_ee_que.get()
                    process_local_transition = copy.deepcopy(transition)
                    self.ee_memory.append(process_local_transition)
                    del transition
                logging.info("Refilled ee memory")

            if from_actor_partition_que.qsize() >= actor_count * 2:
                unqued_partitions = []
                for _ in range(actor_count*2):
                    partition = from_actor_partition_que.get()
                    process_local_partition = copy.deepcopy(partition)
                    unqued_partitions.append(process_local_partition)
                    del partition
                best_partition = max(unqued_partitions, key=lambda item: item[1])[0]
                for _ in range(actor_count):
                    to_actor_partition_que.put(copy.deepcopy(best_partition))
                logging.info("Pushed partitions")

            # while we have more than 10% replay memory, learn
            while len(self.replay_memory.memory) >= self.update_memory_break_point \
                and len(self.ee_memory) >= self.update_ee_memory_break_point:
                self.agent.update(self.replay_memory, self.ee_memory)

            logging.info("I processed 90% of que")

            for _ in range(actor_count):
                self.q_network_que.put(copy.deepcopy(self.agent.Qnet.state_dict()))
                self.e_network_que.put(copy.deepcopy(self.agent.EEnet.state_dict()))
                self.q_t_network_que.put(copy.deepcopy(self.agent.targetQnet.state_dict()))
                self.e_t_network_que.put(copy.deepcopy(self.agent.targetEEnet.state_dict()))
            logging.info("Pushed networks")
