import copy
import itertools
import queue
import time

import tools
from pathlib import Path
import datetime
import logging
import numpy as np
from memory import ReplayMemory
import sys
from init import setup
import traceback
import torch
import torchvision.transforms as transforms

transform_to_image = transforms.ToPILImage()


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
                 actor_count,
                 should_use_rnd):

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)

        self.path = Path(__file__).parent
        Path(self.path / 'logs').mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        self.now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)-15s | %(message)s',
                            filename=(str(self.path) + self.now_but_text + "-learner" + "-log.txt"),
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
        self.agent = setup(args[0], should_use_rnd)[1]
        self.partition = 0
        self.replay_memory = ReplayMemory(max_memory_size=learner_que_max_size)
        self.ee_memory = []
        self.learner_que_max_size = learner_que_max_size
        self.learner_ee_que_max_size = learner_ee_que_max_size
        try:
            self.learn(learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que,
                       actor_count, should_use_rnd)
        except Exception as err:
            logging.info(err)
            logging.info(traceback.format_exc())

    def learn(self, learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que, actor_count,
              should_use_rnd):
        logging.info("Started with empty memory")
        learn_count = 0
        ee_update_count = 0
        ee_done = False
        i = 0
        while True:
            try:
                if (
                        not learner_replay_que.qsize() == self.learner_que_max_size
                        or
                        not learner_replay_que.qsize() > self.learner_que_max_size * 0.99) \
                    and (
                        not learner_ee_que.qsize() == self.learner_ee_que_max_size
                        or
                        not learner_ee_que.qsize() > self.learner_ee_que_max_size * 0.99
                ):
                    time.sleep(0.001)

                else:
                    l = 0
                    while l < self.learner_que_max_size:
                        try:
                            transition = learner_replay_que.get(False)
                            process_local_transition = copy.deepcopy(transition)
                            self.replay_memory.memory.append(process_local_transition)
                            del transition
                            l += 1
                        except queue.Empty:
                            pass
                    logging.info("Refilled r :" + str(len(self.replay_memory.memory)))
                    l = 0
                    while l < self.learner_ee_que_max_size:
                        try:
                            transition = learner_ee_que.get(False)
                            process_local_transition = copy.deepcopy(transition)
                            self.ee_memory.append(process_local_transition)
                            del transition
                            l += 1
                        except queue.Empty:
                            torch.cuda.empty_cache()
                            pass
                    logging.info("Refilled e :" + str(len(self.ee_memory)))
                    ee_update_count += 1

                    if not ee_done and ee_update_count * self.learner_ee_que_max_size > 2e6:
                        ee_done = True

                    if from_actor_partition_que.qsize() >= actor_count * 2:
                        unqued_partitions = []
                        for _ in range(0, actor_count * 2):
                            try:
                                partition = from_actor_partition_que.get(False)
                                process_local_partition = copy.deepcopy(partition)
                                unqued_partitions.append(process_local_partition)
                                del partition
                            except queue.Empty:
                                pass
                        best_partition = max(unqued_partitions, key=lambda item: item[1])
                        path = (self.path / (self.now_but_text + "patition_" + str(self.partition) + ".png")).__str__()
                        self.partition += 1
                        try:
                            transform_to_image(best_partition[0][0][0]).save(path)
                        except:
                            logging.info("partition not saved as png")
                        for _ in range(actor_count):
                            try:
                                to_actor_partition_que.put(copy.deepcopy(best_partition),False)
                            except queue.Full:
                                logging.info("Error full partition queue")
                                pass
                        unqued_partitions.clear()

                        try:
                            while True:
                                from_actor_partition_que.get(False)
                        except queue.Empty:
                            pass
                        logging.info("Pushed partition: " + str(self.partition))

                    # while we have more than 10% replay memory, learn
                    # ToDO this should prob just do entire que its a frakensetein of old concepts
                    while self.replay_memory.memory and self.ee_memory:
                        self.agent.update(self.replay_memory, self.ee_memory, ee_done, should_use_rnd=should_use_rnd)
                    i += 1
                    if i % 10 == 0:
                        self.agent.targetQnet = copy.deepcopy(self.agent.Qnet)
                        if not should_use_rnd:
                            self.agent.targetEEnet = copy.deepcopy(self.agent.EEnet)

                    learn_count += 1

                    logging.info("I processed que: " + str(learn_count))

                    c1 = self.agent.Qnet.state_dict()
                    for key in c1.keys():
                        c1[key] = c1[key].to("cpu")
                    c2 = self.agent.EEnet.state_dict()
                    for key in c2.keys():
                        c2[key] = c2[key].to("cpu")
                    c3 = self.agent.targetQnet.state_dict()
                    for key in c3.keys():
                        c3[key] = c3[key].to("cpu")
                    c4 = self.agent.targetEEnet.state_dict()
                    for key in c4.keys():
                        c4[key] = c4[key].to("cpu")

                    self.clear(self.q_network_que)
                    self.clear(self.e_network_que)
                    self.clear(self.q_t_network_que)
                    self.clear(self.e_t_network_que)
                    for _ in range(actor_count):
                        try:
                            self.q_network_que.put(copy.deepcopy(c1), False)
                        except queue.Full:
                            logging.info("Error full qnet queue")
                            pass
                        try:
                            self.e_network_que.put(copy.deepcopy(c2), False)
                        except queue.Full:
                            logging.info("Error full enet queue")
                            pass
                        try:
                            self.q_t_network_que.put(copy.deepcopy(c3), False)
                        except queue.Full:
                            logging.info("Error full qtnet queue")
                            pass
                        try:
                            self.e_t_network_que.put(copy.deepcopy(c4), False)
                        except queue.Full:
                            logging.info("Error full etnet queue")
                            pass

                    logging.info("Pushed networks")
                    
                    del c1
                    del c2
                    del c3
                    del c4
                    self.ee_memory = []
                    self.replay_memory.memory = []
            except Exception as err:
                if not "shared" in err.args[0]:
                    raise
                pass
    
    def clear(self, q):
        try:
            while True:
                q.get(False)
        except queue.Empty:
            pass
