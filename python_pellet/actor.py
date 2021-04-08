import tools
from pathlib import Path
import datetime
import logging
import numpy as np
import sys
from init import setup
import torch
import copy
import time






class Actor:
    def __init__(self, args, process_itterator, replay_que, partition_que, q_network_que, e_network_que, q_t_network_que, e_t_network_que):
        self.e_t_network_que = e_t_network_que
        self.q_t_network_que = q_t_network_que
        self.e_network_que = e_network_que
        self.q_network_que = q_network_que
        self.partition_que = partition_que
        path = Path(__file__).parent
        Path(path / 'logs').mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            filename=(str(path) + now_but_text + "-actor[" + str(process_itterator) + "]" + "-log.txt"),
                            filemode='w')
        logger = tools.get_writer()
        sys.stdout = logger

        partition_candidate = None
        terminating = False
        total_score = 0
        dmax = np.NINF
        distance = np.NINF
        episode_buffer = []

        game_actions, self.agent, opt, env = setup(args[1])

        state = env.reset()
        local_partition_memory = [[state, 0]]

        for i in range(1, int(args[2])):
            start = time.process_time()
            action, policy = self.agent.find_action(state, i)

            auxiliary_reward = self.calculate_auxiliary_reward(policy, action)

            state_prime, reward, terminating, info = env.step(action)
            total_score += reward
            reward = int(max(min(reward, 1), -1))
            visited, visited_prime, distance = self.agent.find_current_partition(state_prime, local_partition_memory)
            episode_buffer.append(
                [state, action, visited, auxiliary_reward, reward, terminating, state_prime, visited_prime])
            if terminating:
                replay_que.put(copy.deepcopy(episode_buffer))
                end = time.process_time()
                elapsed = (end - start)
                state_prime = env.reset()
                self.update_partitions(self.agent.visited, local_partition_memory)
                self.agent.visited = []
                episode_buffer.clear()
                logging.info("step: " + str(i) + " total_score: " + str(total_score) + " Time: " + str(elapsed))
                total_score = 0

            if distance > dmax:
                partition_candidate = state_prime
                partition_que.put(partition_candidate)
                dmax = distance

            if i % int(args[4]) == 0 and partition_candidate is not None:
                local_partition_memory.append([partition_candidate, 0])
                dmax = 0

            state = state_prime
            self.check_ques_for_updates()
            # TODO update partition memory

    @staticmethod
    def calculate_auxiliary_reward(policy, aidx):
        aux = [0] * (policy.size()[1])
        policy = policy.squeeze(0)
        for i in range(len(aux)):
            if aidx == i:
                aux[i] = 1 - policy[i].item()
            else:
                aux[i] = -policy[i].item()
        return aux

    @staticmethod
    def partitiondeterminarion(ee_network, s_n, r):
        mindist = [np.inf, None]
        for s_pi in r:
            dist = ee_network.distance(ee_network, s_n, s_pi[0], r[0][0])
            if mindist[0] > dist:
                mindist[0] = dist
                mindist[1] = s_pi
        return mindist[1]

    @staticmethod
    def update_partitions(visited_partitions, partition_memory):
        for visited in visited_partitions:
            for i, partition in enumerate(partition_memory):
                if torch.equal(visited[0], partition[0]):
                    partition_memory[i][1] += 1
                    break

    def check_ques_for_updates(self):
        if not self.q_network_que.empty():
            parameters = self.q_network_que.get()
            for name, single_param in self.agent.Qnet.state_dict().items():
                single_param = parameters[name]
                self.agent.Qnet.state_dict()[name].copy_(single_param)
                logging.info("updated q network")
        if not self.q_t_network_que:
            parameters = self.q_t_network_que.get()
            for name, single_param in self.agent.targetQnet.state_dict().items():
                single_param = parameters[name]
                self.agent.targetQnet.state_dict()[name].copy_(single_param)
                logging.info("updated q_t network")
        if not self.e_network_que:
            parameters = self.e_network_que.get()
            for name, single_param in self.agent.EEnet.state_dict().items():
                single_param = parameters[name]
                self.agent.EEnet.state_dict()[name].copy_(single_param)
                logging.info("updated e network")
        if not self.e_t_network_que:
            parameters = self.e_t_network_que.get()
            for name, single_param in self.agent.targetEEnet.state_dict().items():
                single_param = parameters[name]
                self.agent.targetEEnet.state_dict()[name].copy_(single_param)
                logging.info("updated e_t network")