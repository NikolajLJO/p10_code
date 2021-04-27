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
import traceback


class Actor:
    def __init__(self,
                 args,
                 process_itterator,
                 replay_que,
                 q_network_que,
                 e_network_que,
                 q_t_network_que,
                 e_t_network_que,
                 from_actor_partition_que,
                 to_actor_partition_que):
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.to_actor_partition_que = to_actor_partition_que
        self.e_t_network_que = e_t_network_que
        self.q_t_network_que = q_t_network_que
        self.e_network_que = e_network_que
        self.q_network_que = q_network_que
        self.local_partition_memory = []
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
        visited = []
        visited_prime = []
        state = env.reset()
        self.local_partition_memory.append([state, 0])
        try:
            for i in range(1, int(args[2])):
                start = time.process_time()
                action, policy = self.agent.find_action(state, i)

                auxiliary_reward = torch.tensor(self.calculate_auxiliary_reward(policy, action.item()),
                                                device=self.agent.device)

                state_prime, reward, terminating, info = env.step(action)
                total_score += reward
                reward = int(max(min(reward, 1), -1))
                if i % 10 == 0:
                    visited, visited_prime, distance = self.agent.find_current_partition(state_prime,
                                                                                         self.local_partition_memory)
                episode_buffer.append([state, action, visited, auxiliary_reward,
                                       torch.tensor(reward, device=self.agent.device).unsqueeze(0),
                                       torch.tensor(terminating, device=self.agent.device).unsqueeze(0),
                                       state_prime,
                                       visited_prime])

                if terminating:
                    replay_que.put(copy.deepcopy(episode_buffer))
                    end = time.process_time()
                    elapsed = (end - start)
                    state_prime = env.reset()
                    self.update_partitions(self.agent.visited,
                                           self.local_partition_memory)  # TODO SHOULD local_partition_memory be shared since we just have replicated data for reading? (asnwer is yes)
                    self.agent.visited = []
                    logging.info("step: |{0}| total_score:  |{1}| Time: |{2:.2f}| Time pr step: |{3:.2f}|"
                                 .format(str(i).rjust(7, " "),
                                         int(total_score),
                                         elapsed,
                                         elapsed / len(episode_buffer)))
                    episode_buffer.clear()
                    visited.clear()
                    visited_prime.clear()
                    total_score = 0

                if distance > dmax:
                    partition_candidate = state_prime
                    dmax = distance

                if i % 10000 == 0:
                    from_actor_partition_que.put(copy.deepcopy([partition_candidate, dmax]))

                state = state_prime
                if i % 1000 == 0:  # TODO this should prob be some better mere defined value
                    try:
                        self.check_ques_for_updates()
                    except Exception as err:
                        logging.info(err)
                        logging.info(traceback.format_exc())

        except Exception as err:
            logging.info(err)
            logging.info(traceback.format_exc())

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
    def update_partitions(visited_partitions, partition_memory):
        for visited in visited_partitions:
            for i, partition in enumerate(partition_memory):
                if torch.equal(visited[0], partition[0]):
                    partition_memory[i][1] += 1
                    break

    def check_ques_for_updates(self):
        self.check_que_and_update_network(self.q_network_que, self.agent.Qnet)
        self.check_que_and_update_network(self.e_network_que, self.agent.EEnet)
        self.check_que_and_update_network(self.q_t_network_que, self.agent.targetQnet)
        self.check_que_and_update_network(self.e_t_network_que, self.agent.targetEEnet)

        if not self.to_actor_partition_que.empty():
            partition = self.to_actor_partition_que.get()
            proces_local_partition = copy.deepcopy(partition)
            if proces_local_partition[0] is None:
                    logging.info("jank")
            self.local_partition_memory.append(proces_local_partition)
            if len(self.local_partition_memory) > 100:  # TODO get self.argument here for length
                self.local_partition_memory.pop(0)
            del partition
            logging.info("updated partition memory")

    @staticmethod
    def check_que_and_update_network(que, network):
        if not que.empty():
            parameters = que.get()
            proces_local_parameters = copy.deepcopy(parameters)
            for name, single_param in network.state_dict().items():
                single_param = proces_local_parameters[name]
                network.state_dict()[name].copy_(single_param)
            del parameters
