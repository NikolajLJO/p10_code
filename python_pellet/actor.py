import queue

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
							format='%(asctime)-15s | %(message)s',
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
		visited = torch.zeros(1,100, device=self.agent.device)
		visited_prime = torch.zeros(1,100, device=self.agent.device)
		state = env.reset()
		self.local_partition_memory.append([state, 0])
		steps_since_reward = 0
		try:
			for i in range(1, int(args[2])):
				start = time.process_time()
				action, policy = self.agent.find_action(state, i, visited, steps_since_reward)

				auxiliary_reward = torch.tensor(self.calculate_auxiliary_reward(policy, action.item()),device=self.agent.device)

				state_prime, reward, terminating, info = env.step(action)
				total_score += reward
				reward = int(max(min(reward, 1), -1))
				if i % 10 == 0:
					visited, visited_prime, distance = self.agent.find_current_partition(state_prime,self.local_partition_memory, visited)
				episode_buffer.append(
					[
						copy.deepcopy(state).to("cpu"),
						copy.deepcopy(action).to("cpu"),
						copy.deepcopy(visited).to("cpu"),
						copy.deepcopy(auxiliary_reward).to("cpu"),
						copy.deepcopy(torch.tensor(reward, device="cpu").unsqueeze(0)),
						copy.deepcopy(torch.tensor(terminating, device="cpu").unsqueeze(0)),
						copy.deepcopy(state_prime).to("cpu"),
						copy.deepcopy(visited_prime).to("cpu")])

				if terminating:
					replay_que.put(copy.deepcopy(episode_buffer))
					end = time.process_time()
					elapsed = (end - start)
					state_prime = env.reset()
					self.update_partitions(visited, self.local_partition_memory)  # TODO SHOULD local_partition_memory be shared since we just have replicated data for reading? (asnwer is yes)
					logging.info("step: |{0}| total_score:  |{1}| Time: |{2:.2f}| Time pr step: |{3:.2f}|".format(str(i).rjust(7, " "),int(total_score),elapsed,elapsed / len(episode_buffer)))
					episode_buffer.clear()
					visited[visited != 0] = 0
					visited_prime[visited_prime != 0] = 0

					total_score = 0
					steps_since_reward = 0



				if reward != 0 or len(visited) != len(visited_prime):
					steps_since_reward = 0
				else:
					steps_since_reward += 1

				if distance > dmax:
					partition_candidate = copy.deepcopy(state_prime).to("cpu")
					dmax = distance

				if i % 10000 == 0:
					from_actor_partition_que.put(copy.deepcopy([partition_candidate, dmax]))

				state = state_prime
				if i % 1000 == 0:  # TODO this should prob be some better mere defined value
					self.check_ques_for_updates()

				if steps_since_reward > 500:
					terminating = True
					episode_buffer[-1][5] = torch.tensor(terminating, device=self.agent.device).unsqueeze(0)
					steps_since_reward = 0

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
	def update_partitions(visited, partition_memory):

		for i, _visited in enumerate(visited[0]):
			if _visited.item() != 0:
				partition_memory[i][1] += 1

	def check_ques_for_updates(self):
		self.check_que_and_update_network(self.q_network_que, self.agent.Qnet)
		self.check_que_and_update_network(self.e_network_que, self.agent.EEnet)
		self.check_que_and_update_network(self.q_t_network_que, self.agent.targetQnet)
		self.check_que_and_update_network(self.e_t_network_que, self.agent.targetEEnet)

		if not self.to_actor_partition_que.empty():
			try:
				partition = self.to_actor_partition_que.get(False)
				proces_local_partition = copy.deepcopy(partition)
				proces_local_partition[0] = proces_local_partition[0].to(self.agent.device)


				if len(self.local_partition_memory) == 100:  # TODO get self.argument here for length
					self.local_partition_memory.pop(0)
					self.local_partition_memory.append(proces_local_partition)
				else:
					self.local_partition_memory.append(proces_local_partition)
				del partition
				logging.info("updated partition memory")
			except queue.Empty:
				pass

	@staticmethod
	def check_que_and_update_network(que, network):
		if not que.empty():
			try:
				parameters = que.get(False)
				proces_local_parameters = copy.deepcopy(parameters)
				for name, single_param in network.state_dict().items():
					single_param = proces_local_parameters[name]
					network.state_dict()[name].copy_(single_param)
				del parameters
			except queue.Empty:
				pass
