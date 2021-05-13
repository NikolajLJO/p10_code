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
from init import setup_agent
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
				 actor_count):
		torch.multiprocessing.set_sharing_strategy('file_system')
		self.path = Path(__file__).parent
		Path(self.path / 'logs').mkdir(parents=True, exist_ok=True)
		now = datetime.datetime.now()
		now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
		logging.basicConfig(level=logging.DEBUG,
							format='%(asctime)-15s | %(message)s',
							filename=(str(self.path) + now_but_text + "-learner" + "-log.txt"),
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
		self.partition = 0
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
		learn_count = 0
		while True:
			# when rpelay memory is almost empty, wait until the que has a full memory size
			while learner_replay_que.qsize() < self.learner_que_max_size:
				pass
			# then when it does, update it
			for _ in range(0, int(self.learner_que_max_size)):
				try:
					transition = learner_replay_que.get(False)
					for elem in itertools.islice(transition, 0, 7):
						elem = elem.to(self.agent.device)
					process_local_transition = copy.deepcopy(transition)
					self.replay_memory.memory.append(process_local_transition)
					del transition
				except queue.Empty:
					pass

			logging.info("Refilled replay memory: " + str(len(self.replay_memory)))

			ee_update_count = 0
			ee_done = False

			while learner_ee_que.qsize() < self.learner_ee_que_max_size:
				pass

			if not learner_ee_que.empty():
				for _ in range(0, int(self.learner_ee_que_max_size)):
					try:
						transition = learner_ee_que.get(False)
						process_local_transition = copy.deepcopy(transition)
						self.ee_memory.append(process_local_transition)
						del transition
					except queue.Empty:
						torch.cuda.empty_cache()
						pass
				logging.info("Refilled ee memory:" + str(len(self.ee_memory)))
				ee_update_count += 1

			if not ee_done and ee_update_count * self.learner_ee_que_max_size > 2e6:
				ee_done = True

			if from_actor_partition_que.qsize() >= actor_count * 2:
				unqued_partitions = []
				for _ in range(actor_count*2):
					try:
						partition = from_actor_partition_que.get(False)
						process_local_partition = copy.deepcopy(partition)
						unqued_partitions.append(process_local_partition)
						del partition
					except queue.Empty:
						pass
				best_partition = max(unqued_partitions, key=lambda item: item[1])
				path = (self.path / ("patition_" + str(self.partition) + ".png")).__str__()
				self.partition += 1
				transform_to_image(best_partition[0][0][0]).save(path)
				for _ in range(actor_count):
					to_actor_partition_que.put(copy.deepcopy(best_partition))
				unqued_partitions.clear()

				try:
					while True:
						from_actor_partition_que.get_nowait()
				except queue.Empty:
					pass
				logging.info("Pushed partition: " + str(self.partition))

			# while we have more than 10% replay memory, learn
			# ToDO this should prob just do entire que its a frakensetein of old concepts
			while len(self.replay_memory.memory) >= self.update_memory_break_point and len(self.ee_memory) >= self.update_ee_memory_break_point:
				self.agent.update(self.replay_memory, self.ee_memory, ee_done)
				learn_count += 1

			self.ee_memory.clear()
			logging.info("I processed que: " + str(learn_count))

			c1 = self.agent.Qnet.state_dict()
			c2 = self.agent.EEnet.state_dict()
			c3 = self.agent.targetQnet.state_dict()
			c4 = self.agent.targetEEnet.state_dict()

			for _ in range(actor_count):
				self.q_network_que.put(copy.deepcopy(c1))
				self.e_network_que.put(copy.deepcopy(c2))
				self.q_t_network_que.put(copy.deepcopy(c3))
				self.e_t_network_que.put(copy.deepcopy(c4))

			logging.info("Pushed networks")
