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
		self.agent = setup(args[1])[0]
		self.partition = 0
		self.replay_memory = ReplayMemory(max_memory_size=learner_que_max_size)
		self.ee_memory = []
		self.learner_que_max_size = learner_que_max_size
		self.learner_ee_que_max_size = learner_ee_que_max_size
		try:
			self.learn(learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que, actor_count)
		except Exception as err:
			logging.info(err)
			logging.info(traceback.format_exc())

	def learn(self, learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que,  actor_count):
		logging.info("Started with empty memory")
		learn_count = 0
		ee_update_count = 0
		ee_done = False
		i = 0
		while True:
			logging.info("itt with: " +
						str(len(self.replay_memory.memory)) +
						"rm  and: " +
						str(len(self.ee_memory)) +
						"eem and: " +
						str(learner_replay_que.qsize()) +
						"r que and: " +
						str(learner_ee_que.qsize()) +
						"eeq")

			# when rpelay memory is almost empty, wait until the que has a full memory size

			while learner_replay_que.qsize() < self.learner_que_max_size:
				pass

			# then when it does, update it

			if learner_replay_que.qsize() == self.learner_que_max_size and len(self.replay_memory.memory) < 32:
				pre = learner_replay_que.qsize()
				logging.info("Refill e memory")
				for _ in range(0, learner_replay_que.qsize()):
					try:
						transition = learner_replay_que.get()
						process_local_transition = copy.deepcopy(transition)
						self.replay_memory.memory.append(process_local_transition)
						del transition
					except queue.Empty:
						pass

				logging.info("Refilled r memory with: " + str(pre - learner_replay_que.qsize()) + "total: " + str(len(self.replay_memory.memory)))

			while learner_ee_que.qsize() < self.learner_ee_que_max_size:
				pass

			if learner_ee_que.qsize() == self.learner_ee_que_max_size and len(self.ee_memory) < 32:
				pre = learner_ee_que.qsize()
				logging.info("Refill ee memory")
				for _ in range(0, int(self.learner_ee_que_max_size)):
					try:
						transition = learner_ee_que.get()
						process_local_transition = copy.deepcopy(transition)
						self.ee_memory.append(process_local_transition)
						del transition
					except queue.Empty:
						torch.cuda.empty_cache()
						pass
				logging.info("Refilled ee memory with: " + str(pre - learner_ee_que.qsize()) + "total: " + str(len(self.ee_memory)))
				ee_update_count += 1

			if not ee_done and i > 500000:
				ee_done = True

			if from_actor_partition_que.qsize() >= actor_count * 2:
				unqued_partitions = []
				for _ in range(actor_count*2):
					try:
						partition = from_actor_partition_que.get()
						process_local_partition = copy.deepcopy(partition)
						unqued_partitions.append(process_local_partition)
						del partition
					except queue.Empty:
						pass
				logging.info("Push partition")
				best_partition = max(unqued_partitions, key=lambda item: item[1])
				path = (self.path / ("patition_" + str(self.partition) + ".png")).__str__()
				self.partition += 1
				transform_to_image(best_partition[0][0][0]).save(path)
				for _ in range(actor_count):
					try:
						to_actor_partition_que.put_nowait(copy.deepcopy(best_partition))
					except queue.Full:
						pass
				unqued_partitions.clear()

				try:
					while True:
						from_actor_partition_que.get_nowait()
				except queue.Empty:
					pass
				logging.info("Pushed partition: " + str(self.partition))
			
			

			# while we have more than 10% replay memory, learn
			# ToDO this should prob just do entire que its a frakensetein of old concepts
			while self.replay_memory.memory and self.ee_memory:
				self.agent.update(self.replay_memory, self.ee_memory, ee_done)
				i += 1
				if i % 10 == 0:
					self.agent.targetQnet = copy.deepcopy(self.agent.Qnet)
					self.agent.targetEEnet=  copy.deepcopy(self.agent.EEnet)
			
			learn_count += 1

			logging.info("I processed que: " + str(learn_count))

			c1 = self.agent.Qnet.state_dict()
			c2 = self.agent.EEnet.state_dict()
			c3 = self.agent.targetQnet.state_dict()
			c4 = self.agent.targetEEnet.state_dict()
			logging.info("Push networks")
			for _ in range(actor_count):
				try:
					self.q_network_que.put_nowait(copy.deepcopy(c1))
					self.e_network_que.put_nowait(copy.deepcopy(c2))
					self.q_t_network_que.put_nowait(copy.deepcopy(c3))
					self.e_t_network_que.put_nowait(copy.deepcopy(c4))
				except queue.Full:
					pass

			logging.info("Pushed networks")
