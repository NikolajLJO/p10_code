import queue

import tools
from pathlib import Path
import datetime
import logging
import sys
import copy
import torch
import traceback

from memory import ReplayMemory


class MemoryManager:
	def __init__(self, replay_que, learner_replay_que, learner_que_max_size, learner_ee_que, learner_ee_que_max_size):
		torch.multiprocessing.set_sharing_strategy('file_system')
		path = Path(__file__).parent
		Path(path / 'logs').mkdir(parents=True, exist_ok=True)
		now = datetime.datetime.now()
		now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
		logging.basicConfig(level=logging.DEBUG,
							format='%(asctime)-15s | %(message)s',
							filename=(str(path) + now_but_text + "-manager" + "-log.txt"),
							filemode='w')
		logger = tools.get_writer()
		sys.stdout = logger
		self.replay_memory = ReplayMemory()
		self.partition_memory = []
		try:
			self.manage(learner_replay_que, learner_que_max_size, replay_que, learner_ee_que, learner_ee_que_max_size)
		except Exception as err:
			logging.info(err)
			logging.info(traceback.format_exc())

	def manage(self, learner_replay_que, learner_que_max_size, replay_que, learner_ee_que, learner_ee_que_max_size):
		while True:
			replay_mem_len = len(self.replay_memory.memory)
			#logging.info("re: " + str(replay_mem_len) + " lq: " + str(learner_replay_que.qsize()) + " eeq: " + str(learner_ee_que.qsize()))
			if not self.replay_memory.memory or len(self.replay_memory.memory) < (learner_que_max_size + learner_ee_que_max_size) or (learner_replay_que.full() and learner_ee_que.full()):
				try:
					optional_replay = replay_que.get_nowait()
					process_local_optional_replay = copy.deepcopy(optional_replay)
					self.replay_memory.save(process_local_optional_replay)
					del optional_replay
					logging.info("replay memory at " + str(len(self.replay_memory.memory)))
				except queue.Empty:
					pass
				except Exception as err:
					logging.info(err)
					logging.info(traceback.format_exc())
					pass
			else:
				if learner_replay_que.qsize() == 0 and replay_mem_len > learner_que_max_size:
					pre = learner_replay_que.qsize()
					logging.info("refill learner replay mem")
					self.fill_learner_replay_que(learner_replay_que, learner_que_max_size)
					logging.info("refilled learner replay mem with |" + str(learner_replay_que.qsize() - pre) + "| elements")

				if learner_ee_que.qsize() == 0 and replay_mem_len > learner_ee_que_max_size:
					pre = learner_ee_que.qsize()
					logging.info("refill learner ee mem")
					self.fill_learner_ee_que(learner_ee_que, learner_ee_que_max_size)
					logging.info("refilled learner ee mem with |" + str(learner_ee_que.qsize() - pre) + "| elements")

	def fill_learner_replay_que(self, learner_replay_que, learner_que_max_size):
		batch = self.replay_memory.sample(forced_batch_size=learner_que_max_size)
		for item in batch:
			try:
				learner_replay_que.put_nowait(copy.deepcopy(item))
			except queue.Full:
				pass

	def fill_learner_ee_que(self, learner_ee_que, learner_ee_que_max_size):
		batch = self.replay_memory.sample_ee_minibatch(forced_batch_size=learner_ee_que_max_size)
		for item in batch:
			try:
				learner_ee_que.put_nowait(copy.deepcopy(item))
			except queue.Full:
				pass
