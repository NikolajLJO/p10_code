import sys
from torch import multiprocessing as mp
from tools import process_score_over_steps, process_dis
from actor import Actor
from learner import Learner
from memory_manager import MemoryManager
from memory import ReplayMemory


class ClearableQueue(mp.Queue):

	def clear(self):
		try:
			while True:
				self.get_nowait()
		except mp.Queue.queue.Empty:
			pass


if __name__ == "__main__":
	mp.set_start_method('spawn')
	thread_count = min(mp.cpu_count(), 32)
	actor_count = thread_count - 4
	replay_memory = ReplayMemory()
	learner_que_max_size = 1000
	learner_ee_que_max_size = 1000
	args = sys.argv

	args = args[1].split(' ')
	args.reverse()
	args.append("shit")
	args.reverse()
	args[2] = int(args[2])

	if (args[3]) == 'y':
		process_score_over_steps(args[4])
		exit()

	if args[3] == 'd':
		process_dis(args[4])
		exit()

	with mp.Pool(processes=thread_count) as pool:
		replay_que = ClearableQueue()
		q_network_que = ClearableQueue()
		e_network_que = ClearableQueue()
		q_t_network_que = ClearableQueue()
		e_t_network_que = ClearableQueue()
		to_actor_partition_que = ClearableQueue()
		from_actor_partition_que = ClearableQueue()
		learner_ee_que = ClearableQueue(maxsize=learner_ee_que_max_size)
		learner_replay_que = ClearableQueue(maxsize=learner_que_max_size)

		manager = mp.Process(target=MemoryManager,
							 args=(replay_que,
								   learner_replay_que,
								   learner_que_max_size,
								   learner_ee_que,
								   learner_ee_que_max_size))
		manager.start()
		learner = mp.Process(target=Learner,
							 args=(args,
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
								   actor_count))
		learner.start()
		actor_list = [
			mp.Process(target=Actor,
					   args=(args,
							 i,
							 replay_que,
							 q_network_que,
							 e_network_que,
							 q_t_network_que,
							 e_t_network_que,
							 from_actor_partition_que,
							 to_actor_partition_que))
			for i in range(actor_count)]
		for process in actor_list:
			process.start()

	for p in actor_list:
		p.join()


