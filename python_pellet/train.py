import sys
import multiprocessing as mp

from actor import Actor
from learner import Learner
from memory_manager import MemoryManager
from memory import ReplayMemory

if __name__ == "__main__":
    mp.set_start_method('spawn')
    thread_count = 4
    replay_memory = ReplayMemory()
    learner_que_max_size = 10000
    with mp.Pool(processes=thread_count) as pool:
        replay_que = mp.Queue()
        partition_que = mp.Queue()
        learner_replay_que = mp.Queue(maxsize=learner_que_max_size)

        manager = mp.Process(target=MemoryManager,
                             args=(sys.argv,
                                   replay_que,
                                   partition_que,
                                   learner_replay_que,
                                   learner_que_max_size))
        manager.start()
        learner = mp.Process(target=Learner,
                             args=(sys.argv,
                                   replay_que,
                                   partition_que,
                                   learner_replay_que,
                                   learner_que_max_size))
        learner.start()
        actor_list = [
            mp.Process(target=Actor,
                       args=(sys.argv,
                             i,
                             replay_que,
                             partition_que))
            for i in range(thread_count)]
        for process in actor_list:
            process.start()
