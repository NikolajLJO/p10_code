import sys
import multiprocessing as mp



from actor import Actor
from learner import Learner
from memory_manager import MemoryManager
from memory import ReplayMemory

if __name__ == "__main__":
    mp.set_start_method('spawn')
    thread_count = 6
    actor_count = 2
    replay_memory = ReplayMemory()
    learner_que_max_size = 1000
    learner_ee_que_max_size = 1000
    args = sys.argv
    with mp.Pool(processes=thread_count) as pool:
        replay_que = mp.Queue()
        partition_que = mp.Queue()
        q_network_que = mp.Queue()
        e_network_que = mp.Queue()
        q_t_network_que = mp.Queue()
        e_t_network_que = mp.Queue()
        learner_ee_que = mp.Queue(maxsize=learner_ee_que_max_size)
        learner_replay_que = mp.Queue(maxsize=learner_que_max_size)

        manager = mp.Process(target=MemoryManager,
                             args=(replay_que,
                                   partition_que,
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
                                   actor_count))
        learner.start()
        actor_list = [
            mp.Process(target=Actor,
                       args=(args,
                             i,
                             replay_que,
                             partition_que,
                             q_network_que,
                             e_network_que,
                             q_t_network_que,
                             e_t_network_que))
            for i in range(actor_count)]
        for process in actor_list:
            process.start()

    actor_list[0].join()
