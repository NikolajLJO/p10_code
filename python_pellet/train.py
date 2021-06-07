import sys
from torch import multiprocessing as mp
from tools import process_score_over_steps, process_dis, calc_percent_per_actor
from actor import Actor
from learner import Learner
from memory_manager import MemoryManager
from memory import ReplayMemory

if __name__ == "__main__":
    mp.set_start_method('spawn')
    thread_count = min(mp.cpu_count(), 32)

    actor_count = 2
    replay_memory = ReplayMemory()
    learner_que_max_size = 1000
    learner_ee_que_max_size = 1000
    args = sys.argv

    args = args[1].split(' ')
    args[1] = int(args[1])

    if len(args) > 1 and (args[2]) == 'y':
        process_score_over_steps(args[3])
        exit()

    if len(args) > 1 and args[2] == 'd':
        process_dis(args[3])
        exit()

    if len(args) > 1 and (args[2]) == 'p':
        calc_percent_per_actor(args[3])
        exit()

    if len(args) > 4 and args[4] == "rnd":
        should_use_rnd = True
    else:
        should_use_rnd = False

    if len(args) > 5:
        try:
            actor_count = int(args[5])
        except ValueError:
            print("your actor count is f'ed")
            actor_count = 1

    with mp.Pool(processes=thread_count) as pool:
        replay_que = mp.Queue(maxsize=100000)
        q_network_que = mp.Queue()
        e_network_que = mp.Queue()
        q_t_network_que = mp.Queue()
        e_t_network_que = mp.Queue()
        to_actor_partition_que = mp.Queue()
        from_actor_partition_que = mp.Queue()
        learner_ee_que = mp.Queue(maxsize=learner_ee_que_max_size)
        learner_replay_que = mp.Queue(maxsize=learner_que_max_size)

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
                                   actor_count,
                                   should_use_rnd))
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
                             to_actor_partition_que,
                             should_use_rnd))
            for i in range(actor_count)]
        for process in actor_list:
            process.start()

    for p in actor_list:
        p.join()

    for p in [manager, learner]:
        p.kill()