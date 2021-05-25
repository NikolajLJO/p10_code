import sys
from torch import multiprocessing as mp
from tools import process_score_over_steps, process_dis
from actor import Actor
from learner import Learner
from memory_manager import MemoryManager
from memory import ReplayMemory
import torch
import numpy as np
from init import setup

if __name__ == "__main__":
    mp.set_start_method('spawn')
    thread_count = min(mp.cpu_count(), 32)
    actor_count = thread_count - 6
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

    if args[5] and args[5] == "rnd":
        should_use_rnd = True
    else:
        should_use_rnd = False

    with mp.Pool(processes=thread_count) as pool:
        replay_que = mp.Queue()
        q_network_que = mp.Queue()
        e_network_que = mp.Queue()
        q_t_network_que = mp.Queue()
        e_t_network_que = mp.Queue()
        to_actor_partition_que = mp.Queue()
        from_actor_partition_que = mp.Queue()
        learner_ee_que = mp.Queue(maxsize=learner_ee_que_max_size)
        learner_replay_que = mp.Queue(maxsize=learner_que_max_size)

        manager = MemoryManager(replay_que,
                                   learner_replay_que,
                                   learner_que_max_size,
                                   learner_ee_que,
                                   learner_ee_que_max_size)

        mp.spawn(manager.manage(replay_que,
                                   learner_replay_que,
                                   learner_que_max_size,
                                   learner_ee_que,
                                   learner_ee_que_max_size))

        learner = Learner(args,
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
                                   should_use_rnd)

        mp.spawn(learner.learn(learner_replay_que, learner_ee_que, from_actor_partition_que, to_actor_partition_que,
                       actor_count, should_use_rnd))


        actor_list = [
            Actor(args,
                             i,
                             replay_que,
                             q_network_que,
                             e_network_que,
                             q_t_network_que,
                             e_t_network_que,
                             from_actor_partition_que,
                             to_actor_partition_que,
                             should_use_rnd)
            for i in range(actor_count)]

        for process in actor_list:
            game_actions, agent, opt, env = setup(args[1], should_use_rnd)
            mp.spawn(process.act(args,
                        np.NINF,
                        np.NINF,
                        env,
                        [],
                        from_actor_partition_que,
                        None,
                        replay_que,
                        env.reset(),
                        0,
                        torch.zeros(1, 100, device=agent.device),
                        torch.zeros(1, 100, device=agent.device)))


