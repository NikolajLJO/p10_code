import numpy as np
import torch
from random import sample
from init import setup
import sys
import multiprocessing as mp
import os
from pathlib import Path
import datetime
import logging


'''
args1 = gamename
args2 = trænigsperiode
args3 = network update frequency
args4 = partition update frequency
'''


def get_writer():
    _, writer = os.pipe()
    return os.fdopen(writer, 'w')


def mainloop(args, process_itterator):
    path = Path(__file__).parent
    Path(path / 'logs').mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        filename=(str(path) + now_but_text + "p" + str(process_itterator) + "-log.txt"),
                        filemode='w')
    logger = get_writer()
    sys.stdout = logger
    
    partition_candidate = None
    terminating = False
    total_score = 0
    dmax = np.NINF
    distance = np.NINF
    episode_buffer = []

    game_actions, replay_memory, agent, opt, env = setup(args[1])

    state = env.reset()
    partition_memory = [state]

    for i in range(1, int(args[2])):
        action, policy = agent.find_action(state)

        auxiliary_reward = calculate_auxiliary_reward(policy, action)

        if not terminating:
            state_prime, reward, terminating, info = env.step(action)
            total_score += reward
            reward = max(min(reward, 1), -1)
            visited, visited_prime, distance = agent.find_current_partition(state_prime, partition_memory)
            episode_buffer.append([state, action, visited, auxiliary_reward, reward, terminating, state_prime, visited_prime])
        else:
            state_prime = env.reset()
            terminating = False
            agent.visited = []
            replay_memory.save(episode_buffer)
            episode_buffer.clear()
            logging.info("step: " + str(i) + " total_score: " + str(total_score))
            total_score = 0
            
        if distance > dmax:
            partition_candidate = state_prime
            dmax = distance
        
        if i % int(args[4]) == 0 and partition_candidate is not None:
            partition_memory.append(partition_candidate)
            dmax = 0

        state = state_prime

        if i % int(args[3]) == 0:
            agent.update(replay_memory)
        
        if i % 1000 == 0:
            agent.update_targets()


def calculate_auxiliary_reward(policy, aidx):
    aux = [0]*(policy.size()[1])
    policy = policy.squeeze(0)
    for i in range(len(aux)):
        if aidx == i:
            aux[i] = 1 - policy[i].item()
        else:
            aux[i] = -policy[i].item()
    return aux


def partitiondeterminarion(ee_network, s_n, r):
    mindist = [np.inf, None]
    for s_pi in r:
        dist = ee_network.distance(ee_network, s_n, s_pi[0], r[0][0])
        if mindist[0] > dist:
            mindist[0] = dist
            mindist[1] = s_pi
    return mindist[1]


def updatepartitions(r, vitited_partitions):
    for i, r in enumerate(r):
        for vitited_partition in vitited_partitions:
            if torch.equal(r[0], vitited_partition[0]):
                r[i] = (r[0], r[1] + 1)
                break
    return r


if __name__ == "__main__":
    mp.set_start_method('spawn')
    thread_count = 4
    with mp.Pool(processes=thread_count) as pool:
        que = mp.Queue()
        process_list = [mp.Process(target=mainloop, args=(sys.argv, x)) for x in range(1, thread_count)]

        for process in process_list:
            process.start()

