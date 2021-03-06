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


MAX_PARTITIONS = 100
start_making_partitions = 50000
initial_memory_fill = 50000
update_targets_frequency = 10000
save_networks_frequency = 500000
'''
args1 = gamename
args2 = traenigsperiode
args3 = network update frequency
args4 = partition update frequency
''' 

def get_writer():
    _, writer = os.pipe()
    return os.fdopen(writer, 'w')


def mainloop(args):
    path = Path(__file__).parent
    Path(path / 'logs').mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        filename=(str(path) + "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute) + "-log.txt"),
                        filemode='w')
    logger = get_writer()
    sys.stdout = logger
    
    partition_candidate = None
    terminating = False
    total_score = 0
    reward = np.NINF
    dmax = np.NINF
    episode_buffer = []

    game_actions, replay_memory, agent, opt, env = setup(args[1])
    
    state = env.reset()
    partition_memory = [[state,0]]
    state_prime = None
    now = datetime.datetime.now()

    for i in range(1, int(args[2])):
        action, policy = agent.find_action(state, i)

        auxiliary_reward = torch.tensor(calculate_auxiliary_reward(policy, action.item()), device=agent.device)

        if not terminating:
            state_prime, reward, terminating, info = env.step(action.item())
            total_score += reward
            reward = max(min(reward,1),-1)
            visited, visited_prime, distance = agent.find_current_partition(state_prime, partition_memory)
            episode_buffer.append([state, action, visited, auxiliary_reward, 
                                   torch.tensor(reward, device=agent.device).unsqueeze(0), 
                                   torch.tensor(terminating, device=agent.device).unsqueeze(0), 
                                   state_prime, 
                                   visited_prime])
        else:
            state_prime = env.reset()
            terminating = False
            update_partitions(agent.visited,partition_memory)
            agent.visited = []
            replay_memory.save(episode_buffer)
            episode_buffer.clear()
            logging.info("step: " + str(i) + " total_score: " + str(total_score) + " time taken: " + str(datetime.datetime.now()-now))
            now = datetime.datetime.now()
            total_score = 0
            
        if distance > dmax and i >= start_making_partitions:
            partition_candidate = state_prime
            dmax = distance
        
        if i % int(args[4]) == 0 and partition_candidate is not None:
            partition_memory.append([partition_candidate, 0])
            dmax = 0
            if len(partition_memory) > MAX_PARTITIONS:
                partition_memory[-MAX_PARTITIONS:]

        state = state_prime

        if i % int(args[3]) == 0 and i >= initial_memory_fill:
            agent.update(replay_memory)
        
        if i % update_targets_frequency == 0:
            agent.update_targets()
            
        if i % save_networks_frequency == 0:
            agent.save_networks(path, i)
    
    agent.save_networks(path, i)


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

def update_partitions(visited_partitions, partition_memory):
    for visited in visited_partitions:
        for i, partition in enumerate(partition_memory):
            if torch.equal(visited[0],partition[0]):
                partition_memory[i][1] += 1
                break


if __name__ == "__main__":
    print("start")
    '''mp.set_start_method('spawn')
    with mp.Pool(processes=4) as pool:
        que = mp.Queue()
        process = mp.Process(target=mainloop, args=(sys.argv,))
'''
    mainloop(sys.argv)
