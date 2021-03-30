import numpy as np
import torch
from random import sample
from init import setup
import sys
import multiprocessing as mp



T_add = 10
batch_size = 32
k = 10
'''
args1 = gamename
args2 = trænigsperiode
args3 = network update frequency
args4 = partition update frequency
'''


def mainloop(args):
    partition_memory = []
    partition_candidate = None
    terminating = False
    total_score = 0
    reward = np.NINF
    dmax = np.NINF

    game_actions, replay_memory, agent, opt, env = setup(args[1])

    state = env.reset()
    state_prime = None

    for i in range(int(args[2])):
        action, policy = agent.find_action(state)

        auxiliary_reward = calculate_auxiliary_reward(policy, action)

        if not terminating:
            state_prime, reward, terminating, info = env.step(action)
            total_score += reward
        else:
            state = env.reset()
            agent.visited = []

        visited, visited_prime, distance = agent.find_current_partition(state_prime, partition_memory)

        if distance > dmax:
            partition_candidate = state_prime
            dmax = distance
        
        replay_memory.save(state, action, visited, auxiliary_reward, reward, terminating, state_prime, visited_prime)

        if i % int(args[4]) == 0 and partition_candidate is not None:
            partition_memory.append(partition_candidate)
            dmax = 0

        state = state_prime

        if i % int(args[3]) == 0:
            agent.update(replay_memory)


def calculate_auxiliary_reward(policy, aidx):
    aux = [0]*policy.size()[0]
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


def sample_ee_minibatch(memory, batch_size, memory_replace_pointer):
    resbatch = []
    batch = sample(list(enumerate(memory)), batch_size)

    # this loop takes the elements in the batch and goes k elements forward to give the
    # auxilaray calculations on what actions has been used between state s_0 and state s_0+k
    for element in batch:
        if len(memory) >= element[0] + k and (
                memory_replace_pointer < element[0] or memory_replace_pointer >= element[0] + k):
            auxs = [element[1][3]]
            for i in range(1, k):
                aux = memory[element[0] + i][3]
                auxs.append(aux)
                resbatch.append([element[1][0], memory[element[0] + i][0], memory[element[0] + 1][0], auxs])

    return resbatch


def updatepartitions(r, vitited_partitions):
    for i, r in enumerate(r):
        for vitited_partition in vitited_partitions:
            if torch.equal(r[0], vitited_partition[0]):
                r[i] = (r[0], r[1] + 1)
                break
    return r


if __name__ == "__main__":
    mp.set_start_method('spawn')
    with mp.Pool(processes=4) as pool:
        que = mp.Queue()
        process = mp.Process(target=mainloop, args=(sys.argv,))

    mainloop(sys.argv)
