import gym
import numpy as np
import torch
import math
from random import randrange, sample
from Qnetwork import Qnet, EEnet
from environment import create_atari_env
from init import setup
import sys

slope = -(1 - 0.05) / 1000000
intercept = 1
total_steps = 0
T_add = 10
nq = 0.1
ne = 0.99
ee_beta = 1
batch_size = 32
Q_discount = 0.99
EE_discount = 0.99
k = 10
'''
args1 = gamename
args2 = trænigsperiode
args3 = update frequency
'''
def mainloop(args):
    partition_memory = []
    partition_candidate = None
    terminating = False
    total_score = 0
    Dmax = np.NINF
    game_actions, agent, replay_memory, opt, env = setup(args[1])

    state = env.reset()

    for i in range(args[2]):
        action = agent.find_action(state)

        if not terminating:
            state_prime, reward, terminating, info = env.step(action)
            total_score += reward
        else:
            state = env.reset()
            agent.visited = []
        
        visited, visited_prime, distance = agent.find_current_partition(state_prime, partition_memory)
        
        if distance > Dmax:
            partition_candidate = state_prime
            Dmax = distance
        
        replay_memory.save(state, action, visited, reward, terminating, state_prime, visited_prime)

        if i % args[4] == 0 and partition_candidate is not None:
            partition_memory.append(partition_candidate)
            Dmax=0

        state = state_prime

        if i % args[3] == 0:
            agent.update(replay_memory)

def Calculateauxiliaryreward(policy, aidx):
    aux = [0]*18
    policy = policy.squeeze(0)
    for i in range(len(aux)):
        if aidx == i:
            aux[i] = 1 - policy[i].item()
        else:
            aux[i] = -policy[i].item()
    return aux

def partitiondeterminarion(EEagent, s_n, R):
    mindist = [np.inf, None]
    for s_pi in R:
        dist = distance(EEagent, s_n, s_pi[0], R[0][0])
        if mindist[0] > dist:
            mindist[0] = dist
            mindist[1] = s_pi
    return mindist[1]

def sampleEEminibatch(memory, batch_size, memory_replace_pointer):
    resbatch = []
    batch = sample(list(enumerate(memory)), batch_size)

    # this loop takes the elements in the batch and goes k elements forward to give the auxilaray calculations on what actions has been used
    # between state s_0 and state s_0+k
    for element in batch:
        if len(memory) >= element[0] + k and (memory_replace_pointer < element[0] or memory_replace_pointer >= element[0] + k):
            auxs = [element[1][3]]
            for i in range(1, k):
                aux = memory[element[0] + i][3]
                auxs.append(aux)
            resbatch.append([element[1][0], memory[element[0] + i][0], memory[element[0]+1][0], auxs])

    return resbatch

def updatepartitions(R, vitited_partitions):
    for i, r in enumerate(R):
        for vitited_partition in vitited_partitions:
            if torch.equal(r[0],vitited_partition[0]):
                R[i] = (r[0], r[1]+1)
                break
    return R

if __name__ == "__main__":
    mainloop(sys.argv)