'''
Train is the main file when you wich to train an agent
it uses some commandline parameters these are:
    args1 = gamename
    args2 = traenigsperiode
    args3 = network update frequency
    args4 = partition update frequency
'''

import os
import datetime
import logging
import time
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from init import setup

transform_to_image = transforms.ToPILImage()

# Partition constants
MAX_PARTITIONS = 100
START_MAKING_PARTITIONS = 200000
PARTITION_ADD_TIME_MULT = 1.2

# EE and RND constants
START_EELEARN = 25000
END_EELEARN = 200000

# Q constants
START_QLEARN = 225000

# Shared constants
UPDATE_TARGETS_FREQUENCY = 1000
SAVE_NETWORKS_FREQUENCY = 50000

def get_writer():
    '''
    this function return the OS writer
    Return: OS writer
    '''
    _, writer = os.pipe()
    return os.fdopen(writer, 'w')


def mainloop(args):
    '''
    mainloop runs the main training and calls all other functions and methods as necessary
    Input: args = [stdinput,
                   gamename,
                   trainingsteps,
                   network_update_frequency,
                   partition_update_frequency,
                   use_RND]
    '''
    torch.set_flush_denormal(True)
    path = Path(__file__).parent
    Path(path / 'logs').mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    logpath= str(path) + "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        filename=(logpath + "-log.txt"),
                        filemode='w')
    logger = get_writer()
    sys.stdout = logger

    partition_candidate = None
    terminating = False
    total_score = 0
    reward = np.NINF
    dmax = 0
    distance = 0
    episode_buffer = []

    _, replay_memory, agent, _, env = setup(args[1], args[5])
    add_partition_freq = int(args[4])
    update_freq = int(args[3])

    state = env.reset()
    partition_memory = [[state, 0]]
    transform_to_image(state[0][0].cpu()).save(logpath + "patition_1.png")
    state_prime = None
    visited = torch.zeros([1,100], device=agent.device)
    visited_prime = torch.zeros([1,100], device=agent.device)
    now = time.process_time()
    steps_since_reward = 0

    for i in range(1, int(args[2])):

        action, policy = agent.find_action(state, i)

        auxiliary_reward = torch.tensor(calculate_auxiliary_reward(policy,
                                                                   action.item()),
                                        device=agent.device)
        # if the state is not terminating take the action and save the trasition
        # else reset the game an all related variables.
        if not terminating:
            state_prime, reward, terminating, _ = env.step(action.item())
            total_score += reward
            reward = max(min(reward,1),-1)
            if i % 10 == 0:
                visited, visited_prime, distance = agent.find_current_partition(state_prime,
                                                                                partition_memory)
            episode_buffer.append([state, action, visited, auxiliary_reward,
                                   torch.tensor(reward, device=agent.device).unsqueeze(0),
                                   torch.tensor(terminating, device=agent.device).unsqueeze(0),
                                   state_prime,
                                   visited_prime])
        else:
            state_prime = env.reset()
            terminating = False
            update_partitions(agent.visited,partition_memory)
            agent.visited[agent.visited != 0] = 0
            visited[visited != 0] = 0
            visited_prime[visited_prime != 0] = 0
            replay_memory.save(episode_buffer)
            episode_time = time.process_time()-now
            logging.info("step: |{0}| total_score:  |{1}| Time: |{2:.2f}| Time pr step: |{3:.4f}| Partition #: |{4}|"
                         .format(str(i).rjust(7, " "),
                                 int(total_score),
                                 episode_time,
                                 episode_time / len(episode_buffer),
                                 len(partition_memory)))
            episode_buffer.clear()
            now = time.process_time()
            total_score = 0
            steps_since_reward = 0

        if reward != 0 or (torch.sum(visited_prime, 1) - torch.sum(visited, 1)).item() != 0:
            steps_since_reward = 0
        else:
            steps_since_reward += 1

        if distance > dmax and i >= START_MAKING_PARTITIONS:
            partition_candidate = state_prime
            dmax = distance

        if i % add_partition_freq == 0 and partition_candidate is not None:
            partition_memory.append([partition_candidate, 0])
            dmax = 0

            partition_memory = partition_memory[-MAX_PARTITIONS:]
            add_partition_freq = int(add_partition_freq * PARTITION_ADD_TIME_MULT)
            transform_to_image(state[0][0].cpu()).save(logpath + "patition_" +
                                                 str(len(partition_memory)) + ".png")

        state = state_prime
        visited = visited_prime

        if i % update_freq == 0 and i >= START_QLEARN:
            agent.qlearn(replay_memory)

        if i % update_freq == 0 and i >= START_EELEARN and (i < END_EELEARN or int(args[5])):
            agent.imagecomparelearn(replay_memory)

        if i % UPDATE_TARGETS_FREQUENCY == 0:
            agent.update_targets()

        if i % SAVE_NETWORKS_FREQUENCY == 0:
            agent.save_networks(logpath, i)

        if steps_since_reward > 500:
            terminating = True
            episode_buffer[-1][5] = torch.tensor(terminating, device=agent.device).unsqueeze(0)
            steps_since_reward = 0

    agent.save_networks(logpath, i)


def calculate_auxiliary_reward(policy, aidx):
    '''
    this function calcules the the auxillary reward for each action
    Input: policy is the calculated values for each action
           aidx is the index of the used action
    output: a list of auxillary rewards for the actions in the action space
    '''
    aux = [0]*(policy.size()[1])
    policy = policy.squeeze(0)
    for i in range(len(aux)):
        if aidx == i:
            aux[i] = 1 - policy[i].item()
        else:
            aux[i] = -policy[i].item()
    return aux

def update_partitions(visited_partitions, partition_memory):
    '''
    this function updates the number of visits that
    each partition has when an episode is terminatied
    Input: visited_partitions is the list of visited partitions in an episode
           partition_memory is the full partition memory
    '''
    for i, visited in enumerate(visited_partitions[0]):
        if visited != 0:
            partition_memory[i][1] += 1


if __name__ == "__main__":
    print("start")
    mainloop(sys.argv)
    print("done")
