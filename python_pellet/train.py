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
from numpy import mean
import torch
import torchvision.transforms as transforms
from init import setup
import copy

transform_to_image = transforms.ToPILImage()

# Partition constants
MAX_PARTITIONS = 100
START_MAKING_PARTITIONS = 2000000
PARTITION_ADD_TIME_MULT = 1.2

# EE and RND constants
START_EELEARN = 250000
END_EELEARN = 2000000

# Q constants
START_QLEARN = 2250000

# Shared constants
UPDATE_TARGETS_FREQUENCY = 10000
SAVE_NETWORKS_FREQUENCY = 500000

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
    done = False
    total_score = 0
    reward = 0
    dmax = 0
    distance = 0
    episode_buffer = []

    replay_memory, agent, env = setup(args[1], args[5], START_QLEARN, int(args[2]))

    partition_addition_step = int(args[4])
    update_freq = int(args[3])

    state = env.reset()
    partition_memory = []
    state_prime = None
    visited = torch.zeros([1,100], device=agent.device)
    visited_prime = torch.zeros([1,100], device=agent.device)
    now = time.process_time()
    scorelist = []
    partaddcount = partition_addition_step

    for i in range(1, int(args[2])):

        if i == START_MAKING_PARTITIONS:
            replay_memory.memory = []
            replay_memory.memory_refrence_pointer = 0

        action, policy = agent.find_action(state, i, visited)

        auxiliary_reward = torch.tensor(calculate_auxiliary_reward(policy,
                                                                   action.item()),
                                        device=agent.device)
        
        # if the state is not terminating take the action and save the trasition
        # else reset the game an all related variables.
        if not done:
            state_prime, reward, terminating, done = env.step(action.item())
            total_score += reward
            reward = max(min(reward,1),-1)
            if terminating:
                reward -= 1
            if i % 1 == 0:
                visited, visited_prime, distance = agent.find_current_partition(state_prime,
                                                                                partition_memory,
                                                                                visited)

                if not terminating and i >= START_MAKING_PARTITIONS and distance > dmax:
                    partition_candidate = state_prime
                    dmax = distance

            episode_buffer.append([state, action, visited, auxiliary_reward,
                                   torch.tensor(reward, device=agent.device).unsqueeze(0),
                                   torch.tensor(terminating, device=agent.device).unsqueeze(0),
                                   state_prime,
                                   copy.deepcopy(visited_prime)])

        else:
            state_prime = env.reset()
            done = False
            update_partitions(visited, partition_memory)
            visited[visited != 0] = 0
            visited_prime[visited_prime != 0] = 0
            if i < END_EELEARN or len(partition_memory) >= 5:
                replay_memory.save(episode_buffer)
            episode_time = time.process_time()-now
            scorelist.append(total_score)
            scorelist = scorelist[-100:]
            logging.info("step: |{0}| total_score:  |{1}| Time: |{2:.2f}| Time pr step: |{3:.4f}| Partition #: |{4}| Average pr. 100: |{5:.2f}| Epsilon: |{6:.2f}|"
                         .format(str(i).rjust(7, " "),
                                 int(total_score),
                                 episode_time,
                                 episode_time / len(episode_buffer),
                                 len(partition_memory),
                                 mean(scorelist),
                                 agent.epsilon))
            episode_buffer.clear()
            now = time.process_time()
            total_score = 0
            agent.steps_since_reward = 0

        if reward != 0 or (torch.sum(visited_prime, 1) - torch.sum(visited, 1)).item() != 0:
            agent.steps_since_reward = 0
        else:
            agent.steps_since_reward += 1

        if partaddcount == 0 and partition_candidate is not None:
            partition_memory.append([partition_candidate, 0])
            dmax = 0
            distance = 0

            partition_addition_step = int(partition_addition_step * PARTITION_ADD_TIME_MULT)
            partaddcount = partition_addition_step

            partition_memory = partition_memory[-MAX_PARTITIONS:]
            transform_to_image(partition_candidate[0].cpu()).save(logpath + "partition_" +
                                                 str(len(partition_memory)) + ".png")

        state = state_prime
        visited = visited_prime
        
        if i >= START_MAKING_PARTITIONS:
                partaddcount -= 1

        if i % update_freq == 0 and i >= START_QLEARN:
            agent.qlearn(replay_memory, partition_memory)

        if i % update_freq == 0 and i >= START_EELEARN and (i < END_EELEARN or int(args[5])):
            agent.imagecomparelearn(replay_memory)  

        if i % UPDATE_TARGETS_FREQUENCY == 0:
            agent.update_targets()

        if i % SAVE_NETWORKS_FREQUENCY == 0:
            agent.save_networks(logpath, i)

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
