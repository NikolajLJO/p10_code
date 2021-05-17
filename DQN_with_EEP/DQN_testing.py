import gym
import gym.wrappers.frame_stack as frame_stacking
import torch
import numpy as np
import csv
from Qlearning import QLearning
from os import listdir
from Logger import Logger


def test(path: str, epoch_steps=135000, game_length=4500, stack_count=4, render=False):
    """
    this function runs an agent in an envionment without training.

    Parameters:
    path(str): path to the folder conttaining models
    episode_num (int): number of games to let the agent run
    stack_count (int): number of stacked frames
    render (bool): do you want a visualisation
    """
    # instantiate enviro
    names = listdir(path)
    for name in names:
        splitname = name.split("_")
        if '.pickle' in name and int(splitname[1]) % 250000 == 0:
            env = gym.make(splitname[0])
            env = frame_stacking.FrameStack(env, stack_count)
            agent = Logger.load_model(path + "/" + name)
            rewards_plt = []
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            total_steps = 0
            highscore = np.NINF
            
            while True:
                if total_steps >= epoch_steps:
                    break

                done, steps, total_steps, total_reward = a_game(game_length, render, env,
                                                                total_steps, epoch_steps,
                                                                device, agent)
                
                if done or steps == game_length - 1:
                    if total_reward > highscore:
                        highscore = total_reward
                    rewards_plt.append(total_reward)
                    
            print("average score: |{0}|".format(str.rjust(str(np.average(rewards_plt)), 6)))
            with open(path + "/" + splitname[0] + "_" + splitname[2] + "_test.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([(int(splitname[1]) / 250000), np.average(rewards_plt), highscore])


def a_game(game_length, render, env, total_steps, epoch_steps, device, agent):
    # reset the environment
    state = QLearning.preprocess_states(env.reset(), device)
    total_reward = 0

    for steps in range(game_length):
        if render:
            env.render()

        # take an action
        if 0.05 > np.random.rand():
            take_action = env.action_space.sample()
        else:
            with torch.no_grad():
                take_action = torch.argmax(agent(state)[0]).item()
        s_prime, score, done, _ = env.step(take_action)
        s_prime = QLearning.preprocess_states(s_prime, device)

        total_steps += 1
        # if the agent has reached a done state print the score
        # and write any important data down
        if done or total_steps >= epoch_steps:
            break

        total_reward += score
        state = s_prime
    return done, steps, total_steps, total_reward
