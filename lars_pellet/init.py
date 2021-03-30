
from environment import create_atari_env
from agent import Agent

from multiprocessing import Pool


def setup(env_name):

    env = create_atari_env(env_name)
    action_space = env.action_space
    agent = Agent(action_space=action_space)
    #replay_memory = ReplayMemory()
    opt = 1
    
    return action_space, agent, opt, env
