from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory


def setup(env_name):

    env = setup_env(env_name)
    action_space = env.action_space
    agent = setup_agent(action_space)
    opt = 1
    
    return action_space, agent, opt, env


def setup_env(env_name):
    return create_atari_env(env_name)


def setup_agent(action_space):
    return Agent(action_space)
