from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory


def setup(env_name):

    env = create_atari_env(env_name)
    action_space = env.action_space
    agent = Agent(action_space=action_space)
    replay_memory = ReplayMemory()
    opt = 1
    
    return action_space, replay_memory, agent, opt, env
