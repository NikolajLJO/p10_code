
from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory
from multiprocessing import Pool


def setup(env_name):
    
    agent = Agent()
    env = create_atari_env(env_name, agent.device)
    action_space = env.action_space
    agent.action_space = action_space
    replay_memory = ReplayMemory()
    opt = 1
    
    return action_space, replay_memory, agent, opt, env
