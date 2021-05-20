
from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory
from multiprocessing import Pool


def setup(env_name, RDN, qlearn_start, training_time):
    env = create_atari_env(env_name)
    agent = Agent(env.action_space, training_time, use_RND=bool(int(RDN)), qlearn_start=qlearn_start)
    replay_memory = ReplayMemory()
    
    return replay_memory, agent, env
