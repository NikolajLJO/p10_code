from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory


def setup(env_name):
	agent = setup_agent()
	env = setup_env(env_name, agent.device)
	action_space = env.action_space
	agent.action_space = action_space
	opt = 1

	return action_space, agent, opt, env


def setup_env(env_name, device):
	return create_atari_env(env_name, device)


def setup_agent():
	return Agent()
