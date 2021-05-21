from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory
import torch


def setup(env_name):
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	env = setup_env(env_name, device)
	agent = setup_agent(env.action_space)

	return agent, env


def setup_env(env_name, device):
	return create_atari_env(env_name, device)


def setup_agent(action_space):
	return Agent(action_space)
