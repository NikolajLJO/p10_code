from environment import create_atari_env
from agent import Agent
from memory import ReplayMemory
import torch


def setup(env_name, should_use_rnd):
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	env = setup_env(env_name, device)
	agent = setup_agent(env.action_space, should_use_rnd)
	action_space = env.action_space
	#agent.action_space = action_space
	opt = 1

	return action_space, agent, opt, env


def setup_env(env_name, device):
	return create_atari_env(env_name, device)


def setup_agent(action_space, should_use_rnd):
	return Agent(action_space, should_use_rnd)
