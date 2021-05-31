import copy
import logging
import math
import time

import numpy as np
import torch

from Qnetwork import Qnet, EEnet


class Agent:
	def __init__(self, action_space, should_use_rnd, nq=0.1, ne=0.1):
		torch.multiprocessing.set_sharing_strategy('file_system')
		torch.set_num_threads(1)
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			logging.info("cuda")
		else:
			self.device = torch.device('cpu')
			logging.info("cpu")
		self.Qnet = Qnet(action_space).to(self.device)
		self.targetQnet = copy.deepcopy(self.Qnet)
		self.NQ = nq
		self.NE = ne

		self.epsilon_start = 1
		self.epsilon = self.epsilon_start
		self.epsilon_end = 0.01
		self.epsilon_endt = 1000000
		self.slope = -(1 - 0.05) / 1000000
		self.intercept = 1
		self.qlearn_start = 50000
		self.total_steps = 0
		self.Q_discount = 0.99
		self.EE_discount = 0.99
		self.action_space = action_space
		self.should_use_rnd = should_use_rnd
		self.mse = torch.nn.MSELoss(reduction='none')

		if should_use_rnd:
			self.EEnet = EEnet(action_space).to(self.device)
			self.targetEEnet = EEnet(action_space).to(self.device)
		else:
			self.EEnet = EEnet(action_space).to(self.device)
			self.targetEEnet = copy.deepcopy(self.EEnet)

		listt = []
		listt.append(torch.tensor([self.EE_discount] * 18, device=self.device).unsqueeze(0))
		for i in range(1, 100):
			listt.append(torch.tensor([self.EE_discount ** (i + 1)] * 18, device=self.device).unsqueeze(0))
		self.EE_discounts = torch.cat(listt)
		self.steps_since_reward = 0
		self.non_reward_steps_before_full_eps = 500

	def cast_to_device(self, tensors):
		for tensor in tensors:
			tensor = tensor.to(device=self.device)

	def find_action(self, state, step, visited, steps_since_reward):
		"""
		this function finds and return an action
		Input: state is the current state and step is the stepcount
		output: the best action according to policy
				policy the full list of q-values
		"""
		action, policy = self.e_greedy_action_choice(state, step, visited, steps_since_reward)
		return action, policy

	def update(self, replay_memory, ee_memory, ee_done: bool, should_use_rnd=False):
		logging.info("ql")
		pre_learn = time.process_time_ns()

		self.qlearn(replay_memory, batch_size=min(32, len(replay_memory.memory)))

		post_learn = time.process_time_ns()
		logging.info("q learned in: " + "%.2f" % ((post_learn - pre_learn) / 1e9))

		logging.info("eel")
		pre_learn = time.process_time_ns()

		if not should_use_rnd and not ee_done:
			self.eelearn(ee_memory, batch_size=min(32, len(ee_memory)))
		elif should_use_rnd:
			self.rndlearn(ee_memory, batch_size=min(32, len(ee_memory)))

		post_learn = time.process_time_ns()
		logging.info("ee learned in: " + "%.2f" % ((post_learn - pre_learn) / 1e9))


	def qlearn(self, replay_memory, batch_size=None):
		# Sample random minibatch of transitions
		# {s; v; a; r; s ; v´} from replay memoryfe
		if batch_size is None:
			batch_size = replay_memory.batch_size
		batch = replay_memory.sample(forced_batch_size=batch_size, should_pop=True)
		logging.info("batch q: " + str(len(batch)))
		states, action, visited, aux_reward, reward, terminating, s_primes, visited_prime, targ_mc, ee_thing = zip(
			*batch)
		states = torch.cat(states).to(self.device)
		action = torch.cat(action).long().unsqueeze(1).to(self.device)
		visited = torch.cat(visited).to(self.device)
		visited_prime = torch.cat(visited_prime).to(self.device)
		reward = torch.cat(reward).to(self.device)
		s_primes = torch.cat(s_primes).to(self.device)
		terminating = torch.cat(terminating).long().to(self.device)
		targ_mc = torch.cat(targ_mc).to(self.device)

		maximum_pellet_reward = []
		for i in range(len(states)):
			maximum_pellet_reward.append([0.1] * 100)
		maximum_pellet_reward = torch.tensor(maximum_pellet_reward, device=self.device)
		visited = torch.min(torch.stack([maximum_pellet_reward, visited], dim=1), dim=1)[0]
		visited_prime = torch.min(torch.stack([maximum_pellet_reward, visited_prime], dim=1), dim=1)[0]

		pellet_rewards = torch.sum(visited_prime, dim=1) - torch.sum(visited, dim=1)

		targ_onesteps = reward + pellet_rewards + self.Q_discount * self.targetQnet(s_primes, visited_prime).max(1)[
			0].detach() * (1 - terminating)
		# Calculate extrinsic and intrinsic returns, R and R+,
		# via the remaining history in the replay memory
		predictions = self.Qnet(states, visited).gather(1, action)

		targ_mix = (1 - self.NQ) * targ_onesteps + self.NQ * targ_mc

		self.Qnet.backpropagate(predictions, targ_mix.unsqueeze(1))
		del states
		del action
		del visited
		del visited_prime
		del reward
		del s_primes
		del terminating
		del targ_mc
		del batch
		del aux_reward
		del ee_thing
		del maximum_pellet_reward
		del predictions
		del targ_mix
		del targ_onesteps
		del pellet_rewards
		del replay_memory

	def eelearn(self, ee_memory, batch_size=1000):
		# Sample a minibatch of state pairs and interleaving
		# auxiliary rewards
		batch = []
		for _ in range(0, min(len(ee_memory), batch_size)):
			batch.append(ee_memory.pop())
		logging.info("batch e: " + str(len(batch)))
		states, s_primes, smid, auxreward = zip(*batch)
		states = torch.cat(states).to(self.device)
		s_primes = torch.cat(s_primes).to(self.device)
		smid = torch.cat(smid).to(self.device)

		targ_onesteps = []
		for i in range(len(smid)):
			targ_onesteps.append(
				auxreward[i][0].to(self.device)
				+ self.EE_discount
				* self.targetEEnet(merge_states_for_comparason(smid[i].unsqueeze(0), s_primes[i].unsqueeze(0))))

		targ_mc = torch.zeros(len(auxreward), self.action_space.n, device=self.device)
		for i, setauxreward in enumerate(auxreward):
			setauxreward = torch.stack(setauxreward).to(self.device)
			targ_mc[i] = torch.sum(setauxreward + self.EE_discounts[:len(setauxreward)], 0)

		# targmixed   (1 􀀀 E)targone-step + EtargMC
		targ_mix = (1 - self.NE) * torch.cat(targ_onesteps) + self.NE * targ_mc

		merged = []
		for i in range(len(states)):
			merged.append(merge_states_for_comparason(states[i].unsqueeze(0), s_primes[i].unsqueeze(0)))
		self.EEnet.backpropagate(self.EEnet(torch.cat(merged)), targ_mix)
		del states
		del s_primes
		del smid
		del auxreward
		del targ_mc
		del targ_mix
		del targ_onesteps
		del setauxreward
		del merged
		del batch
		del ee_memory

	def rndlearn(self, ee_memory, batch_size: int):
		"""
		rndlearn calucaltes the variabels used for backpropagating the q-network
		Input: replay_memory to sample from
		"""
		batch = []
		for _ in range(0, min(len(ee_memory), batch_size)):
			batch.append(ee_memory.pop())
		logging.info("batch rnd: " + str(len(batch)))
		states, s_primes, _, _ = zip(*batch)
		netinput = []
		for i in range(len(states)):
			netinput.append(merge_states_for_comparason(states[i], s_primes[i]))

		netinput = torch.cat(netinput).to(self.device)
		pred = self.EEnet(netinput)
		targ = self.targetEEnet(netinput)

		self.EEnet.backpropagate(pred, targ)
		del batch
		del netinput
		del pred
		del targ

	def find_current_partition(self, state, partition_memory, visited):
		"""
		This function findc the partition the agent is currently in,
		and adding it to the list of visited partitions.
		Input: state should be the current state
				partition_memory is the list of all partitions
		output: visited is the list of visited partition before the transition
				visited_prime is the list of visited states after the transition
				distance is the maximum distance between the state and all partitions
		"""
		with torch.no_grad():
			min_distance = np.Inf
			index = 0
			if self.should_use_rnd:
				merged_states_forward = []
				merged_states_back = []
				for s2 in partition_memory:
					merged_states_forward.append(merge_states_for_comparason(state, s2[0]))
					merged_states_back.append(merge_states_for_comparason(s2[0], state))
				merged_states_forward = torch.cat(merged_states_forward)
				merged_states_back = torch.cat(merged_states_back)

				aux_forward = self.EEnet(merged_states_forward)
				aux_target_forward = self.targetEEnet(merged_states_forward)

				aux_back = self.EEnet(merged_states_back)
				aux_target_back = self.targetEEnet(merged_states_back)

				aux_forward = torch.mean(self.mse(aux_forward, aux_target_forward), dim=1)
				aux_back = torch.mean(self.mse(aux_back, aux_target_back), dim=1)

				novelties = torch.max(torch.stack([aux_forward, aux_back], dim=1), 1)[0]

				argmin_value, argmin_index = torch.min(novelties, 0)
				index = argmin_index.item()
				min_distance = argmin_value.item()
			else:
				for i, s2 in enumerate(partition_memory):
					max_distance = np.NINF
					state_to_ref = []
					s2_to_ref = []
					ref_to_state = []
					ref_to_s2 = []
					for refrence in partition_memory[:5]:
						state_to_ref.append(merge_states_for_comparason(state, refrence[0]))
						s2_to_ref.append(merge_states_for_comparason(s2[0], refrence[0]))
						ref_to_state.append(merge_states_for_comparason(refrence[0], state))
						ref_to_s2.append(merge_states_for_comparason(refrence[0], s2[0]))

					state_to_ref = torch.cat(state_to_ref)
					s2_to_ref = torch.cat(s2_to_ref)
					forward = torch.sum(self.EEnet(state_to_ref) - self.EEnet(s2_to_ref), dim=1)

					ref_to_state = torch.cat(ref_to_state)
					ref_to_s2 = torch.cat(ref_to_s2)
					backward = torch.sum(self.EEnet(ref_to_state) - self.EEnet(ref_to_s2), dim=1)

					max_distance = torch.max(torch.max(torch.stack([forward, backward], dim=1), 1)[0], 0)[0].item()

					if max_distance < min_distance:
						min_distance = max_distance
						index = i

			visited_prime = copy.deepcopy(visited)

			if visited[0][index] is None:
				visited[0][index] = torch.tensor([partition_memory.calc_pellet_reward(partition_memory[index][1])],
												device=self.device)

			return visited, visited_prime, min_distance

	def e_greedy_action_choice(self, state, step, visited, steps_since_reward):
		"""
		This function chooses the agction greedily or randomly according to the epsilon value
		Input: state is the state you want an action for
				step is the current step used to calculate the epsilon for the next decisions
		Output: action a tensor with the best action index
				policy is all q-values at the state
		"""
		with torch.no_grad():
			policy = self.Qnet(state, visited)

			if self.steps_since_reward > self.non_reward_steps_before_full_eps:
				self.epsilon = self.epsilon_start
			elif step <= self.qlearn_start:
				self.epsilon = self.epsilon_start
			else:
				self.epsilon = self.epsilon_end + max(0.0, (self.epsilon_start - self.epsilon_end) * (
							self.epsilon_endt - max(0, step - self.qlearn_start)) / self.epsilon_endt)

			if np.random.rand() > self.epsilon:
				action = torch.argmax(policy[0])
			else:
				action = torch.tensor(self.action_space.sample(), device=self.device)

			return action.unsqueeze(0), policy

	def distance_prime(self, s1, s2, partition_memory):
		max_distance = np.NINF
		for partition in partition_memory:
			distance = self.distance(s1, s2, partition[0])
			if distance > max_distance:
				max_distance = distance
		return max_distance

	def distance(self, s1, s2, reference_point):
		with torch.no_grad():
			return max(
				torch.sum(
					abs(
						self.EEnet(merge_states_for_comparason(reference_point, s1)) -
						self.EEnet(merge_states_for_comparason(reference_point, s2)))),
				torch.sum(
					abs(
						self.EEnet(merge_states_for_comparason(s1, reference_point)) -
						self.EEnet(merge_states_for_comparason(s2, reference_point))))).item()

	def update_targets(self):
		self.targetQnet = copy.deepcopy(self.Qnet)
		self.targetEEnet = copy.deepcopy(self.EEnet)

	def save_networks(self, path, step):
		"""
		This function saves the networks parameters on the provided path
		using step to indicate when in the training the networks is from
		Input: path the path to the save location
				step the step the training is at
		"""
		torch.save(self.Qnet.state_dict(), str(path) + "Qagent_" + str(step) + ".p")
		torch.save(self.EEnet.state_dict(), str(path) + "EEagent_" + str(step) + ".p")


def calc_pellet_reward(ee_beta, visits):
	return ee_beta / math.sqrt(max(1, visits))


def merge_states_for_comparason(s1, s2):
	return torch.stack([s1, s2], dim=2).squeeze(0)


def is_tensor_in_list(mtensor, mlist):
	"""
	This function tests if a tensor os in the list
	Input: mtensor is a tensor
			mlist is a list of tensors of same form as mtensor
	Output: boolean dependant on if the mtensor is in the list
	"""
	for element in mlist:
		if torch.equal(mtensor[0], element[0]):
			return True
	return False
