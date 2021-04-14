import math

from Qnetwork import Qnet, EEnet
import copy
import torch
import numpy as np
import random


class Agent:
    def __init__(self, nq=0.1, ne=0.1):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("cuda")
        else:
            self.device = torch.device('cpu')
            print("cpu")
        self.Qnet = Qnet().to(self.device)
        self.targetQnet = copy.deepcopy(self.Qnet)
        self.EEnet = EEnet().to(self.device)
        self.targetEEnet = copy.deepcopy(self.EEnet)
        self.visited = []
        self.NQ = nq
        self.NE = ne

        self.epsilon = 0
        self.slope = -(1 - 0.05) / 1000000
        self.intercept = 1
        self.total_steps = 0
        self.Q_discount = 0.99
        self.EE_discount = 0.99
        self.action_space = 0

        listt= []
        listt.append(torch.tensor([self.EE_discount]*18, device = self.device).unsqueeze(0))
        for i in range(1,100):
            listt.append(torch.tensor([self.EE_discount**(i+1)]*18, device = self.device).unsqueeze(0))
        self.EE_discounts = torch.cat(listt)

    def cast_to_device(self, tensors):
        for tensor in tensors:
            tensor = tensor.to(device=self.device)

    def find_action(self, state, step):
        action, policy = self.e_greedy_action_choice(state, step)
        return action, policy

    def update(self, replay_memory, ee_memory):
        self.qlearn(replay_memory)
        self.eelearn(ee_memory)

    def qlearn(self, replay_memory):
        if len(replay_memory.memory) > replay_memory.batch_size:
            # Sample random minibatch of transitions
            # {s; v; a; r; s ; v´} from replay memoryfe
            batch = replay_memory.sample()
            pellet_rewards = []
            states, action, visited, aux_reward, reward, terminating, s_primes, visited_prime, targ_mc, ee_thing = zip(*batch)
            states = torch.cat(states)
            action = torch.cat(action).long().unsqueeze(1)
            reward = torch.cat(reward)
            s_primes = torch.cat(s_primes)
            terminating = torch.cat(terminating).long()
            targ_mc = torch.cat(targ_mc)

            for i in range(replay_memory.batch_size):
                if len(visited[i]) < len(visited_prime[i]):
                    # TODO 1 replaced with self.ee_beta correct parameter here for calc_pellet_reward
                    pellet_rewards.append(calc_pellet_reward(1, visited_prime[i][-1][1]))
                else:
                    pellet_rewards.append(0)
            pellet_rewards = torch.tensor(pellet_rewards, device=self.device)

            targ_onesteps = reward + pellet_rewards + self.Q_discount * self.targetQnet(s_primes).max(1)[0].detach() * (1 - terminating)
            # Calculate extrinsic and intrinsic returns, R and R+,
            # via the remaining history in the replay memory
            predictions = self.Qnet(states).gather(1, action)

            targ_mix = (1 - self.NQ) * targ_onesteps + self.NQ * targ_mc
            self.Qnet.backpropagate(predictions, targ_mix.unsqueeze(1))

    def eelearn(self, ee_memory):
        if len(ee_memory) > 32:
            # Sample a minibatch of state pairs and interleaving
            # auxiliary rewards
            batch = random.sample(ee_memory, 32)
            states, s_primes, smid, auxreward = zip(*batch)
            targ_onesteps = []
            for i in range(len(smid)):
                targ_onesteps.append(
                    torch.tensor(auxreward[i][0])
                    + self.EE_discount
                    * self.targetEEnet(merge_states_for_comparason(smid[i], s_primes[i])))

            targ_mc = torch.zeros(len(auxreward), 18)
            for i, setauxreward in enumerate(auxreward):
                for j, r in enumerate(setauxreward):
                    targ_mc[i] = targ_mc[i] + self.EE_discount ** (j + 1) * torch.tensor(r).unsqueeze(0)
            targ_mix = (1 - self.NE) * torch.cat(targ_onesteps) + self.NE * targ_mc

            merged = []
            for i in range(len(states)):
                merged.append(merge_states_for_comparason(states[i], s_primes[i]))
            self.EEnet.backpropagate(self.EEnet(torch.cat(merged)), targ_mix)

    def find_current_partition(self, state, partition_memory):
        current_partition = None
        min_distance = np.Inf
        for partition in partition_memory:
            distance = self.distance_prime(state, partition[0], partition_memory)
            if distance < min_distance:
                min_distance = distance
                current_partition = partition

        visited = copy.deepcopy(self.visited)
        
        if not is_tensor_in_list(current_partition, self.visited):
            self.visited.append(current_partition)

        return visited, self.visited, min_distance

    def e_greedy_action_choice(self, state, step):
        policy = self.Qnet(state)
        if np.random.rand() > self.epsilon:
            action = torch.argmax(policy[0])
        else:
            action = torch.tensor(np.random.randint(1, self.action_space.n), device=self.device)

        self.epsilon = self.slope * step + self.intercept

        return action.unsqueeze(0), policy

    def distance_prime(self, s1, s2, partition_memory):
        max_distance = np.NINF
        for partition in partition_memory:
            distance = self.distance(s1, s2, partition[0])
            if distance > max_distance:
                max_distance = distance
        return max_distance

    def distance(self, s1, s2, reference_point):
        return max(
            torch.sum(abs(self.EEnet(merge_states_for_comparason(reference_point, s1)) - self.EEnet(merge_states_for_comparason(reference_point, s2)))),
            torch.sum(abs(self.EEnet(merge_states_for_comparason(s1, reference_point)) - self.EEnet(merge_states_for_comparason(s2, reference_point))))).item()
    
    def update_targets(self):
        self.targetQnet = copy.deepcopy(self.Qnet)
        self.targetEEnet = copy.deepcopy(self.EEnet)

    def save_networks(self, path, step):
        torch.save(self.Qnet.state_dict(), str(path) + "/logs/" + "Qagent_" + str(step) + ".p")
        torch.save(self.EEnet.state_dict(), str(path) + "/logs/" + "EEagent_" + str(step) + ".p")


def calc_pellet_reward(ee_beta, visits):
    return ee_beta / math.sqrt(max(1, visits))


def merge_states_for_comparason(s1, s2):
    return torch.stack([s1, s2], dim=2).squeeze(0)


def is_tensor_in_list(mtensor, mlist):
    for element in mlist:
        if torch.equal(mtensor[0], element[0]):
            return True
    return False
