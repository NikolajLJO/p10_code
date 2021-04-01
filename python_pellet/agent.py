import math

from Qnetwork import Qnet, EEnet
import copy
import torch
import numpy as np


class Agent:
    def __init__(self, action_space, nq=0.99, ne=0.95):
        self.Qnet = Qnet()
        self.targetQnet = copy.deepcopy(self.Qnet)
        self.EEnet = EEnet()
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
        self.action_space = action_space
        #self.cuda = torch.device('cuda')     # Default CUDA device

    def cast_to_gpu(tensors):
        for tensor in tensors:
            tensor = tensor.cuda()

    def find_action(self, state):
        action, policy = self.e_greedy_action_choice(state)
        return action, policy

    def update(self, replay_memory):
        self.qlearn(replay_memory)
        self.eelearn(replay_memory)

    def qlearn(self, replay_memory):
        if len(replay_memory.memory) > replay_memory.batch_size:
            targ_onesteps = []

            # Sample random minibatch of transitions
            # {s; v; a; r; s ; v´} from replay memory
            batch = replay_memory.sample()
            pellet_rewards = []

            states, action, visited, reward, terminating, s_primes, visited_prime, targ_mc = zip(*batch)
            states = torch.cat(states)
            action = torch.tensor(action).long().unsqueeze(0)
            reward = torch.tensor(reward)
            s_primes = torch.cat(s_primes)
            terminating = torch.tensor(terminating).long()
            targ_mc = torch.tensor(targ_mc)

            cast_to_gpu([states,action,reward,s_primes,terminating,targ_mc])

            # if v 6= v0 then
            # r+  pellet reward for the partition visited
            # (i.e. the single partition in v0 n v)
            # else
            # r+   0
            # end if
            
            for i in range(replay_memory.batch_size):
                if len(visited[i]) < len(visited_prime[i]):
                    # TODO correct parameter here for calc_pellet_reward
                    pellet_rewards.append(replay_memory.calc_pellet_reward(visited_prime[i][-1][1]))
                else:
                    pellet_rewards.append(0)
            pellet_rewards = torch.tensor(pellet_rewards)

            # targone-step   r + r+ + maxa Q(s0; v0; a)
            targ_onesteps = reward + pellet_rewards + self.Q_discount * self.targetQnet(s_primes).max(1)[0].detach() * (1 - terminating)

            # Calculate extrinsic and intrinsic returns, R and R+,
            # via the remaining history in the replay memory
            predictions = self.Qnet(states).gather(1, action)

            # targMC   R + R+
            # targmixed   (1 􀀀 Q)targone-step + QtargMC
            # Update Q(s; v; a) towards targmixed
            targ_mix = (1 - self.NQ) * targ_onesteps + self.NQ * targ_mc
            self.Qnet.backpropagate(predictions, targ_mix.unsqueeze(0))

    def eelearn(self, replay_memory):
        if len(replay_memory.memory) > replay_memory.batch_size:
            # Sample a minibatch of state pairs and interleaving
            # auxiliary rewards fst; st+k; f^rt ; : : : ; ^rt+k􀀀1gg
            # from the replay memory with k < m
            batch = replay_memory.sample_ee_minibatch()

            states, s_primes, smid, auxreward = zip(*batch)
            targ_onesteps = []
            for i in range(len(smid)):
                # targone-step   ^rt + Em(st+1; st+k􀀀1)
                targ_onesteps.append(
                    torch.tensor(auxreward[i][0])
                    + self.EE_discount
                    * self.targetEEnet(merge_states_for_comparason(smid[i], s_primes[i])))

            targ_mc = torch.zeros(len(auxreward), 18).cuda()
            # targMC Pk􀀀1 i=0 i^rt+i
            for i, setauxreward in enumerate(auxreward):
                for j, r in enumerate(setauxreward):
                    targ_mc[i] = targ_mc[i] + self.EE_discount ** (j + 1) * torch.tensor(r).unsqueeze(0).cuda()

            # targmixed   (1 􀀀 E)targone-step + EtargMC
            targ_mix = (1 - self.NE) * torch.cat(targ_onesteps).cuda() + self.NE * targ_mc

            merged = []
            for i in range(len(states)):
                merged.append(merge_states_for_comparason(states[i], s_primes[i]))
            # Update Em(st; st+k􀀀1) towards targmixed
            self.EEnet.backpropagate(self.EEnet(torch.cat(merged).cuda()), targ_mix)

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
    def e_greedy_action_choice(self, state):
        policy = self.Qnet(state)
        if np.random.rand() > self.epsilon:
            action = torch.argmax(policy[0]).item()
        else:
            action = np.random.randint(1, self.action_space.n)

        self.epsilon = self.slope * self.total_steps + self.intercept

        return action, policy

    def distance_prime(self, s1, s2, partition_memory):
        max_distance = np.NINF
        for partition in partition_memory:
            distance = self.distance(s1, s2, partition[0])
            if distance > max_distance:
                max_distance = distance
        return max_distance

    def distance(self, s1, s2, dfactor):
        return max(
            torch.sum(abs(self.EEnet(merge_states_for_comparason(dfactor, s1)) - self.EEnet(merge_states_for_comparason(dfactor, s2)))),
            torch.sum(abs(self.EEnet(merge_states_for_comparason(s1, dfactor)) - self.EEnet(merge_states_for_comparason(s2, dfactor))))).item()
    
    def update_targets(self):
        self.targetQnet = copy.deepcopy(self.Qnet)
        self.targetEEnet = copy.deepcopy(self.EEnet)


def merge_states_for_comparason(s1, s2):
    return torch.stack([s1, s2], dim=2).squeeze(0)

def is_tensor_in_list(mtensor, mlist):
    for element in mlist:
        if torch.equal(mtensor[0], element[0]):
            return True
    return False
