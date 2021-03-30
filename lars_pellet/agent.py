from Qnetwork import Qnet, EEnet
import copy
import torch
import numpy as np

class Agent:
    def __init__(self, nq = 0.99, ne = 0.95):
        self.Qnet = Qnet()
        self.tQnet = copy.deepcopy(self.Qnet)
        self.EEnet = EEnet()
        self.tEEnet = copy.deepcopy(self.EEnet)
        self.visited = []
        self.NQ = nq
        self.NE = ne

    
    def find_action(self, state):
        action, policy = e_greedy_action_choice(state)
        return action, policy
    
    def update(replay_memory):
        QLEARN(replay_memory)
        EELEARN(replay_memory)

    def QLEARN(self, replay_memory):
        if len(replay_memory.memory) > replay_memory.batch_size:
            targ_onesteps = []

            #Sample random minibatch of transitions
            #{s; v; a; r; s ; v´} from replay memory
            batch = replay_memory.sample()
            pellet_rewards = []

            states, visited, action, _, reward, s_primes, visited_prime = zip(*batch)
            states = torch.cat(states)
            action = torch.cat(action).long()
            reward = torch.cat(reward)
            s_primes = torch.cat(s_primes)
            
            #if v 6= v0 then
            #r+  pellet reward for the partition visited
            #(i.e. the single partition in v0 n v)
            #else
            #r+   0
            #end if
            
            for i in range(replay_memory.batch_size):
                if visited[i] is not visited_prime[i]:
                    pellet_rewards.append(calc_pellet_reward(visited_prime[i][-1][1]))
                else:
                    pellet_rewards.append(0)
            pellet_rewards = torch.tensor(pellet_rewards)
            
            #targone-step   r + r+ + maxa Q(s0; v0; a)
            targ_onesteps = reward + pellet_rewards + Q_discount * self.TQnet(s_primes).max(1)[0].detach()
                
            #Calculate extrinsic and intrinsic returns, R and R+,
            #via the remaining history in the replay memory
            predictions = self.Qnet(states).gather(1, action)

            #targMC   R + R+
            #targmixed   (1 􀀀 Q)targone-step + QtargMC
            #Update Q(s; v; a) towards targmixed
            targ_mc = reward + pellet_rewards
            targ_mix = (1-self.NQ) * targ_onesteps + self.NQ*targ_mc
            self.Qnet.backpropagate(predictions, targ_mix)
    
    def EELEARN(self, replay_memory):
        if len(replay_memory.memory) > replay_memory.batch_size:
            #Sample a minibatch of state pairs and interleaving
            #auxiliary rewards fst; st+k; f^rt ; : : : ; ^rt+k􀀀1gg
            #from the replay memory with k < m
            batch = replay_memory.sampleEEminibatch()

            states, s_primes, smid, auxreward = zip(*batch)
            targ_onesteps = []
            for i in range(len(smid)):
            #targone-step   ^rt + Em(st+1; st+k􀀀1)
                targ_onesteps.append(torch.tensor(auxreward[i][0]).unsqueeze(0) + EE_discount * self.tEEnet(merge_states_for_comparason(smid[i],
                                                                                                                                        s_primes[i])))
            
            targ_mc = torch.zeros(len(auxreward),18)
            #targMC Pk􀀀1 i=0 i^rt+i
            for i, setauxreward in enumerate(auxreward):
                for j, r in enumerate(setauxreward):
                    targ_mc[i] = targ_mc[i] + EE_discount**(j+1) * torch.tensor(r).unsqueeze(0)

            #targmixed   (1 􀀀 E)targone-step + EtargMC
            targ_mix = (1-self.NE) * torch.cat(targ_onesteps) + self.NE * targ_mc

            merged = []
            for i in range(len(states)):
                merged.append(merge_states_for_comparason(states[i], s_primes[i]))
            #Update Em(st; st+k􀀀1) towards targmixed
            self.EEnet.backpropagate(EEagent(torch.cat(merged)), targ_mix)

    def find_current_partition(state, partition_memory):
        min_distance = np.Inf
        for partition in partition_memory:
            distance = self.distance_prime(state, partition, partition_memory)
            if distance < min_distance:
                min_distance = distance
                current_partition = partition
        
        visited = self.visited
        if current_partition not in self.visited
            self.visited.append(current_partition)
        
        return visited, self.visited, min_distance
    
    def e_greedy_action_choice(self, state, epsilon):
        policy = self.Qnet(state)
        if np.random.rand() > self.epsilon:
            action = torch.argmax(policy[0]).item()
        else:
            action = np.random.randint(1, selt.action_space)
        
        self.epsilon = slope * total_steps + intercept

        return action, policy
    
    def distance_prime(self, s1, s2, partition_memory):
        max_distance = np.NINF
        for partition in partition_memory:
            distance = distance(s1, s2, partition)
            if distance > max_distance:
                max_distance = distance
        return max_distance

    def distance(self, s1, s2, dfactor):
        return max(abs(self.EEnet(merge_states_for_comparason(dfactor, s1)) - self.EEnet(merge_states_for_comparason(dfactor, s2))),
                   abs(self.EEnet(merge_states_for_comparason(s1, dfactor)) - self.EEnet(merge_states_for_comparason(s2, dfactor))))

def calc_pellet_reward(visits):
    return ee_beta / math.sqrt(max(1, visits))

def merge_states_for_comparason(s1,s2):
    return torch.stack([s1,s2], dim=2).squeeze(0)

