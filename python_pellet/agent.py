'''
In this file the agent and belonging methods can be found.
'''
import copy
import math
import torch
import numpy as np
from Qnetwork import Qnet, EEnet


class Agent:
    '''
    The agent is the primary object containing multiple neural networks
    and methods to use them
    '''
    def __init__(self, actionspace, training_time,  NQ=0.1, NE=0.1, use_RND=False, double_DQN = True, qlearn_start=2250000):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.q_net = Qnet(actionspace.n).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net)
        if use_RND:
            self.ee_net = EEnet(actionspace.n).to(self.device)
            self.target_ee_net = EEnet(actionspace.n).to(self.device)
        else:
            self.ee_net = EEnet(actionspace.n).to(self.device)
            self.target_ee_net = copy.deepcopy(self.ee_net)
        self.use_rnd = use_RND
        self.nq = NQ
        self.ne = NE
        self.qlearn_start = qlearn_start
        
        self.epsilon_start = 1
        self.epsilon = self.epsilon_start
        self.epsilon_end = 0.01
        self.epsilon_endt = training_time/10
        self.slope = -(1 - 0.05) / 1000000
        self.intercept = 1
        self.q_discount = 0.99
        self.ee_discount = 0.99
        self.action_space = actionspace

        temp_discounts = []
        temp_discounts.append(torch.tensor([self.ee_discount]*actionspace.n, device=self.device).unsqueeze(0))
        for i in range(1,100):
            temp_discounts.append(torch.tensor([self.ee_discount**(i+1)]*actionspace.n,
                                                device=self.device).unsqueeze(0))
        self.ee_discounts = torch.cat(temp_discounts)
        self.mse = torch.nn.MSELoss(reduction='none')

        self.steps_since_reward = 0
        self.non_reward_steps_before_full_eps = 500
        self.double = double_DQN
        self.maximum_pellet_reward = []
        for i in range(32):
            self.maximum_pellet_reward.append([0.1]*100)
        self.maximum_pellet_reward = torch.tensor(self.maximum_pellet_reward, device=self.device)

    def find_action(self, state, step, visited):
        '''
        this function finds and return an action
        Input: state is the current state and step is the stepcount
        output: the best action according to policy
                policy the full list of q-values
        '''
        action, policy = self.e_greedy_action_choice(state, step, visited)
        return action, policy

    def qlearn(self, replay_memory, partition_memory):
        '''
        qlearn calucaltes the variabels used for backpropagating the q-network
        Input: replay_memory to sample from
        '''
        if len(replay_memory.memory) > replay_memory.batch_size:
            targ_onesteps = []

            # Sample random minibatch of transitions
            # {s; v; a; r; s ; v´, mcr} from replay memory
            batch = replay_memory.sample()
            pellet_rewards = []

            # targ_mc is the target montecarlo reward, i. e.
            # the cummulative reward we can get from a state to the terminating state.
            states, action, visited, reward, terminating, s_primes, visited_prime, targ_mc = zip(*batch)
            states = torch.cat(states)
            action = torch.cat(action).long()
            reward = torch.cat(reward)
            s_primes = torch.cat(s_primes)
            terminating = torch.cat(terminating).long()
            targ_mc = torch.cat(targ_mc)
            visited = torch.cat(visited)
            visited_prime = torch.cat(visited_prime)        
            partitionreward = torch.zeros([len(visited),100], device=self.device)

            for i, partition in enumerate(partition_memory):
                partitionreward[0][i] = replay_memory.calc_pellet_reward(partition[1])
            for i in range(1,len(visited)):
                partitionreward[i] = partitionreward[0]
            partitionrewardprime  = copy.deepcopy(partitionreward)

            partitionreward[visited == 0] = 0
            partitionrewardprime[visited_prime == 0] = 0
            partitionreward= torch.min(torch.stack([self.maximum_pellet_reward, partitionreward], dim=1),dim=1)[0]
            partitionrewardprime= torch.min(torch.stack([self.maximum_pellet_reward, partitionrewardprime], dim=1),dim=1)[0]

            pellet_rewards = (torch.sum(partitionrewardprime, 1) - torch.sum(partitionreward, 1))

            # Represents the target reward for one step.
            # In contrast to targ_mc, we calculate the rest of the reward
            # by using our network target_q_net.
            if self.double:
                best_next_action = self.q_net(s_primes, partitionrewardprime).max(1)[1].detach().unsqueeze(1)
                future_reward = self.target_q_net(s_primes, partitionrewardprime).gather(1, best_next_action).squeeze(1) * (1 - terminating)
            else:
                future_reward = self.target_q_net(s_primes, partitionrewardprime).max(1)[0].detach() * (1 - terminating)
            targ_onesteps = reward + pellet_rewards + self.q_discount * future_reward

            # Calculate extrinsic and intrinsic returns, R and R+,
            # via the remaining history in the replay memory
            predictions = self.q_net(states, partitionreward).gather(1, action)

            targ_mix = (1 - self.nq) * targ_onesteps + self.nq * targ_mc
            self.q_net.backpropagate(predictions, targ_mix.unsqueeze(1))

    def eelearn(self, replay_memory):
        '''
        eelearn calucaltes the variabels used for backpropagating the ee-network
        Input: replay_memory to sample from
        '''
        if len(replay_memory.memory) > replay_memory.batch_size:
            # Sample a minibatch of state pairs and interleaving
            batch = replay_memory.sample_ee_minibatch()

            states, s_primes, smid, auxreward = zip(*batch)
            # targ_onesteps is a list of targets where the first step is what is done and future is calculated by a network
            targ_onesteps = []
            # merged is a list of all the merged states used to calculate future
            merged = []
            for i in range(len(smid)):
                targ_onesteps.append(auxreward[i][0])
                merged.append(merge_states_for_comparason(smid[i], s_primes[i]))
            
            merged = torch.cat(merged)
            targ_onesteps = torch.stack(targ_onesteps)
            future = self.ee_discount * self.target_ee_net(merged).detach()
            targ_onesteps = targ_onesteps + future

            targ_mc = torch.zeros(len(auxreward),self.action_space.n, device=self.device)

            for i, setauxreward in enumerate(auxreward):
                discounted_aux = torch.stack(setauxreward) + self.ee_discounts[:len(setauxreward)]
                targ_mc[i] = torch.sum(discounted_aux, 0)

            targ_mix = (1 - self.ne) * targ_onesteps + self.ne * targ_mc
            # merged is changed to contain a list of states calculating now and future.
            merged = []
            for i in range(len(states)):
                merged.append(merge_states_for_comparason(states[i], s_primes[i]))

            self.ee_net.backpropagate(self.ee_net(torch.cat(merged).to(device=self.device)),
                                      targ_mix)

    def rndlearn(self, replay_memory):
        '''
        rndlearn calucaltes the variabels used for backpropagating the q-network
        Input: replay_memory to sample from
        '''
        if len(replay_memory.memory) > replay_memory.batch_size:
            batch = replay_memory.sample_ee_minibatch()
            states, s_primes, _, _ = zip(*batch)
            netinput = []
            for i in range(len(states)):
                netinput.append(merge_states_for_comparason(states[i], s_primes[i]))

            netinput = torch.cat(netinput)
            pred = self.ee_net(netinput)
            targ = self.target_ee_net(netinput)

            self.ee_net.backpropagate(pred, targ)

    def imagecomparelearn(self, replay_memory):
        '''
        This function decides how to train the state compareson network
        Input: replay_memory is sent to the correct training algorithm
        '''
        if self.use_rnd:
            self.rndlearn(replay_memory)
        else:
            self.eelearn(replay_memory)

    def find_current_partition(self, state, partition_memory, visited):
        '''
        This function findc the partition the agent is currently in,
        and adding it to the list of visited partitions.
        Input: state should be the current state
               partition_memory is the list of all partitions
        output: visited is the list of visited partition before the transition
                visited_prime is the list of visited states after the transition
                distance is the maximum distance between the state and all partitions
        '''
        min_distance = np.Inf

        if self.use_rnd:
            merged_states_forward = []
            merged_states_back = []
            for s2 in partition_memory:
                merged_states_forward.append(merge_states_for_comparason(state, s2[0]))
                merged_states_back.append(merge_states_for_comparason(s2[0], state))
            merged_states_forward = torch.cat(merged_states_forward)
            merged_states_back = torch.cat(merged_states_back)
   
            aux_forward = self.ee_net(merged_states_forward)
            aux_target_forward = self.target_ee_net(merged_states_forward)
            
            aux_back = self.ee_net(merged_states_back)
            aux_target_back = self.target_ee_net(merged_states_back)
            
            aux_forward = torch.mean(self.mse(aux_forward,aux_target_forward), dim = 1)
            aux_back =  torch.mean(self.mse(aux_back,aux_target_back), dim = 1)

            novelties = torch.max(torch.stack([aux_forward,aux_back], dim=1),1)[0]

            argmin_value, argmin_index = torch.min(novelties,0)
            current_partition = partition_memory[argmin_index.item()]
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
                forward = torch.sum(self.ee_net(state_to_ref) - self.ee_net(s2_to_ref), dim=1)

                ref_to_state = torch.cat(ref_to_state)
                ref_to_s2 = torch.cat(ref_to_s2)
                backward = torch.sum(self.ee_net(ref_to_state)- self.ee_net(ref_to_s2), dim=1)

                max_distance = torch.max(torch.max(torch.stack([forward,backward], dim=1),1)[0],0)[0].item()

                if max_distance < min_distance:
                    min_distance = max_distance
                    current_partition = s2
                    index = i
                

        pre_visited = copy.deepcopy(visited)

        if "index" in locals() and visited[0][index].item() == 0:
            visited[0][index] = 1 / math.sqrt(max(1, partition_memory[index][1]))

        return pre_visited, visited, min_distance

    def e_greedy_action_choice(self, state, step, visited):
        '''
        This function chooses the agction greedily or randomly according to the epsilon value
        Input: state is the state you want an action for
               step is the current step used to calculate the epsilon for the next decisions
        Output: action a tensor with the best action index
                policy is all q-values at the state
        '''
        policy = self.q_net(state, visited)
        
        if self.steps_since_reward > self.non_reward_steps_before_full_eps:
            self.epsilon = self.epsilon_start
        elif step <= self.qlearn_start:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = self.epsilon_end + max(0, (self.epsilon_start - self.epsilon_end) * (self.epsilon_endt - max(0, step - self.qlearn_start)) / self.epsilon_endt)
        
        if np.random.rand() > self.epsilon:
            action = policy.max(1)[1].view(1, 1).type(torch.int8)
        else:
            action = torch.tensor([[self.action_space.sample()]], device=self.device).type(torch.int8)

        return action, policy

    def update_targets(self):
        '''
        This function updates the target networks
        If the agent is using RND the EE-target will not be updated.
        '''
        self.target_q_net = copy.deepcopy(self.q_net)
        if not self.use_rnd:
            self.target_ee_net = copy.deepcopy(self.ee_net)

    def save_networks(self, path,step):
        '''
        This function saves the networks parameters on the provided path
        using step to indicate when in the training the networks is from
        Input: path the path to the save location
               step the step the training is at
        '''
        torch.save(self.q_net.state_dict(), str(path) + "Qagent_"+ str(step) +".p")
        torch.save(self.ee_net.state_dict(), str(path) + "EEagent_"+ str(step) +".p")


def merge_states_for_comparason(s1, s2):
    '''
    this funktion combines two states
    Input: s1 is a state
           s2 is a state
    Output: A tensor containing the two states on the secound dimention
    '''
    return torch.stack([s1[0][0], s2[0][0]]).squeeze(0).unsqueeze(0)

def is_tensor_in_list(mtensor, mlist):
    '''
    This function tests if a tensor os in the list
    Input: mtensor is a tensor
           mlist is a list of tensors of same form as mtensor
    Output: boolean dependant on if the mtensor is in the list
    '''
    for element in mlist:
        if torch.equal(mtensor[0], element[0]):
            return True
    return False
