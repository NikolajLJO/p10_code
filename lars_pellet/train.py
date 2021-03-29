import gym
import numpy as np
import torch
import math
from random import randrange, sample
from Qnetwork import Qnet, EEnet
from environment import create_atari_env

slope = -(1 - 0.05) / 1000000
intercept = 1
total_steps = 0
T_add = 10
nq = 0.1
ne = 0.99
ee_beta = 1
batch_size = 32
Q_discount = 0.99
EE_discount = 0.99
k = 10

def mainloop():
    epsilon = 1
    Dmax = 0
    memory = []
    memory_size = 100
    memory_replace_pointer = 0

    #resetgame()
    env = create_atari_env("MontezumaRevenge-v0")
    s, vitited_partitions = reset(env)
    Qagent = Qnet()
    EEagent = EEnet()
    
    R = [(s,0)] # usikkerhed i hvad {s_p1} er deres kildekode siger reward som kommer sammen med state som på de her tidspunkt virker til at være s_0
    t_partition = 0

    # foreach episode do
    while True:
        terminating = False
        while terminating is not True:
            #pi = epsilon-greedy policy derived from Q
            #select a according to pi(s)
            action, action_values, epsilon = e_greedy_action_choice(Qagent, s, epsilon)
            
            #Calculate aux. reward ^r as per Eq. 3
            auxreward = Calculateauxiliaryreward(action_values, action)
            #Take action a, observe r, s´
            s_n, reward, terminating, _ = env.step(action.item())
            
            #Determine the current partition
            #s_pc = argmin_{spi in R} d(s´; s_pi )
            s_pc = partitiondeterminarion(EEagent, s_n, R)
            
            #// Update the set of visited partitions
            #v´ = v U spc
            vitited_partitions_new = list(set().union(vitited_partitions, [s_pc])) 
            
            #// Update the best candidate according to the
            #// distance measure defined by Equation 6
            #if d(s´; spc ) > Dmax then
            #   spn+1   s´
            #   D_{max}   d(s´; spc )
            #end if
            if distance(EEagent, s_n, s_pc[0], R[0][0]) > Dmax:
                s_pn1 = s_n
                dmax = distance(EEagent, s_n, s_pc[0], R[0][0])

            #Store transition info {s v; a; ^r; r; s´; v}
            #in the replay memory
            memory, memory_replace_pointer = store_transition(s, vitited_partitions, action, auxreward, reward, s_n, vitited_partitions_new, memory, memory_size, memory_replace_pointer)
            
            #// Add a new rep. state every Tadd steps
            #tpartition   tpartition + 1
            #if tpartition > Tadd then
            #   R:add(~spn+1)
            #   Dmax   0
            #   tpartition   0
            #end if
            t_partition = t_partition + 1
            if t_partition > T_add:
                R.append((s_pn1,0))
                Dmax = 0
                t_partition = 0

            #QLEARN()
            #EELEARN()
            QLEARN(memory, batch_size, Qagent)
            EELEARN(memory, batch_size, EEagent, memory_replace_pointer)

            #s   s0
            #v   v0
            s = s_n
            vitited_partitions = vitited_partitions_new

        #Update all partitions’ visit counts based on v
        #RESET()
        R = updatepartitions(R, vitited_partitions)
        s, vitited_partitions = reset(env)

def reset(env):
    #Reset the game and set s equal to the initial state
    return env.reset(), []

def QLEARN(memory, batch_size, Qagent):
    if len(memory) > batch_size:
        targ_onesteps = []

        #Sample random minibatch of transitions
        #{s; v; a; r; s ; v´} from replay memory
        batch = sample(memory, batch_size)
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
        
        for i in range(batch_size): # TODO her kan RDN komme til at tage over
            if visited[i] is not visited_prime[i]:
                pellet_rewards.append(calc_pellet_reward(visited_prime[i][-1][1]))
            else:
                pellet_rewards.append(0)
        pellet_rewards = torch.tensor(pellet_rewards)
        
        #targone-step   r + r+ + maxa Q(s0; v0; a)
        targ_onesteps = reward + pellet_rewards + Q_discount * Qagent(s_primes).max(1)[0].detach() # TODO jeg tror at qagent her skal være en target agent
            
        #Calculate extrinsic and intrinsic returns, R and R+,
        #via the remaining history in the replay memory
        predictions = Qagent(states).gather(1, action)

        #targMC   R + R+
        #targmixed   (1 􀀀 Q)targone-step + QtargMC
        #Update Q(s; v; a) towards targmixed
        targ_mc = reward + pellet_rewards
        targ_mix = (1-nq) * targ_onesteps + nq*targ_mc
        backpropagate(predictions, targ_mix, Qagent)
    
def EELEARN(memory, batch_size, EEagent, memory_replace_pointer):
    if len(memory) > batch_size:
        #Sample a minibatch of state pairs and interleaving
        #auxiliary rewards fst; st+k; f^rt ; : : : ; ^rt+k􀀀1gg
        #from the replay memory with k < m
        batch = sampleEEminibatch(memory, batch_size, memory_replace_pointer)

        states, s_primes, smid, auxreward = zip(*batch)
        targ_onesteps = []
        for i in range(len(smid)):
        #targone-step   ^rt + Em(st+1; st+k􀀀1)
            targ_onesteps.append(torch.tensor(auxreward[i][0]).unsqueeze(0) + EE_discount * EEagent(merge_states_for_comparason(smid[i], s_primes[i])))#TODO jeg tror ogsp at EEagent skal være en target agent
        
        targ_mc = torch.zeros(len(auxreward),18)
        #targMC Pk􀀀1 i=0 i^rt+i
        for i, setauxreward in enumerate(auxreward):
            for j, r in enumerate(setauxreward):
                targ_mc[i] = targ_mc[i] + EE_discount**(j+1) * torch.tensor(r).unsqueeze(0)

        #targmixed   (1 􀀀 E)targone-step + EtargMC
        targ_mix = (1-ne) * torch.cat(targ_onesteps) + ne * targ_mc

        merged = []
        for i in range(len(states)):
            merged.append(merge_states_for_comparason(states[i], s_primes[i]))
        #Update Em(st; st+k􀀀1) towards targmixed
        backpropagate(EEagent(torch.cat(merged)), targ_mix, EEagent)

def e_greedy_action_choice(agent, state, epsilon):
    Qs = agent(state)
    if np.random.rand() > epsilon:
        action = qs.max(1)[1].view(1, 1).type(torch.int8)
    else:
        action = torch.tensor([[randrange(18)]], dtype=torch.int8)
     
    epsilon = slope * total_steps + intercept

    return action, Qs, epsilon

def Calculateauxiliaryreward(policy, aidx):
    aux = [0]*18
    policy = policy.squeeze(0)
    for i in range(len(aux)):
        if aidx == i:
            aux[i] = 1 - policy[i].item()
        else:
            aux[i] = -policy[i].item()
    return aux

def partitiondeterminarion(EEagent, s_n, R):
    mindist = [np.inf, None]
    for s_pi in R:
        dist = distance(EEagent, s_n, s_pi[0], R[0][0])
        if mindist[0] > dist:
            mindist[0] = dist
            mindist[1] = s_pi
    return mindist[1]

def store_transition(s, vitited_partitions, a, auxreward,reward, s_n, vitited_partitions_new, memory, memory_size, memory_replace_pointer):
    transition = [s, vitited_partitions, a, auxreward, torch.tensor([reward]), s_n, vitited_partitions_new]
    if len(memory) < memory_size:
        memory.append(transition)
    else:
        memory[memory_replace_pointer] = transition
        memory_replace_pointer = (memory_replace_pointer + 1) % memory_size
    
    return memory, memory_replace_pointer

def calc_pellet_reward(visits):
    return ee_beta / math.sqrt(max(1, visits))

def sampleEEminibatch(memory, batch_size, memory_replace_pointer):
    resbatch = []
    batch = sample(list(enumerate(memory)), batch_size)

    # this loop takes the elements in the batch and goes k elements forward to give the auxilaray calculations on what actions has been used
    # between state s_0 and state s_0+k
    for element in batch:
        if len(memory) >= element[0] + k and (memory_replace_pointer < element[0] or memory_replace_pointer >= element[0] + k):
            auxs = [element[1][3]]
            for i in range(1, k):
                aux = memory[element[0] + i][3]
                auxs.append(aux)
            resbatch.append([element[1][0], memory[element[0] + i][0], memory[element[0]+1][0], auxs])

    return resbatch

def backpropagate(predictions, targets, agent):
    agent.optimizer.zero_grad()
    loss = agent.loss(predictions, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
    agent.optimizer.step()

def merge_states_for_comparason(s1,s2):
    return torch.stack([s1,s2], dim=2).squeeze(0)

def distance_normalisation_calculation(action_usage):
    res = 0
    for action in action_usage.squeeze(0):
        res = action.item()**2
    return math.sqrt(res)

def distance(EEagent, s1, s2, dfactor):
    return max(distance_normalisation_calculation(EEagent(merge_states_for_comparason(dfactor, s1)) - EEagent(merge_states_for_comparason(dfactor, s2))),
               distance_normalisation_calculation(EEagent(merge_states_for_comparason(s1, dfactor)) - EEagent(merge_states_for_comparason(s2, dfactor))))

def updatepartitions(R, vitited_partitions):
    for i, r in enumerate(R):
        for vitited_partition in vitited_partitions:
            if torch.equal(r[0],vitited_partition[0]):
                R[i] = (r[0], r[1]+1)
                break
    return R

mainloop()