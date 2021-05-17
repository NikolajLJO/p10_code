import copy
import random
import gym
import gym.wrappers.frame_stack as frame_stacking
import numpy as np
import torch
from torch._C import device
import torchvision.transforms as transforms
import Logger
import Model
import math
from itertools import count
from PER import Memory

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize((84, 84)),
                             transforms.Grayscale(num_output_channels=1)])

transform_to_image = transforms.ToPILImage()

class QLearning:
    def __init__(self, env_name, episode_steps=4500, stack_count=4, explore_steps=50000, double=True, dueling = False,
                 pertype="none", replay_batch_size=32, target_update_frequency=10000, max_memory=250000, gamma=0.99,
                 use_all_controls=False, update_frequency=4, max_steps=10000000, learning_rate=0.00025,
                 initial_eps=1.0, trained_eps=0.1, final_eps=0.01, eps_initial_frames=1000000, print_freq=20):
        """
        Main Q-Learning class

        Parameters:
        env_name (String): String containing the name of the game evironment
        episode_num (Integer): Number of episodes to play
        episode_steps (Integer): Amount of steps per episode
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = Logger.Logger()
        
        # Training lenth stats
        self.max_steps = max_steps
        self.episode_steps = episode_steps
        self.explore_steps = explore_steps  # Hyperparameter from DQN
        self.total_steps = 0
        self.print_freq = print_freq
        
        # Environment
        self.env = gym.make(env_name)
        self.env = frame_stacking.FrameStack(self.env, stack_count)

        # Networks
        state = self.preprocess_states(self.env.reset(), self.device)
        if dueling:
            self.model = Model.Dueling_DQN(state,18 if use_all_controls else self.env.action_space.n, 
                                           learning_rate=learning_rate).to(self.device)
        else:
            self.model = Model.DQN(state, self.env.action_space.n, learning_rate=learning_rate).to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.double = double
        self.qtrain = True
        self.min_part_for_qlearn = 5
        self.qn = 0.1

        self.ee_net = Model.EEnet(self.env.action_space.n).to(self.device)
        self.target_ee = copy.deepcopy(self.ee_net)
        self.ee_train_start = 250000
        self.ee_train_end = 2000000
        self.eetrain = True
        self.EE_TIME_SEP_CONSTANT_M = 100
        self.ee_discount = gamma
        temp_discounts = []
        for i in range(0,100):
            temp_discounts.append(torch.tensor([self.ee_discount**(i+1)]*self.env.action_space.n,
                                                device=self.device).unsqueeze(0))
        self.ee_discounts = torch.cat(temp_discounts)
        self.en = 0.1

        # Exploration factor
        self.eps_initial_frames = eps_initial_frames
        self.epsilon_start = initial_eps
        self.epsilon = initial_eps
        self.epsilon_end = trained_eps
        self.epsilon_endt = eps_initial_frames
        self.slope = -(initial_eps - trained_eps) / eps_initial_frames
        self.intercept = initial_eps - self.slope * self.explore_steps
        self.slope_2 = -(trained_eps - final_eps) / (self.max_steps - eps_initial_frames - self.explore_steps)
        self.intercept_2 = final_eps - self.slope_2 * self.max_steps
        self.non_reward_steps_before_full_eps = 500

        # Experience replay memory
        self.memory_size = max_memory
        self.memory_replace_pointer = 0
        if pertype == "prop":
            self.memory = Memory(self.memory_size)
        else:
            self.memory = []
        self.per = pertype
        
        self.partition_memory = []
        self.START_MAKING_PARTITIONS = 2000000
        self.MAX_PARTITIONS = 100
        self.PARTITION_ADD_TIME_MULT = 1.2
        self.partition_addition_step = 20000

        # Optimizer variables
        self.replay_batch_size = replay_batch_size
        self.discount = gamma
        self.target_update_frequency = target_update_frequency
        self.update_frequency = update_frequency
        

        print("trainer initialized on " + self.device)
        print(f"double = {double}, dueling = {dueling}")
        
        self.maximum_pellet_reward = []
        for i in range(replay_batch_size):
            self.maximum_pellet_reward.append([0.1]*self.MAX_PARTITIONS)
        self.maximum_pellet_reward = torch.tensor(self.maximum_pellet_reward, device=self.device)

    @staticmethod
    def preprocess_states(frames, device):
        preprocess = []

        for frame in frames:
            resize_state = np.array(resize(frame[34:34 + 160, :160]))
            #resize_state = np.array(resize(frame))
            resize_state = torch.tensor(resize_state, dtype=torch.uint8, device=device).unsqueeze(0)
            preprocess.append(resize_state)

        return torch.cat(preprocess).unsqueeze(0).to(device)
        # TODO måske man kunne vae en envionment fil til at optille invionment
        # og holde preprocess så vi ungår cirkulære dependencies

    def __replay(self):
        """
        Experience Replay code
        Updates the current network, using the target network
        Updates the target network
        to be equal to the current, after this code is run
        """
        if self.explore_steps > self.total_steps:
            return

        if self.total_steps % self.update_frequency == 0:
            if self.qtrain: 
                self.qtrain = False
                print("qtrain")
            if self.per == "prop":
                batch, ids, weights = self.memory.sample(self.replay_batch_size)
            else:
                batchindex = random.sample(range(0, len(self.memory)), self.replay_batch_size)
            batch = [self.memory[i] for i in batchindex]
            states, action, reward, s_prime, soft_terminating_state, pre_visited, visited, auxiliary_reward, mcr, time_to_term = zip(*batch)
            states = torch.cat(states)
            action = torch.cat(action).long()
            reward = torch.cat(reward)
            soft_terminating_state = torch.cat(soft_terminating_state)
            s_prime = torch.cat(s_prime)
            pre_visited = torch.cat(pre_visited)
            visited = torch.cat(visited)
            mcr = torch.cat(mcr).view(self.replay_batch_size,1)
            partitionrewardsingle = torch.zeros([100], device=self.device)
            partitionreward = torch.zeros([len(visited),100], device=self.device)

            for i, partition in enumerate(self.partition_memory):
                partitionrewardsingle[i] = 1 / math.sqrt(max(1, partition[1]))
            for i in range(1,len(visited)):
                partitionreward[i] = partitionrewardsingle
            partitionrewardprime  = copy.deepcopy(partitionreward)
            partitionreward[pre_visited == 0] = 0
            partitionrewardprime[visited == 0] = 0

            partitionreward= torch.min(torch.stack([self.maximum_pellet_reward, partitionreward], dim=1),dim=1)[0]
            partitionrewardprime= torch.min(torch.stack([self.maximum_pellet_reward, partitionrewardprime], dim=1),dim=1)[0]
            pellet_reward = (torch.sum(partitionrewardprime, 1) - torch.sum(partitionreward, 1))

            #TODO lars er nået til mcr for pellets (flyttet til save)
            '''
            for i, index in enumerate(batchindex):
                pellet_mcr = 0
                for i in range(self.memory[index][-1], 0, -1):
                    discounted_pellet_reward_pre = copy.deepcopy(partitionrewardsingle).unsqueeze(0)
                    discounted_pellet_reward_post = copy.deepcopy(partitionrewardsingle).unsqueeze(0)
                    discounted_pellet_reward_pre[self.memory[index+i][5] == 0] = 0
                    discounted_pellet_reward_post[self.memory[index+i][6] == 0] = 0
                    discounted_pellet_reward_pre = torch.min(torch.stack([self.maximum_pellet_reward[0].unsqueeze(0), discounted_pellet_reward_pre], dim=1),dim=1)[0]
                    discounted_pellet_reward_post = torch.min(torch.stack([self.maximum_pellet_reward[0].unsqueeze(0), discounted_pellet_reward_post], dim=1),dim=1)[0]
                    pellet_mcr = (torch.sum(discounted_pellet_reward_post, 1) - torch.sum(discounted_pellet_reward_pre, 1)) + self.discount * pellet_mcr
                mcr[i] += pellet_mcr
            '''

            if self.double:
                double_action = torch.argmax(self.model(s_prime, partitionrewardprime), 1).detach().view(-1, 1)
                q_values_next = self.target_model(s_prime, partitionrewardprime).gather(1, double_action).detach().squeeze(1)
            else:
                q_values_next = self.target_model(s_prime, partitionrewardprime).max(1)[0].detach()

            predictions = self.model(states, partitionreward).gather(1, action)
            target = (reward + pellet_reward + (self.discount * q_values_next) * (1 - soft_terminating_state)).unsqueeze(1)
            
            target = (1-self.qn) * target + self.qn * mcr

            if self.per == "prop":
                errors = torch.abs(predictions - target)
                [self.memory.update(ids[i], errors[i][0].item()) for i in range(self.replay_batch_size)]
                self.model.back_propegate2(predictions, target, torch.tensor(weights, device=self.device))
            else:
                self.model.back_propegate(predictions, target, self.logger)

    def __ee_replay(self):
        """
        Experience Replay code
        Updates the current network, using the target network
        Updates the target network
        to be equal to the current, after this code is run
        """
        if self.ee_train_start > self.total_steps or self.ee_train_end < self.total_steps:
            return

        if self.total_steps % self.update_frequency == 0:
            if self.eetrain: 
                self.eetrain = False
                print("eetrain")
            batch = self.ee_sample()
            states, s_prime, s_mid, aux = zip(*batch)
            
            targ_onesteps = []
            # merged is a list of all the merged states used to calculate future
            merged = []
            for i in range(len(s_mid)):
                targ_onesteps.append(aux[i][0])
                merged.append(merge_states_for_comparason(s_mid[i], s_prime[i]))
            
            merged = torch.cat(merged)
            targ_onesteps = torch.stack(targ_onesteps)
            future = self.ee_discount * self.target_ee(merged).detach()
            targ_onesteps = targ_onesteps + future

            targ_mc = torch.zeros(len(aux),self.env.action_space.n, device=self.device)
            for i, setauxreward in enumerate(aux):
                discounted_aux = torch.stack(setauxreward) + self.ee_discounts[:len(setauxreward)]
                targ_mc[i] = torch.sum(discounted_aux, 0)
            
            target = (1-self.en) * targ_onesteps + self.en * targ_mc
            merged = []
            for i in range(len(states)):
                merged.append(merge_states_for_comparason(states[i], s_prime[i]))
            # TODO lars har ikke noget MRC for denne del af læringen
            self.ee_net.backpropagate(self.ee_net(torch.cat(merged).to(device=self.device)),
                                      target)

    def __take_action(self, state, visited):
        """
        Decides if we should explore or exploit
        following epsilon and a random generator

        Parameter:
        state (Array): the current game state

        Return:
        action to take for the given state.
        This can either be a q table or a singular value
        """
        qs = self.model(state, visited)
        if self.epsilon > np.random.rand() or self.explore_steps > self.total_steps:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.int8)
        else:
            with torch.no_grad():
                action = qs.max(1)[1].view(1, 1).type(torch.int8)
        
        if self.steps_since_reward > self.non_reward_steps_before_full_eps:
            self.epsilon = self.epsilon_start
        elif  self.total_steps <= self.explore_steps:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = self.epsilon_end + max(0, (self.epsilon_start - self.epsilon_end) * (self.epsilon_endt - max(0, self.total_steps - self.explore_steps)) / self.epsilon_endt)
            
        '''
        if self.steps_since_reward > self.non_reward_steps_before_full_eps:
            self.epsilon = self.epsilon_start
        elif step <= self.qlearn_start:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = self.epsilon_end + max(0, (self.epsilon_start - self.epsilon_end) * (self.epsilon_endt - max(0, step - self.qlearn_start)) / self.epsilon_endt)
        '''
        
        auxiliary_reward = torch.tensor(self.calculate_auxiliary_reward(qs, action.item()),
                                        device=self.device)

        s_prime, score, done, info = self.env.step(action.item())
        s_prime = self.preprocess_states(s_prime, self.device)
        
        return s_prime, score, done, info, action, auxiliary_reward

    def run(self, no_op_steps=10):
        """
        Runs the game for episode_num amount of times
        The AI has episode_step amount of steps per game.
        """
        scores = []
        average = 0
        soft_terminating_state = False
        partition_candidate = None
        dmax = 0
        partaddcount = 0
        
        for episode_num in count():
            if self.total_steps > self.max_steps:
                break

            state = self.preprocess_states(self.env.reset(), self.device)
            lives = self.env.ale.lives()
            total_score = 0
            visited = torch.zeros([1,100], device=self.device)
            episode_buffer = []
            self.steps_since_reward = 0

            for _ in range(self.episode_steps):
                if self.total_steps > self.max_steps:
                    break

                state = self.take_noop(state, soft_terminating_state, no_op_steps)

                # take an action depending on #explore_steps
                s_prime, score, done, info, taken_action, auxiliary_reward = self.__take_action(state, visited)

                # if the score changed this step calculate a reward and saves the result in a tensor
                reward = torch.tensor([self.calculate_reward(score)], device=self.device)

                # if the agent lose a lives it get a negative reward
                # and soft resets the game
                soft_terminating_state = done
                if info['ale.lives'] < lives:
                    reward -= 1
                    lives = info['ale.lives']
                    soft_terminating_state = True
                soft_terminating_state = torch.tensor([soft_terminating_state], device=self.device, dtype=torch.long)

                if self.total_steps % 1 == 0:
                    pre_visited, visited, distance = self.find_current_partition(s_prime, visited)

                if not soft_terminating_state and self.total_steps >= self.START_MAKING_PARTITIONS and distance > dmax:
                    partition_candidate = s_prime[0][0].unsqueeze(0).unsqueeze(0)
                    dmax = distance

                episode_buffer.append([soft_terminating_state, reward, s_prime, state, taken_action, pre_visited, visited, auxiliary_reward])

                # if we are done exploring use replay memory
                self.__replay()
                self.__ee_replay()

                if (partaddcount == 0 or self.total_steps == self.START_MAKING_PARTITIONS) and partition_candidate is not None:
                    self.partition_memory.append([partition_candidate, 0])
                    dmax = 0
                    distance = 0

                    self.partition_memory = self.partition_memory[-self.MAX_PARTITIONS:]

                    self.partition_addition_step = int(self.partition_addition_step * self.PARTITION_ADD_TIME_MULT)
                    partaddcount = self.partition_addition_step
                    print("partition made at: " + str(self.total_steps) + " next partition comes at:" + str(self.partition_addition_step + self.total_steps))
                    transform_to_image(partition_candidate[0].cpu()).save(self.logger.foldername + "/partition_" +
                                                        str(len(self.partition_memory)) + ".png")

                # update taget model after self.target_update_frequency steps
                if self.total_steps % self.target_update_frequency == 0:
                    self.target_model = copy.deepcopy(self.model)
                    self.target_ee = copy.deepcopy(self.ee_net)

                if self.total_steps % 250000 == 0:
                    self.logger.save_model(self.model, self.env.spec.id + "_" + str(self.total_steps))
                
                if self.total_steps == self.ee_train_end:
                    self.memory = []
                    self.memory_replace_pointer = 0

                if reward != 0 or not torch.equal(pre_visited, visited):
                    self.steps_since_reward = 0
                else:
                    self.steps_since_reward += 1
                
                partaddcount -= 1
                total_score += score
                self.total_steps += 1

                # If an action leads to the game terminating the gamescore is saved
                if done:
                    break
                state = s_prime

            # store the experience in replay memory
            if self.total_steps < self.ee_train_end or len(self.partition_memory) >= self.min_part_for_qlearn:
                self.store_experience(episode_buffer)
            
            self.update_partitions(visited)
            
            scores.append(total_score)
            if len(scores) >= 100:
                average = np.average(scores[-100:])

            self.printnlog(total_score, episode_num, average)

    def store_experience(self, episode_buffer):
        #soft_terminating_state, reward, s_prime, state, taken_action, pre_visited, visited, auxiliary_reward = zip(*episode_buffer)
        experiences = []
        mcr = 0
        time_to_term = 0
        for soft_terminating_state, reward, s_prime, state, taken_action, pre_visited, visited, auxiliary_reward in reversed(episode_buffer):
            if soft_terminating_state:
                time_to_term = 0
                mcr = 0

            discounted_pellet_reward_pre = copy.deepcopy(pre_visited)
            discounted_pellet_reward_post = copy.deepcopy(visited)
            discounted_pellet_reward_pre = torch.min(torch.stack([self.maximum_pellet_reward[0].unsqueeze(0), discounted_pellet_reward_pre], dim=1),dim=1)[0]
            discounted_pellet_reward_post = torch.min(torch.stack([self.maximum_pellet_reward[0].unsqueeze(0), discounted_pellet_reward_post], dim=1),dim=1)[0]
            pellet_reward = (torch.sum(discounted_pellet_reward_post, 1) - torch.sum(discounted_pellet_reward_pre, 1))

            mcr = (reward + pellet_reward) + self.discount * mcr
            experience = state, taken_action, reward, s_prime, soft_terminating_state, pre_visited, visited, auxiliary_reward, mcr, time_to_term
            experiences.append(experience)
            time_to_term += 1

        for experience in reversed(experiences):
            if self.per == "prop":
                error = self.calc_TDerror(experience)
                self.memory.add(error, experience)
            else:
                if len(self.memory) < self.memory_size:
                    self.memory.append(experience)
                else:
                    self.memory[self.memory_replace_pointer] = experience
                    self.memory_replace_pointer = (self.memory_replace_pointer + 1) % self.memory_size
    
    def calc_TDerror(self, experience):
        prediction = self.model(experience[0]).gather(1, experience[1].long())
        q_values_next = self.target_model(experience[3]).max(1)[0].detach()
        target = (experience[2] + (self.discount * q_values_next) * (1 - experience[4])).unsqueeze(1)
        return abs((prediction - target).item())

    @staticmethod
    def calculate_reward(score):
        if score == 0.0:
            return 0
        else:
            return 1 if score > 0 else -1

    def take_noop(self, state, terminal, no_op_steps):
        if terminal:
            for _ in range(random.randint(1, no_op_steps)):
                state, _, _, _ = self.env.step(0)  # 1 is fire action, 0 should be noop
            state = self.preprocess_states(state, self.device)
        return state

    def printnlog(self, total_score, episode_num, average):
        if episode_num % self.print_freq == 0:
            print("score: |{0}| Epsilon: |{1}| Episode: |{2}| Steps: |{3}| Average: |{4}|".format(
                str.rjust(str(total_score), 6),
                str.rjust(str(round(self.epsilon, 4)), 6),
                str.rjust(str(episode_num), 4),
                str.rjust(str(self.total_steps), 8),
                str.rjust(str(average), 4)))
        self.logger.save_to_file([str(total_score), str(round(self.epsilon, 4)), str(episode_num), str(self.total_steps),
                                  str(average)], "trainingdata")


    def find_current_partition(self, state, visited):
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
        current_partition = None
        with torch.no_grad():
            for i, s2 in enumerate(self.partition_memory):
                max_distance = np.NINF
                state_to_ref = []
                s2_to_ref = []
                ref_to_state = []
                ref_to_s2 = []
                for refrence in self.partition_memory[:5]:
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
            visited[0][index] = 1 / math.sqrt(max(1, self.partition_memory[index][1]))

        return pre_visited, visited, min_distance

    def calculate_auxiliary_reward(self, policy, aidx):
        '''
        this function calcules the the auxillary reward for each action
        Input: policy is the calculated values for each action
            aidx is the index of the used action
        output: a list of auxillary rewards for the actions in the action space
        '''
        aux = [0]*(policy.size()[1])
        policy = policy.squeeze(0)
        for i in range(len(aux)):
            if aidx == i:
                aux[i] = 1 - policy[i].item()
            else:
                aux[i] = -policy[i].item()
        return aux

    def ee_sample(self):
        batch = []
        for i in range(self.replay_batch_size):
            state_index = np.random.randint(0, (len(self.memory)))
            while self.memory[state_index][-1] <= 2:
                state_index = np.random.randint(0, (len(self.memory)))
            
            offset = np.random.randint(1, self.memory[state_index][-1])
            offset = min(offset, self.EE_TIME_SEP_CONSTANT_M)

            state_prime_index = (state_index + offset) % self.memory_size

            aux = []
            for j in range(offset):
                auxiliary_reward = self.memory[(state_index + j) % self.memory_size][7]
                aux.append(auxiliary_reward)

            batch.append([self.memory[state_index][0],
                          self.memory[state_prime_index][0],
                          self.memory[state_index][3],
                          aux])
        return batch
    
    def update_partitions(self,visited_partitions):
        '''
        this function updates the number of visits that
        each partition has when an episode is terminatied
        Input: visited_partitions is the list of visited partitions in an episode
            partition_memory is the full partition memory
        '''
        for i, visited in enumerate(visited_partitions[0]):
            if visited != 0:
                self.partition_memory[i][1] += 1

def merge_states_for_comparason(s1, s2):
    '''
    this funktion combines two states
    Input: s1 is a state
           s2 is a state
    Output: A tensor containing the two states on the secound dimention
    '''
    return torch.stack([s1[0][0], s2[0][0]]).squeeze(0).unsqueeze(0)