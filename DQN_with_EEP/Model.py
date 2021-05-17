import torch
import torch.nn.functional as f
import copy


def conv2d_size_out(height: int, width: int, stride: int, kernel_size: int, padding=0):
    result_1 = ((height - kernel_size + 2 * padding) / stride) + 1
    result_2 = ((width - kernel_size + 2 * padding) / stride) + 1
    return result_1, result_2


class DQN(torch.nn.Module):
    def __init__(self, initial_images, action_space, conv_1_stride=4, conv_2_stride=2, conv_3_stride=1,
                 conv_1_kernel=8, conv_2_kernel=4, conv_3_kernel=3, conv_1_channels=32, conv_2_channels=64,
                 conv_3_channels=64, learning_rate=0.0001):
        """
        DQN CNN.
        Parameters:
        action_space (int): Integer value of amount of actions
        """
        super(DQN, self).__init__()

        channels, initial_width, initial_height, = initial_images[0].shape

        # conv_3_channels needs replacement if conv layer setup changes
        self.conv_1 = torch.nn.Conv2d(channels, conv_1_channels, conv_1_kernel, stride=conv_1_stride)
        self.conv_2 = torch.nn.Conv2d(conv_1_channels, conv_2_channels, conv_2_kernel, stride=conv_2_stride)
        self.conv_3 = torch.nn.Conv2d(conv_2_channels, conv_3_channels, conv_3_kernel, stride=conv_3_stride)

        height, width = conv2d_size_out(initial_height, initial_width, conv_1_stride, conv_1_kernel)
        height, width = conv2d_size_out(height, width, conv_2_stride, conv_2_kernel)
        height, width = conv2d_size_out(height, width, conv_3_stride, conv_3_kernel)

        self.layer_node_count = int(height * width * conv_3_channels)
        self.lay1 = torch.nn.Linear(self.layer_node_count + 100, 512)
        self.lay2 = torch.nn.Linear(512, action_space)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

    def back_propegate(self, prediction, target, logger):
        """
        Back propagation
        Parameters:
        prediction (array): Prediction from the agent
        target (Integer): Action taken in the state
        """
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()
        # logger.save_to_file([loss.item(), prediction.tolist(), target.tolist()], "backwardsLog")

    def back_propegate2(self, prediction, target, weights):
        """
        Back propagation

        Parameters:
        prediction (array): Prediction from the agent
        target (Integer): Action taken in the state
        """
        self.optimizer.zero_grad()
        loss = (self.loss(prediction, target) * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()

    def forward(self, state, visited):
        """
        Forward function, given a state will forward propagate the network
        Parameters:
        state (array): Contains a array of the current state.
        Returns
        q_table (array): Each actions given q value
        """
        state = state.float() / 255
        state = f.relu(self.conv_1(state))
        state = f.relu(self.conv_2(state))
        state = f.relu(self.conv_3(state))
        state = f.relu(self.lay1(torch.cat([state.view(state.shape[0], -1), visited],1)))
        qvalues = self.lay2(state)
        return qvalues


class Dueling_DQN(torch.nn.Module):
    def __init__(self, action_space, initial_images, conv_1_stride=4, conv_2_stride=2, conv_3_stride=1,
                 conv_1_kernel=8, conv_2_kernel=4, conv_3_kernel=3, conv_1_channels=32, conv_2_channels=64,
                 conv_3_channels=64, learning_rate=0.0001):
        """
        DQN CNN.
        Parameters:
        action_space (int): Integer value of amount of actions
        """
        super(Dueling_DQN, self).__init__()

        channels, initial_width, initial_height, = initial_images[0].shape

        # conv_3_channels needs replacement if conv layer setup changes
        self.conv_1 = torch.nn.Conv2d(channels, conv_1_channels, conv_1_kernel, stride=conv_1_stride)
        self.conv_2 = torch.nn.Conv2d(conv_1_channels, conv_2_channels, conv_2_kernel, stride=conv_2_stride)
        self.conv_3 = torch.nn.Conv2d(conv_2_channels, conv_3_channels, conv_3_kernel, stride=conv_3_stride)

        height, width = conv2d_size_out(initial_height, initial_width, conv_1_stride, conv_1_kernel)
        height, width = conv2d_size_out(height, width, conv_2_stride, conv_2_kernel)
        height, width = conv2d_size_out(height, width, conv_3_stride, conv_3_kernel)

        self.layer_node_count = int(height * width * conv_3_channels)

        self.advantage_layer_1 = torch.nn.Linear(self.layer_node_count, 512)
        self.value_layer_1 = torch.nn.Linear(self.layer_node_count, 512)

        self.advantage_layer_2 = torch.nn.Linear(512, action_space)
        self.value_layer_2 = torch.nn.Linear(512, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

    def back_propegate(self, prediction, target, logger):
        """
        Back propagation

        Parameters:
        prediction (array): Prediction from the agent
        target (Integer): Action taken in the state
        """
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()
        # logger.save_to_file([loss.item(), prediction.tolist(), target.tolist()], "backwardsLog")

    def back_propegate2(self, prediction, target, weights):
        """
        Back propagation

        Parameters:
        prediction (array): Prediction from the agent
        target (Integer): Action taken in the state
        """
        self.optimizer.zero_grad()
        loss = (self.loss(prediction, target) * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()

    def forward(self, state):
        """
        Forward function, given a state will forward propagate the network

        Parameters:
        state (array): Contains a array of the current state.

        Returns
        q_table (array): Each actions given q value
        """
        state = state.float() / 255
        state = f.relu(self.conv_1(state))
        state = f.relu(self.conv_2(state))
        state = f.relu(self.conv_3(state))

        advantage_stream = f.relu(self.advantage_layer_1(state.view(state.shape[0], -1)))
        value_stream = f.relu(self.value_layer_1(state.view(state.shape[0], -1)))

        advantage = self.advantage_layer_2(advantage_stream)
        value = self.value_layer_2(value_stream)
        qvalue = value + (advantage - (torch.mean(advantage, 1, keepdim=True)))
        return qvalue


class EEnet(torch.nn.Module):
    def __init__(self, actionspace):
        super(EEnet, self).__init__()
        
        # conv_3_channels needs replacement if conv layer setup changes
        self.conv_net = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, stride=4),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(16, 16, 4, stride=2),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(16, 16, 3, stride=1),
                                            torch.nn.ReLU()
                                            )
        self.parallel_conv_net = copy.deepcopy(self.conv_net)

        height, width = conv2d_size_out(84, 84, 4, 8)
        height, width = conv2d_size_out(height, width, 2, 4)
        height, width = conv2d_size_out(height, width, 1, 3)

        layer_node_count = int(height * width * 16)

        self.linear_net = torch.nn.Sequential(torch.nn.Linear(layer_node_count*2, 128),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(128, actionspace))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0000625)
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, state):
        state = state.float() / 255
        state1, state2 = torch.split(state, 1, 1)
        state1, state2 = self.conv_net(state1), self.parallel_conv_net(state2)
        
        sub = state1-state2
        mean = (state1+state2)/2
        state = torch.cat([sub.view(sub.shape[0], -1), mean.view(mean.shape[0], -1)], 1)

        output = self.linear_net(state)
        return output

    def backpropagate(self, prediction, target):
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()