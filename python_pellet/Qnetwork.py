import torch
import torch.nn.functional as functional


def conv2d_size_out(height: int, width: int, stride: int, kernel_size: int, padding=0):
    result_1 = ((height - kernel_size + 2 * padding) / stride) + 1
    result_2 = ((width - kernel_size + 2 * padding) / stride) + 1
    return result_1, result_2


class Qnet(torch.nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        # conv_3_channels needs replacement if conv layer setup changes
        self.conv_1 = torch.nn.Conv2d(1, 32, 8, stride=4)
        self.conv_2 = torch.nn.Conv2d(32, 64, 4, stride=2)
        self.conv_3 = torch.nn.Conv2d(64, 64, 3, stride=1)

        height, width = conv2d_size_out(84, 84, 4, 8)
        height, width = conv2d_size_out(height, width, 2, 4)
        height, width = conv2d_size_out(height, width, 1, 3)

        self.layer_node_count = int(height * width * 64)
        self.lay1 = torch.nn.Linear(self.layer_node_count, 512)
        self.lay2 = torch.nn.Linear(512, 18)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, state):
        state = state.float() / 255
        state = functional.relu(self.conv_1(state))
        state = functional.relu(self.conv_2(state))
        state = functional.relu(self.conv_3(state))
        state = functional.relu(self.lay1(state.view(state.shape[0], -1)))
        qvalues = self.lay2(state)
        return qvalues

    def backpropagate(self, prediction, target):
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()


class EEnet(torch.nn.Module):
    def __init__(self):
        super(EEnet, self).__init__()
        # conv_3_channels needs replacement if conv layer setup changes
        self.conv_1 = torch.nn.Conv2d(1, 16, 8, stride=4)
        self.conv_2 = torch.nn.Conv2d(16, 16, 4, stride=2)
        self.conv_3 = torch.nn.Conv2d(16, 16, 3, stride=1)
        
        self.conv_net = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, stride=4),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(16, 16, 4, stride=2),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(16, 16, 3, stride=1),
                                            torch.nn.ReLU()
                                            )

        height, width = conv2d_size_out(84, 84, 4, 8)
        height, width = conv2d_size_out(height, width, 2, 4)
        height, width = conv2d_size_out(height, width, 1, 3)

        self.layer_node_count = int(height * width * 64)
        self.lay1 = torch.nn.Linear(self.layer_node_count, 512)
        self.lay2 = torch.nn.Linear(512, 18)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, state):
        state = state.float() / 255
        state1, state2 = torch.split(state, 1, 1)
        state2 = self.conv_net(state2)
        state1 = functional.relu(self.conv_1(state1))
        state1 = functional.relu(self.conv_2(state1))
        state1 = functional.relu(self.conv_3(state1))
        state = functional.relu(self.lay1(state.view(state.shape[0], -1)))
        qvalues = self.lay2(state)
        return qvalues

    def backpropagate(self, prediction, target):
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()
