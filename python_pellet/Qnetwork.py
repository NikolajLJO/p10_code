import torch
import torch.nn.functional as functional


def conv2d_size_out(height: int, width: int, stride: int, kernel_size: int, padding=0):
    result_1 = ((height - kernel_size + 2 * padding) / stride) + 1
    result_2 = ((width - kernel_size + 2 * padding) / stride) + 1
    return int(result_1), int(result_2)


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

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
        state = functional.relu(self.conv_1(state))
        state = functional.relu(self.conv_2(state))
        state = functional.relu(self.conv_3(state))
        state = functional.relu(self.lay1(state.view(state.shape[0], -1)))
        qvalues = self.lay2(state)
        return qvalues

    def backpropagate(self, prediction, target):
        if len(target.shape) == 1:
            target = target.unsqueeze(0)

        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()


class Qnet(Network):
    def __init__(self):
        super(Qnet, self).__init__()


class EEnet(Network):
    def __init__(self):
        super(EEnet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(2, 32, 8, stride=4)
