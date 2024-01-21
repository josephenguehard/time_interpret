import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        input = 1 - F.relu(1 - input)
        return input.sum(-1).sum(-1).unsqueeze(-1)


class BasicModel5_MultiArgs(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1, input2, additional_input1, additional_input2=0):
        relu_out1 = F.relu(input1 - 1) * additional_input1
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2 * additional_input1
        return F.relu(relu_out1 - relu_out2)[:, additional_input2].sum(-1).unsqueeze(-1)


class BasicLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x.sum(1))


class BasicLinearModel5_MultiArgs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 1)

    def forward(self, input1, input2):
        return self.linear(th.cat((input1, input2), dim=-1)).sum(1)


class BasicRnnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(5, 3, batch_first=True)
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(self.rnn(x)[1][0])
