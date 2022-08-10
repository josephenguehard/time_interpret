import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        input = 1 - F.relu(1 - input)
        return input


class BasicModel5_MultiArgs(nn.Module):
    """
    Slightly modified example model from the paper
    https://arxiv.org/pdf/1703.01365.pdf
    f(x1, x2) = RELU(ReLU(x1 - 1) * x3[0] - ReLU(x2) * x3[1])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1, input2, additional_input1, additional_input2=0):
        relu_out1 = F.relu(input1 - 1) * additional_input1[0]
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2 * additional_input1[1]
        return F.relu(relu_out1 - relu_out2)[:, additional_input2]
