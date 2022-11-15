import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class VocClassifier(nn.Module):
    def __init__(self):
        super(VocClassifier, self).__init__()
        self.resnet = resnet18(pretrained=True)

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=-1)
