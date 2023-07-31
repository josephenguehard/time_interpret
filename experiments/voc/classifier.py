import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, inception_v3
from typing import Union

from tint.models import Net


class VocClassifier(nn.Module):
    def __init__(self, model):
        super(VocClassifier, self).__init__()

        if model == "inception":
            self.model = inception_v3(pretrained=True, init_weights=False)
        elif model == "resnet":
            self.model = resnet18(pretrained=True)
        else:
            raise NotImplementedError

        self.linear = nn.Linear(1000, 20)

        # Freeze resnet layers
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        if not isinstance(x, torch.Tensor):
            x = x.logits
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class VocClassifierNet(Net):
    def __init__(
        self,
        model: str = "resnet",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        super().__init__(
            layers=VocClassifier(model=model),
            loss="nll",
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        target = th.zeros(len(x), 20).long().to(x.device)
        for i in range(len(x)):
            idx = list(set(y[i].unique().tolist()) - {0, 255})
            idx = [x - 1 for x in idx]
            target[i, idx] = 1
        loss = self.loss(y_hat, target)
        return loss
