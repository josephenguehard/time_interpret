import torch as th

from typing import Callable, Union

from experiments.hmm.classifier import StateClassifier
from tint.models import Net


class MimicRegressorNet(Net):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        classifier = StateClassifier(
            feature_size=feature_size,
            n_state=n_state,
            hidden_size=hidden_size,
            rnn=rnn,
            dropout=dropout,
            regres=regres,
            bidirectional=bidirectional,
        )

        super().__init__(
            layers=classifier,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )
        self.save_hyperparameters()

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        t = th.randint(batch[1].shape[-1], (1,)).item()
        x, y = batch
        x = x[:, : t + 1]
        y = y[:, t]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        t = batch[1].shape[-1] - 1
        x, y = batch
        x = x[:, : t + 1]
        return self(x)
