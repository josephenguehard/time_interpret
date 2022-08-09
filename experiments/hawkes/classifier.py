import torch as th

from typing import Callable, Union

from experiments.hmm.classifier import StateClassifierNet


class HawkesClassifier(StateClassifierNet):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        regres: bool = True,
        bidirectional: bool = False,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        super().__init__(
            feature_size=feature_size,
            n_state=n_state,
            hidden_size=hidden_size,
            rnn=rnn,
            regres=regres,
            bidirectional=bidirectional,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def test_step(self, batch, batch_idx):
        t = th.randint(batch[1].shape[-1], (1,)).item()
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="test", t=t)
        self.log("test_loss", loss)
