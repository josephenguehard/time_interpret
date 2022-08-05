import torch as th
import torch.nn as nn

from torchmetrics import Accuracy, Precision, Recall, AUROC
from typing import Callable, Union

from tint.models import Net


class StateClassifier(nn.Module):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        regres: bool = True,
        bidirectional: bool = False,
    ):
        super(StateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.rnn_type = rnn
        self.regres = regres
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.n_state),
        )

    def forward(self, x):
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(x)
        else:
            all_encodings, (encoding, state) = self.rnn(x)

        if self.regres:
            return self.regressor(encoding.view(encoding.shape[1], -1))
        return encoding.view(encoding.shape[1], -1)


class StateClassifierNet(Net):
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
        classifier = StateClassifier(
            feature_size=feature_size,
            n_state=n_state,
            hidden_size=hidden_size,
            rnn=rnn,
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

        for stage in ["train", "val", "test"]:
            setattr(self, stage + "_acc", Accuracy())
            setattr(self, stage + "_pre", Precision())
            setattr(self, stage + "_rec", Recall())
            setattr(self, stage + "_auroc", AUROC())

    def step(self, batch, stage, t):  # noqa
        x, y = batch
        x = x[:, : t + 1]
        y = y[:, t]
        y_hat = self(x)
        loss = self._loss(y_hat, y.long())

        for metric in ["acc", "pre", "rec", "auroc"]:
            getattr(self, stage + "_" + metric)(y_hat[:, 1], y.long())
            self.log(stage + "_" + metric, getattr(self, stage + "_" + metric))

        return loss

    def training_step(self, batch, batch_idx):
        t = th.randint(batch[1].shape[-1], (1,)).item()
        loss = self.step(batch=batch, stage="train", t=t)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        t = th.randint(batch[1].shape[-1], (1,)).item()
        loss = self.step(batch=batch, stage="val", t=t)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        t = batch[1].shape[-1] - 1
        loss = self.step(batch=batch, stage="test", t=t)
        self.log("test_loss", loss)
