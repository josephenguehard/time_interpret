import pytorch_lightning as pl
import torch as th
import torch.nn as nn

from typing import Union


LOSSES = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
}


LR_SCHEDULERS = {
    "reduce_on_plateau": {
        "scheduler": th.optim.lr_scheduler.ReduceLROnPlateau,
        "monitor": "val_loss",
    },
}


class Net(pl.LightningModule):
    """
    Base NN class.

    This provides a flexible class for various neural networks.

    Args:
        layers (list): The base layers.
        loss (str): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    Examples:
        >>> import torch.nn as nn
        >>> from tint.models import MLP, Net
        <BLANKLINE>
        >>> mlp = MLP(units=[5, 10, 1])  # Simple fc with relu activations.
        >>> net = Net([mlp])  # Wrap the mlp into a PyTorch Lightning Net
    """

    def __init__(
        self,
        layers: list,
        loss: str = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential()
        for layer in layers:
            self.net.add_module(layer.__class__.__name__.lower(), layer)

        self._loss = LOSSES[loss]
        self._optim = optim
        self.lr = lr
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args or dict()
        self.l2 = l2

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

    def step(self, batch):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self._loss(y, y_hat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch=batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch=batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch=batch)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch["x"], batch["y"]
        return self(x)

    def configure_optimizers(self):
        if self._optim == "adam":
            optim = th.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}
