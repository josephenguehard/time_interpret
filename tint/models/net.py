import pytorch_lightning as pl
import torch as th
import torch.nn as nn

from typing import Callable, List, Union


LOSSES = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "nll": nn.NLLLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "soft_cross_entropy": nn.CrossEntropyLoss,
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
    Base Net class.

    This provides a wrapper around any Pytorch model into the
    Pytorch Lightning framework.

    Net adds a loss and an optimizer to the model. The following losses are
    available:

    - MAE: ``'l1'``
    - MSE: ``'mse'``
    - NLL: ``'nll'``
    - CrossEntropy: ``'cross_entropy'``
    - CrossEntropy with soft labels: ``'soft_cross_entropy'``
    - BCE with logits: ``'bce_with_logits'``

    The following optimizer are available:

    - SGD: ``'sgd'``
    - Adam: ``'adam'``

    It is also possible to pass a custom learning rate to the Net,
    as well as a learning rate scheduler. Both SGD and Adam also
    support l2 regularisation.

    Args:
        layers (list, nn.Module): The base layers. Can be either a Pytorch
            module or a list of Pytorch modules.
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    References:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    Examples:
        >>> import torch.nn as nn
        >>> from tint.models import MLP, Net
        <BLANKLINE>
        >>> mlp = MLP(units=[5, 10, 1])  # Simple fc with relu activations.
        >>> net = Net([mlp])  # Wrap the mlp into a PyTorch Lightning Net
    """

    def __init__(
        self,
        layers: Union[List[nn.Module], nn.Module],
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        super().__init__()
        self._soft_labels = False

        if isinstance(layers, nn.Module):
            self.net = layers
        else:
            self.net = nn.Sequential()
            for i, layer in enumerate(layers):
                self.net.add_module(
                    f"{layer.__class__.__name__.lower()}_{i}", layer
                )

        if isinstance(loss, str):
            if loss == "soft_cross_entropy":
                self._soft_labels = True
            loss = LOSSES[loss]()
        if isinstance(lr_scheduler, str):
            lr_scheduler = LR_SCHEDULERS[lr_scheduler]

        self._loss = loss
        self._optim = optim
        self.lr = lr
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args or dict()
        self.l2 = l2

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

    def loss(self, inputs, target):
        inputs = inputs.reshape(-1, inputs.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        if isinstance(self._loss, (nn.CrossEntropyLoss, nn.NLLLoss)):
            if self._soft_labels:
                target = target.softmax(-1)
            else:
                if inputs.shape == target.shape:
                    target = target.argmax(-1)
                target = target.reshape(-1).long()

        return self._loss(inputs, target)

    def step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        if isinstance(y_hat, tuple):  # For Bert model
            y_hat = y_hat[0]
        loss = self.loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="val")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="test")
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x.float())

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
