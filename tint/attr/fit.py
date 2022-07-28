import copy
import torch as th
import torch.nn.functional as F

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
    _format_output,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Callable

from .models import JointFeatureGenerator, JointFeatureGeneratorNet


def kl_multilabel(p1, p2, reduction="none"):
    # treats each column as separate class and calculates KL over the class,
    # sums it up and sends batched
    n_classes = p1.shape[1]
    total_kl = th.zeros(p1.shape)

    for n in range(n_classes):
        p2_tensor = th.stack([p2[:, n], 1 - p2[:, n]], dim=1)
        p1_tensor = th.stack([p1[:, n], 1 - p1[:, n]], dim=1)
        kl = F.kl_div(th.log(p2_tensor), p1_tensor, reduction=reduction)
        total_kl[:, n] = th.sum(kl, dim=1)

    return total_kl


class Fit(PerturbationAttribution):
    """
    Feature Importance in Time.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        https://proceedings.neurips.cc/paper/2015/hash/b618c3210e934362ac261db280128c22-Abstract.html
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        trainer: Trainer = None,
        generator: JointFeatureGeneratorNet = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided. Default to ``None``
            generator (JointFeatureGeneratorNet): A generative model to predict
                future observations. If ``None``, a default model will be
                provided. Default to ``None``
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. Default to ``None``

        Returns:
            (th.Tensor, tuple): Attributions.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_input(inputs)

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=100)

        # Get representations
        outputs_list = list()
        for x in inputs:
            outputs_list.append(
                self._attributes(
                    inputs=x,
                    trainer=trainer,
                    generator=generator,
                    additional_forward_args=additional_forward_args,
                ).reshape(-1, 1)
            )
        outputs = _reduce_list(outputs_list)

        return _format_output(is_inputs_tuple, outputs)

    def _attributes(
        self,
        inputs: th.Tensor,
        trainer: Trainer,
        generator: JointFeatureGeneratorNet = None,
        batch_size: int = 32,
        additional_forward_args: Any = None,
    ):
        # Get input and output shape
        shape = inputs.shape

        # Init MaskNet if not provided
        if generator is None:
            generator = JointFeatureGeneratorNet()
        else:
            generator = copy.deepcopy(generator)

        # Init model
        generator.net.init(feature_size=shape[-1])

        # Prepare data
        dataloader = DataLoader(TensorDataset(inputs), batch_size=batch_size)

        # Fit model
        trainer.fit(generator, train_dataloaders=dataloader)

        # Set model to eval mode
        generator.eval()

        return self.representation(
            generator=generator.net,
            inputs=inputs,
            additional_forward_args=additional_forward_args,
        )

    def representation(
        self,
        generator: JointFeatureGenerator,
        inputs: th.Tensor,
        additional_forward_args: Any = None,
        n_samples: int = 10,
        distance_metric: str = "kl",
        multilabel: bool = False,
    ):
        """
        Get representations based on a generator and inputs.

        Args:
            generator (JointFeatureGenerator): A generator.
            inputs (th.Tensor): Input data.
            additional_forward_args (Any): Optional additional args to be
                passed into the model. Default to ``None``
            n_samples (int): Number of Monte-Carlo samples. Default to 10
            distance_metric (str): Distance metric. Default to ``'kl'``
            multilabel (bool): Whether the task is single or multi-labeled.
                Default to ``False``

        Returns:
            th.Tensor: attributions.
        """
        assert distance_metric in [
            "kl",
            "mean_divergence",
            "LHS",
            "RHS",
        ], "Unrecognised distance metric."

        _, t_len, n_features = inputs.shape
        score = th.zeros(inputs.shape)

        if multilabel:
            activation = F.sigmoid
        else:
            activation = lambda x: F.softmax(x, -1)

        p_y_t = activation(
            _run_forward(
                forward_func=self.forward_func,
                inputs=inputs,
                additional_forward_args=additional_forward_args,
            )
        )

        for t in range(1, t_len):
            p_tm1 = activation(
                _run_forward(
                    forward_func=self.forward_func,
                    inputs=inputs[:, :t, :],
                    additional_forward_args=additional_forward_args,
                )
            )

            for i in range(n_features):
                x_hat = inputs[:, 0 : t + 1, :].clone()
                div_all = []

                for _ in range(n_samples):
                    x_hat_t, _ = generator.forward_conditional(
                        inputs[:, :t, :], inputs[:, t, :], [i]
                    )
                    x_hat[:, t, :] = x_hat_t
                    y_hat_t = activation(
                        _run_forward(
                            forward_func=self.forward_func,
                            inputs=x_hat,
                            additional_forward_args=additional_forward_args,
                        )
                    )

                    if distance_metric == "kl":
                        if not multilabel:
                            div = th.sum(
                                F.kl_div(
                                    th.log(p_tm1), p_y_t, reduction="none"
                                ),
                                -1,
                            ) - th.sum(
                                F.kl_div(
                                    th.log(y_hat_t), p_y_t, reduction="none"
                                ),
                                -1,
                            )
                        else:
                            t1 = kl_multilabel(p_y_t, p_tm1)
                            t2 = kl_multilabel(p_y_t, y_hat_t)
                            div, _ = th.max(t1 - t2, dim=1)
                            # div = div[:,0] #flatten
                        div_all.append(div.cpu().detach())

                    elif distance_metric == "mean_divergence":
                        div = th.abs(y_hat_t - p_y_t)
                        div_all.append(th.mean(div.detach().cpu(), -1))

                    elif distance_metric == "LHS":
                        div = th.sum(
                            F.kl_div(th.log(p_tm1), p_y_t, reduction="none"),
                            -1,
                        )
                        div_all.append(div.cpu().detach())

                    elif distance_metric == "RHS":
                        div = th.sum(
                            F.kl_div(th.log(y_hat_t), p_y_t, reduction="none"),
                            -1,
                        )
                        div_all.append(div.cpu().detach())
                    else:
                        raise NotImplementedError

                e_div = th.Tensor(div_all).mean(0)
                if distance_metric == "kl":
                    score[:, t, i] = 2.0 / (1 + th.exp(-5 * e_div)) - 1
                elif distance_metric == "mean_divergence":
                    score[:, t, i] = 1 - e_div
                else:
                    score[:, t, i] = e_div

            return score
