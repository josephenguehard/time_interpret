import torch
import typing
import warnings

from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import (
    _format_input,
    _format_input_baseline,
    _validate_input,
)
from captum.log import log_usage
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from typing import Any, Callable, List, Tuple, Union

from tint.utils import astar_path


class GeodesicIntegratedGradients(GradientAttribution):
    """
    Geodesic Integrated Gradients.

    Args:
        forward_func (callable):  The forward function of the model or any
            modification of it
        data (Tensor): Data to fit the knn algorithm. If not provided, the
            knn will be fitted when calling attribute using the provided
            inputs data.
            Default: None
        n_neighbors (int, tuple): Number of neighbors to use by default.
            Can be an integer (same for every inputs) or a tuple.
            Default: None
        multiply_by_inputs (bool, optional): Indicates whether to factor
            model inputs' multiplier in the final attribution scores.
            In the literature this is also known as local vs global
            attribution. If inputs' multiplier isn't factored in,
            then that type of attribution method is also called local
            attribution. If it is, then that type of attribution
            method is called global.
            More detailed can be found here:
            https://arxiv.org/abs/1711.06104

            In case of integrated gradients, if `multiply_by_inputs`
            is set to True, final sensitivity scores are being multiplied by
            (inputs - baselines).
    """

    def __init__(
        self,
        forward_func: Callable,
        data: TensorOrTupleOfTensorsGeneric = None,
        n_neighbors: Union[int, Tuple[int]] = None,
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(forward_func=forward_func)
        self._multiply_by_inputs = multiply_by_inputs

        # Fit NearestNeighbors if data is provided
        self.n_neighbors = None
        self.nn = None
        if data is not None:
            data = _format_input(data)

            assert n_neighbors is not None, "You must provide n_neighbors"
            if isinstance(n_neighbors, int):
                n_neighbors = tuple(n_neighbors for _ in data)

            self.n_neighbors = n_neighbors
            self.nn = tuple(
                NearestNeighbors(n_neighbors=n, **kwargs).fit(
                    X=x.reshape(-1, x.shape[-1]).cpu()
                )
                for x, n in zip(data, n_neighbors)
            )

            n_components = tuple(
                sparse.csgraph.connected_components(nn.kneighbors_graph())[0]
                for nn in self.nn
            )
            if any(n > 1 for n in n_components):
                warnings.warn(
                    "The knn graph is disconnected. You should increase n_neighbors"
                )

    # The following overloaded method signatures correspond to the case where
    # return_convergence_delta is False, then only attributions are returned,
    # and when return_convergence_delta is True, the return type is
    # a tuple with both attributions and deltas.
    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_neighbors: Union[int, Tuple[int]] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        n_steiner: int = None,
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
        **kwargs,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_neighbors: Union[int, Tuple[int]] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        n_steiner: int = None,
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
        **kwargs,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_neighbors: Union[int, Tuple[int]] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        n_steiner: int = None,
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        **kwargs,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:
            inputs (tensor or tuple of tensors):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                Baselines define the starting point from which integral
                is computed and can be provided as:

                - a single tensor, if inputs is a single tensor, with
                  exactly the same dimensions as inputs or the first
                  dimension is one and the remaining dimensions match
                  with inputs.

                - a single scalar, if inputs is a single tensor, which will
                  be broadcasted for each input value in input tensor.

                - a tuple of tensors or scalars, the baseline corresponding
                  to each tensor in the inputs' tuple can be:

                  - either a tensor with matching dimensions to
                    corresponding tensor in the inputs' tuple
                    or the first dimension is one and the remaining
                    dimensions match with the corresponding
                    input tensor.

                  - or a scalar, corresponding to a tensor in the
                    inputs' tuple. This scalar value is broadcasted
                    for corresponding input tensor.

                In the cases when `baselines` is not provided, we internally
                use zero scalar corresponding to each input tensor.

                Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                For general 2D outputs, targets can be either:

                - a single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                  the number of examples in inputs (dim 0). Each integer
                  is applied as the target for the corresponding example.

                For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                  elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                  examples in inputs (dim 0), and each tuple containing
                  #output_dims - 1 elements. Each tuple is applied as the
                  target for the corresponding example.

                Default: None
            additional_forward_args (any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                For a tensor, the first dimension of the tensor must
                correspond to the number of examples. It will be
                repeated for each of `n_steps` along the integrated
                path. For all other types, the given argument is used
                for all forward evaluations.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
            method (string, optional): Method for approximating the integral,
                one of `riemann_right`, `riemann_left`, `riemann_middle`,
                `riemann_trapezoid` or `gausslegendre`.
                Default: `gausslegendre` if no method is provided.
            n_steiner (int, optional): If provided, creates a number of steiner
                points to improve the geodesic path.
                Default: None
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. internal_batch_size must be at least equal to
                #examples.
                For DataParallel models, each batch is split among the
                available devices, so evaluations on each available
                device contain internal_batch_size / num_devices examples.
                If internal_batch_size is None, then all evaluations are
                processed in one batch.
                Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                convergence delta or not. If `return_convergence_delta`
                is set to True convergence delta will be returned in
                a tuple following attributions.
                Default: False
        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Integrated gradients with respect to each input feature.
                attributions will always be the same size as the provided
                inputs, with each value providing the attribution of the
                corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                The difference between the total approximated and true
                integrated gradients. This is computed using the property
                that the total sum of forward_func(inputs) -
                forward_func(baselines) must equal the total sum of the
                integrated gradient.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                examples in inputs.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        # If baseline is float or int, create a tensor
        baselines = tuple(
            torch.ones_like(input) * baseline
            if isinstance(baseline, (int, float))
            else baseline
            for input, baseline in zip(inputs, baselines)
        )

        _validate_input(inputs, baselines, n_steps, method)

        # Fit NearestNeighbors if not provided
        n_neighbors = n_neighbors or self.n_neighbors
        assert n_neighbors is not None, "You must provide n_neighbors"
        nn = self.nn
        if nn is None:
            nn = tuple(
                NearestNeighbors(**kwargs).fit(
                    X=x.reshape(-1, x.shape[-1]).cpu()
                )
                for x in inputs
            )

            n_components = tuple(
                sparse.csgraph.connected_components(nn_.kneighbors_graph())[0]
                for nn_ in nn
            )
            if any(n > 1 for n in n_components):
                warnings.warn(
                    "The knn graph is disconnected. You should increase n_neighbors"
                )

        # Get knns
        knns = tuple(
            torch.from_numpy(
                nn_.kneighbors(x, return_distance=False, n_neighbors=n + 1)[
                    :, 1:
                ].reshape(-1)
            )
            for nn_, x, n in zip(nn, inputs, n_neighbors)
        )
        idx = tuple(
            torch.arange(len(x)).repeat_interleave(n)
            for x, n in zip(inputs, n_neighbors)
        )

        # Get baselines knns
        knns_baselines = tuple(
            torch.from_numpy(
                nn_.kneighbors(
                    x, return_distance=False, n_neighbors=n
                ).reshape(-1)
            )
            for nn_, x, n in zip(nn, baselines, n_neighbors)
        )
        idx_baselines = tuple(
            torch.arange(len(y), len(x) + len(y)).repeat_interleave(n)
            for x, y, n in zip(inputs, baselines, n_neighbors)
        )

        # Concat inputs and baselines
        inputs_and_baselines = tuple(
            torch.cat([x, y]) for x, y in zip(inputs, baselines)
        )
        knns = tuple(torch.cat([x, y]) for x, y in zip(knns, knns_baselines))
        idx = tuple(torch.cat([x, y]) for x, y in zip(idx, idx_baselines))

        # Compute grads for inputs and baselines
        if internal_batch_size is not None:
            num_examples = inputs_and_baselines[0][knns[0]].shape[0]
            grads = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=tuple(
                    x[knn] for x, knn in zip(inputs_and_baselines, knns)
                ),
                baselines=tuple(
                    x[id] for x, id in zip(inputs_and_baselines, idx)
                ),
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
            )
        else:
            grads = self._attribute(
                inputs=tuple(
                    x[knn] for x, knn in zip(inputs_and_baselines, knns)
                ),
                baselines=tuple(
                    x[id] for x, id in zip(inputs_and_baselines, idx)
                ),
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )

        # Compute norm of grads
        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        grads_norm = tuple(
            torch.linalg.norm(
                grad,
                dim=2,
            ).sum(0)
            for grad in grads
        )

        # Multiply by inputs - baselines
        attributions_norm = tuple(
            grad_norm * torch.linalg.norm(input[knn] - input[id], dim=1)
            for grad_norm, input, knn, id in zip(
                grads_norm, inputs_and_baselines, knns, idx
            )
        )

        # Create graph for the A* algorithm
        graphs = tuple(dict() for _ in inputs)
        for graph, id, knn, attr in zip(graphs, idx, knns, attributions_norm):
            for i, k, a in zip(id.tolist(), knn.tolist(), attr.tolist()):
                graph[i] = graph.get(i, list()) + [(k, a)]

        # Compute A* paths
        paths = tuple(
            [
                astar_path(graph, i, j)
                for i, j in zip(range(len(x), 2 * len(x)), range(len(x)))
            ]
            for graph, x in zip(graphs, inputs)
        )

        # Get paths lengths
        paths_len = tuple([len(x) - 1 for x in path] for path in paths)

        # Make them pairwise
        paths = tuple(
            torch.cat([torch.Tensor(list(zip(x, x[1:]))).long() for x in path])
            for path in paths
        )

        # Get grad indexes
        grads_idx = tuple(
            [
                torch.where((id == i) * (knn == j))[0][0].item()
                for i, j in zip(path[:, 0], path[:, 1])
            ]
            for id, knn, path in zip(idx, knns, paths)
        )

        # Get grads of each path
        total_grads = tuple(
            grad[:, grad_idx] for grad, grad_idx in zip(grads, grads_idx)
        )

        # Split for each path
        total_grads = tuple(
            torch.split(grad, split_size_or_sections=path_len, dim=1)
            for grad, path_len in zip(total_grads, paths_len)
        )

        # Sum over points and paths
        # and stack result
        total_grads = tuple(
            tuple(x.sum(0).sum(0) for x in grad) for grad in total_grads
        )
        total_grads = tuple(torch.stack(grad) for grad in total_grads)

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(
                    total_grads, inputs, baselines
                )
            )

        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_output(is_inputs_tuple, attributions), delta
        return _format_output(is_inputs_tuple, attributions)

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[
            None, Tuple[List[float], List[float]]
        ] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas],
                dim=0,
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multiply it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # Reshape scaled_grads
        scaled_grads = tuple(
            scaled_grad.reshape(
                (n_steps, grad.shape[0] // n_steps) + grad.shape[1:]
            )
            for scaled_grad, grad in zip(scaled_grads, grads)
        )

        return scaled_grads

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs
