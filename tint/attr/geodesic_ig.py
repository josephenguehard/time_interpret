import numpy as np
import torch
import typing
import warnings

from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.common import (
    _format_input,
    _format_input_baseline,
    _reshape_and_sum,
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

from tint.utils import astar_path, _geodesic_batch_attribution

from tqdm.notebook import tqdm


class GeodesicIntegratedGradients(GradientAttribution):
    """
    Geodesic Integrated Gradients.

    Args:
        forward_func (callable):  The forward function of the model or any
            modification of it.
        nn (NearestNeighbors, tuple): Nearest neighbors method.
            If not provided, will be created when calling __init__ or
            attribute.
            Default: None
        data (Tensor, tuple): Data to fit the knn algorithm. If not provided,
            the knn will be fitted when calling attribute using the provided
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
        nn: Union[NearestNeighbors, Tuple[NearestNeighbors, ...]] = None,
        data: TensorOrTupleOfTensorsGeneric = None,
        n_neighbors: Union[int, Tuple[int]] = None,
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(forward_func=forward_func)
        self._multiply_by_inputs = multiply_by_inputs

        self.n_neighbors = None
        self.nn = None
        self.data = None

        # Register nn if provided
        if nn is not None:
            if not isinstance(nn, tuple):
                nn = (nn,)
            self.nn = nn
            self.n_neighbors = tuple(nn_.n_neighbors for nn_ in self.nn)

        # Fit NearestNeighbors if data is provided
        if data is not None:
            data = _format_input(data)

            assert n_neighbors is not None, "You must provide n_neighbors"
            if isinstance(n_neighbors, int):
                n_neighbors = tuple(n_neighbors for _ in data)

            self.n_neighbors = n_neighbors
            self.nn = tuple(
                NearestNeighbors(n_neighbors=n, **kwargs).fit(
                    X=x.reshape(len(x), -1).cpu()
                )
                for x, n in zip(data, n_neighbors)
            )
            self.data = data

            n_components = tuple(
                sparse.csgraph.connected_components(nn.kneighbors_graph())[0]
                for nn in self.nn
            )
            if any(n > 1 for n in n_components):
                warnings.warn(
                    "The knn graph is disconnected. You should increase n_neighbors."
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
        n_steps: int = 5,
        n_steiner: int = None,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
        distance: str = "geodesic",
        show_progress: bool = False,
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
        n_steps: int = 5,
        n_steiner: int = None,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
        distance: str = "geodesic",
        show_progress: bool = False,
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
        n_steps: int = 5,
        n_steiner: int = None,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        distance: str = "geodesic",
        show_progress: bool = False,
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
                method. Default: 5.
            method (string, optional): Method for approximating the integral,
                one of `riemann_right`, `riemann_left`, `riemann_middle`,
                `riemann_trapezoid` or `gausslegendre`.
                Default: `gausslegendre` if no method is provided.
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
            distance (str, optional): Which distance to use with the A*
                algorithm:

                - 'geodesic': the geodesic distance using the gradients norms.

                - 'euclidean': using the plain euclidean distance between
                  points. This method amounts to the one described here:
                  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02055-7

                Default: 'geodesic'
            show_progress (bool, optional): Displays the progress of computation.
                It will try to use tqdm if available for advanced features
                (e.g. time estimation). Otherwise, it will fallback to
                a simple output of progress.
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

        # If additional_forward_args has a tensor, assert inputs
        # consists of one sample
        if additional_forward_args is not None:
            if any(isinstance(x, Tensor) for x in additional_forward_args):
                assert (
                    len(inputs[0]) == 1
                ), "Only one sample must be passed when additional_forward_args has a tensor."

        # Check distance
        assert distance in [
            "geodesic",
            "euclidean",
        ], f"distance must be either 'geodesic' or 'euclidean', got {distance}"

        # Fit NearestNeighbors if not provided
        n_neighbors = n_neighbors or self.n_neighbors
        assert n_neighbors is not None, "You must provide n_neighbors"
        nn = self.nn
        if nn is None:
            if isinstance(n_neighbors, int):
                n_neighbors = tuple(n_neighbors for _ in inputs)

            nn = tuple(
                NearestNeighbors(n_neighbors=n, **kwargs).fit(
                    X=x.reshape(len(x), -1).cpu()
                )
                for x, n in zip(inputs, n_neighbors)
            )

            n_components = tuple(
                sparse.csgraph.connected_components(nn_.kneighbors_graph())[0]
                for nn_ in nn
            )
            if any(n > 1 for n in n_components):
                warnings.warn(
                    "The knn graph is disconnected. You should increase n_neighbors."
                )

        # Concat data, inputs and baselines
        if self.data is None:
            data = tuple(torch.cat([x, y]) for x, y in zip(inputs, baselines))
        else:
            data = tuple(
                torch.cat([x, y, z])
                for x, y, z in zip(self.data, inputs, baselines)
            )

        # Get knns
        idx, knns, dists = self._get_knns(
            nn=nn,
            inputs=data,
            n_neighbors=n_neighbors,
        )

        # If steiner is provided, augment inputs
        if n_steiner is not None:
            # Get number of points to add
            max_dists = tuple(d.max() for d in dists)
            n_points = tuple(
                torch.div(d, m / n_steiner, rounding_mode="floor").long() + 1
                for d, m in zip(dists, max_dists)
            )

            # Augment inputs
            data = tuple(
                torch.cat(
                    [x]
                    + [
                        torch.stack(
                            [
                                x[id][i] + (k / n) * (x[knn][i] - x[id][i])
                                for k in range(1, n)
                            ]
                        )
                        for i, n in enumerate(n_point)
                        if n > 1
                    ],
                    dim=0,
                )
                for x, knn, id, n_point in zip(data, knns, idx, n_points)
            )

            # Get knns
            knns, idx, _ = self._get_knns(
                nn=nn,
                inputs=data,
                n_neighbors=n_neighbors,
            )

        # Compute grads for inputs and baselines
        if internal_batch_size is not None:
            grads_norm, total_grads = _geodesic_batch_attribution(
                attr_method=self,
                inputs=data,
                idx=idx,
                knns=knns,
                internal_batch_size=internal_batch_size,
                show_progress=show_progress,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )
        else:
            grads_norm, total_grads = self._attribute(
                inputs=tuple(x[knn] for x, knn in zip(data, knns)),
                baselines=tuple(x[id] for x, id in zip(data, idx)),
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )

        # Get ||xi - xj|| for all data if euclidean
        if distance == "euclidean":
            grads_norm = tuple(
                torch.linalg.norm(
                    (x[knn] - x[id]).reshape(len(x[knn]), -1),
                    dim=1,
                )
                for x, knn, id in zip(data, knns, idx)
            )

        # Create undirected graph for the A* algorithm
        graphs = tuple(dict() for _ in data)
        for graph, id, knn, attr in zip(graphs, idx, knns, grads_norm):
            for i, k, a in zip(id.tolist(), knn.tolist(), attr.tolist()):
                graph[k] = list(set(graph.get(k, list()) + [(i, a)]))
                graph[i] = list(set(graph.get(i, list()) + [(k, a)]))

        # Def heuristic for A* algorithm: euclidean distance to target
        def heuristic(u, v, d):
            return torch.linalg.norm(d[u] - d[v]).item()

        # Compute A* paths
        if self.data is not None:
            inputs_idx = tuple(
                range(len(x), len(x) + len(y))
                for x, y in zip(self.data, inputs)
            )
            baselines_idx = tuple(
                range(len(x) + len(y), len(x) + 2 * len(y))
                for x, y in zip(self.data, inputs)
            )
        else:
            inputs_idx = tuple(range(len(x)) for x in inputs)
            baselines_idx = tuple(range(len(x), 2 * len(x)) for x in inputs)

        paths = tuple(
            [
                astar_path(graph, i, j, heuristic=None, d=d)
                for i, j in zip(input_idx, baseline_idx)
            ]
            for graph, input_idx, baseline_idx, d in zip(
                graphs, inputs_idx, baselines_idx, data
            )
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
                torch.where((id == i) * (knn == j) + (id == j) * (knn == i))[
                    0
                ][0].item()
                for i, j in zip(path[:, 0], path[:, 1])
            ]
            for id, knn, path in zip(idx, knns, paths)
        )

        # Get grads of each path
        total_grads = tuple(
            grad[grad_idx] for grad, grad_idx in zip(total_grads, grads_idx)
        )

        # Split for each path
        total_grads = tuple(
            torch.split(grad, split_size_or_sections=path_len, dim=0)
            for grad, path_len in zip(total_grads, paths_len)
        )

        # Sum over points and paths
        # and stack result
        total_grads = tuple(
            tuple(x.sum(0) for x in grad) for grad in total_grads
        )
        total_grads = tuple(torch.stack(grad) for grad in total_grads)

        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                total_grads,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_output(is_inputs_tuple, total_grads), delta
        return _format_output(is_inputs_tuple, total_grads)

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
    ) -> (Tuple[Tensor, ...], Tuple[Tensor, ...]):
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

        # Compute norm of grads
        grads_norm = tuple(
            torch.linalg.norm(
                grad.reshape(grad.shape[:2] + (-1,)),
                dim=2,
            ).sum(0)
            for grad in scaled_grads
        )

        # Multiply by inputs - baselines
        grads_norm = tuple(
            grad_norm
            * torch.linalg.norm(
                (input - baseline).reshape(len(input), -1), dim=1
            )
            for grad_norm, input, baseline in zip(
                grads_norm, inputs, baselines
            )
        )

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # Multiply by inputs - baselines if necessary
        if self.multiplies_by_inputs:
            total_grads = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(
                    total_grads, inputs, baselines
                )
            )

        return grads_norm, total_grads

    @staticmethod
    def _get_knns(
        nn: Tuple[NearestNeighbors, ...],
        inputs: Tuple[Tensor, ...],
        n_neighbors: Tuple[int, ...],
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """
        Get k nearest neighbors.

        Args:
            nn (tuple): A tuple of NN methods.
            inputs (tuple): Input data.
            n_neighbors (tuple): Number of neighbors for each method.
        """
        # Get kneighbors_graph
        graphs = tuple(
            nn_.kneighbors_graph(
                x.reshape(len(x), -1).detach().cpu(),
                n_neighbors=n,
                mode="distance",
            )
            for nn_, x, n in zip(nn, inputs, n_neighbors)
        )

        # Get graphs components
        components = tuple(
            sparse.csgraph.connected_components(graph[: graph.shape[1]])
            for graph in graphs
        )

        # Get smallest norms and indexes if multiple components
        add_dists = tuple()
        add_idx = tuple()
        add_knns = tuple()
        for component, input in zip(components, inputs):
            add_dist_list = list()
            add_id_list = list()
            add_knn_list = list()
            for i in range(component[0]):
                for j in range(component[0]):
                    if i < j:
                        input_i = input[: len(component[1])][component[1] == i]
                        input_j = input[: len(component[1])][component[1] == j]
                        norm = torch.linalg.norm(
                            (
                                torch.cat([input_i] * len(input_j), dim=0)
                                - input_j.repeat_interleave(
                                    len(input_i), dim=0
                                )
                            ).reshape(len(input_i) * len(input_j), -1),
                            dim=1,
                        )
                        dist = norm.min().item()
                        norm_idx = norm.argmin().item()
                        id = norm_idx % len(input_i)
                        id = np.where(component[1] == i)[0][id]
                        knn = norm_idx // len(input_i)
                        knn = np.where(component[1] == j)[0][knn]

                        add_dist_list.append(dist)
                        add_id_list.append(id)
                        add_knn_list.append(knn)

            add_dists += (torch.Tensor(add_dist_list),)
            add_idx += (torch.Tensor(add_id_list).long(),)
            add_knns += (torch.Tensor(add_knn_list).long(),)

        # Get dists
        dists = tuple(torch.from_numpy(graph.data) for graph in graphs)
        dists = tuple(d[d != 0.0] for d in dists)
        dists = tuple(
            torch.cat([d, a], dim=0) if len(a) > 0 else d
            for d, a in zip(dists, add_dists)
        )

        # Get nonzeros
        nonzeros = tuple(graph.nonzero() for graph in graphs)

        # Get idx and knns
        idx = tuple(
            torch.from_numpy(nonzero[0]).long() for nonzero in nonzeros
        )
        idx = tuple(
            torch.cat([i, a], dim=0) if len(a) > 0 else i
            for i, a in zip(idx, add_idx)
        )

        knns = tuple(
            torch.from_numpy(nonzero[1]).long() for nonzero in nonzeros
        )
        knns = tuple(
            torch.cat([k, a], dim=0) if len(a) > 0 else k
            for k, a in zip(knns, add_knns)
        )

        return idx, knns, dists

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs
