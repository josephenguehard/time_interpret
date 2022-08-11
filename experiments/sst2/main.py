import torch

from argparse import ArgumentParser
from captum.attr import (
    DeepLift,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
)
from typing import List

from tint.attr import (
    DiscretetizedIntegratedGradients,
)
from tint.attr.models import scale_inputs
from tint.models import Bert
from tint.utils import get_progress_bars

from experiments.sst2.knn import knn
from experiments.sst2.metrics import (
    eval_comprehensiveness,
    eval_log_odds,
    eval_sufficiency,
)
from experiments.sst2.utils import (
    ForwardModel,
    get_base_token_emb,
    get_inputs,
    load_mappings,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def main(
    explainers: List[str],
    device: str,
    steps: int,
    factor: int,
    strategy: str,
    knns_path: str,
    n_neighbors: int,
    n_jobs: int,
    topk: float,
    log_n_steps: int,
):
    # Load data
    assert load_dataset is not None, "datasets is not installed."
    dataset = load_dataset("glue", "sst2")["test"]
    data = list(zip(dataset["sentence"], dataset["label"], dataset["idx"]))

    # Load model and tokenizer
    # This can be replaced with another hugingface model
    tokenizer, model = Bert(
        pretrained_model_name_or_path="textattack/bert-base-uncased-SST-2",
    )

    # Load knn mapping
    if knns_path is not None:
        auxiliary_data = load_mappings(knns_path)
    else:
        auxiliary_data = knn(
            tokenizer=tokenizer,
            model=model,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )

    # Get ref token embedding
    base_token_emb = get_base_token_emb(
        tokenizer=tokenizer, model=model, device=device
    )

    # Prepare forward model
    nn_forward_func = ForwardModel(model=model)

    # Prepare attributions and metrics
    attr = dict()
    _log_odds = dict()
    _comprehensiveness = dict()
    _sufficiency = dict()
    for explainer in explainers:
        attr[explainer] = list()
        _log_odds[explainer] = list()
        _comprehensiveness[explainer] = list()
        _sufficiency[explainer] = list()

    # Compute attributions
    for i, row in get_progress_bars()(enumerate(data), total=len(data)):
        (
            input_ids,
            ref_input_ids,
            input_embed,
            ref_input_embed,
            position_embed,
            ref_position_embed,
            type_embed,
            ref_type_embed,
            attention_mask,
        ) = get_inputs(
            tokenizer=tokenizer, model=model, text=row[0], device=device
        )

        if "deep_lift" in explainers:
            explainer = DeepLift(nn_forward_func)
            _attr = explainer.attribute(
                input_embed,
                additional_forward_args=(
                    attention_mask,
                    position_embed,
                    type_embed,
                ),
            )
            _attr = summarize_attributions(_attr)
            attr["deep_lift"].append(_attr)

        if "discretized_integrated_gradients" in explainers:
            scaled_features = scale_inputs(
                input_ids.squeeze().tolist(),
                ref_input_ids.squeeze().tolist(),
                device,
                auxiliary_data,
                steps=steps,
                factor=factor,
                strategy=strategy,
            )

            explainer = DiscretetizedIntegratedGradients(nn_forward_func)
            _attr = explainer.attribute(
                scaled_features=scaled_features,
                additional_forward_args=(
                    attention_mask,
                    position_embed,
                    type_embed,
                ),
                n_steps=(2**factor) * (steps + 1) + 1,
            )
            _attr = summarize_attributions(_attr)
            attr["discretized_integrated_gradients"].append(_attr)

        if "gradient_shap" in explainers:
            explainer = GradientShap(nn_forward_func)
            _attr = explainer.attribute(
                input_embed,
                baselines=torch.cat([input_embed * 0, input_embed]),
                additional_forward_args=(
                    attention_mask,
                    position_embed,
                    type_embed,
                ),
            )
            _attr = summarize_attributions(_attr)
            attr["gradient_shap"].append(_attr)

        if "input_x_gradient" in explainers:
            explainer = InputXGradient(nn_forward_func)
            _attr = explainer.attribute(
                input_embed,
                additional_forward_args=(
                    attention_mask,
                    position_embed,
                    type_embed,
                ),
            )
            _attr = summarize_attributions(_attr)
            attr["input_x_gradient"].append(_attr)

        if "integrated_gradients" in explainers:
            explainer = IntegratedGradients(nn_forward_func)
            _attr = explainer.attribute(
                input_embed,
                additional_forward_args=(
                    attention_mask,
                    position_embed,
                    type_embed,
                ),
            )
            _attr = summarize_attributions(_attr)
            attr["integrated_gradients"].append(_attr)

        # Append metrics
        for explainer, _attr in attr.items():
            _log_odds[explainer].append(
                eval_log_odds(
                    forward_fn=nn_forward_func,
                    input_embed=input_embed,
                    position_embed=position_embed,
                    type_embed=type_embed,
                    attention_mask=attention_mask,
                    base_token_emb=base_token_emb,
                    attr=_attr[-1],
                    topk=int(topk * 100),
                )[0]
            )
            _comprehensiveness[explainer].append(
                eval_comprehensiveness(
                    forward_fn=nn_forward_func,
                    input_embed=input_embed,
                    position_embed=position_embed,
                    type_embed=type_embed,
                    attention_mask=attention_mask,
                    attr=_attr[-1],
                    topk=int(topk * 100),
                )
            )
            _sufficiency[explainer].append(
                eval_sufficiency(
                    forward_fn=nn_forward_func,
                    input_embed=input_embed,
                    position_embed=position_embed,
                    type_embed=type_embed,
                    attention_mask=attention_mask,
                    attr=_attr[-1],
                    topk=int(topk * 100),
                )
            )

        # Print metrics
        if i % log_n_steps == 0:
            for explainer in explainers:
                print(
                    f"{explainer}, log_odds: {torch.Tensor(_log_odds[explainer]).mean():.4}"
                )
                print(
                    f"{explainer}, comprehensiveness: {torch.Tensor(_comprehensiveness[explainer]).mean():.4}"
                )
                print(
                    f"{explainer}, sufficiency: {torch.Tensor(_sufficiency[explainer]).mean():.4}"
                )

    with open("results.csv", "a") as fp:
        for k in attr:
            fp.write(k + ",")
            fp.write(f"{torch.Tensor(_log_odds[k]).mean():.4},")
            fp.write(f"{torch.Tensor(_comprehensiveness[k]).mean():.4},")
            fp.write(f"{torch.Tensor(_sufficiency[k]).mean():.4},")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "deep_lift",
            "discretized_integrated_gradients",
            "gradient_shap",
            "input_x_gradient",
            "integrated_gradients",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of steps for the DIG attributions.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=0,
        help="Factor for the DIG attributions steps.",
    )
    parser.add_argument(
        "--strategy",
        default="greedy",
        type=str,
        choices=["greedy", "maxcount"],
        help="The algorithm to find the next anchor point,",
    )
    parser.add_argument(
        "--knns-path",
        type=str,
        default="knns/bert_sst2.pkl",
        help="Where the knns are stored. If not provided, compute them.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=500,
        help="Number of neighbors for the knns",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of jobs for the knns",
    )
    parser.add_argument(
        "--topk",
        type=float,
        default=0.2,
        help="Topk attributions for the metrics.",
    )
    parser.add_argument(
        "--log-n-steps",
        type=int,
        default=2,
        help="How often to print the metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        device=args.device,
        steps=args.steps,
        factor=args.factor,
        strategy=args.strategy,
        knns_path=args.knns_path,
        n_neighbors=args.n_neighbors,
        n_jobs=args.n_jobs,
        topk=args.topk,
        log_n_steps=args.log_n_steps,
    )
