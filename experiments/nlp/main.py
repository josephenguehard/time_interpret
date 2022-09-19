import multiprocessing as mp
import random
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
    SequentialIntegratedGradients,
)
from tint.attr.models import scale_inputs
from tint.models import Bert, DistilBert, Roberta
from tint.utils import get_progress_bars

from experiments.nlp.knn import knn
from experiments.nlp.metrics import (
    eval_comprehensiveness,
    eval_log_odds,
    eval_sufficiency,
)
from experiments.nlp.utils import (
    ForwardModel,
    get_base_token_emb,
    get_inputs,
    load_mappings,
    model_dict,
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
    dataset_name: str,
    model_name: str,
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
    if dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2", cache_dir="datasets")["test"]
        data = list(zip(dataset["sentence"], dataset["label"], dataset["idx"]))
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb", cache_dir="datasets")["test"]
        data = list(zip(dataset["text"], dataset["label"]))
        data = [x for x in data if len(x[0]) < 2000]
        data = random.sample(data, 2000)
    elif dataset_name == "rotten":
        dataset = load_dataset("rotten_tomatoes", cache_dir="datasets")["test"]
        data = list(zip(dataset["text"], dataset["label"]))
    else:
        raise NotImplementedError

    # Load model and tokenizer
    # This can be replaced with another hugingface model
    pretrained_model_name_or_path = model_dict[dataset_name][model_name]
    if model_name == "bert":
        tokenizer, model = Bert(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir="models",
        )
    elif model_name == "distilbert":
        tokenizer, model = DistilBert(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir="models",
        )
    elif model_name == "roberta":
        tokenizer, model = Roberta(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir="models",
        )
    else:
        raise NotImplementedError

    # Set model to device
    model.to(device)

    # Load knn mapping
    if knns_path is not None:
        auxiliary_data = load_mappings(knns_path)
    else:
        auxiliary_data = knn(
            dataset_name=dataset_name,
            model_name=model_name,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            tokenizer=tokenizer,
            model=model,
        )

    # Get ref token embedding
    base_token_emb = get_base_token_emb(
        tokenizer=tokenizer, model=model, model_name=model_name, device=device
    )

    # Prepare forward model
    nn_forward_func = ForwardModel(model=model, model_name=model_name)

    # Prepare attributions and metrics
    attr = dict()
    _log_odds = dict()
    _comprehensiveness = dict()
    _sufficiency = dict()
    for explainer in explainers:
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
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            text=row[0],
            device=device,
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
            attr["deep_lift"] = _attr

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
            attr["discretized_integrated_gradients"] = _attr

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
            attr["gradient_shap"] = _attr

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
            attr["input_x_gradient"] = _attr

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
            attr["integrated_gradients"] = _attr

        if "sequential_integrated_gradients" in explainers:
            explainer = SequentialIntegratedGradients(nn_forward_func)
            _attr = explainer.attribute(
                input_embed,
                additional_forward_args=(
                    attention_mask,
                    position_embed,
                    type_embed,
                ),
            )
            _attr = summarize_attributions(_attr)
            attr["sequential_integrated_gradients"] = _attr

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
                    attr=_attr,
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
                    attr=_attr,
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
                    attr=_attr,
                    topk=int(topk * 100),
                )
            )

        # Clear cuda cache
        torch.cuda.empty_cache()

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

    with open("results.csv", "a") as fp, mp.Lock():
        for k in attr:
            fp.write(dataset_name + ",")
            fp.write(model_name + ",")
            fp.write(k + ",")
            fp.write(f"{torch.Tensor(_log_odds[k]).mean():.4},")
            fp.write(f"{torch.Tensor(_comprehensiveness[k]).mean():.4},")
            fp.write(f"{torch.Tensor(_sufficiency[k]).mean():.4}")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        help="Dataset name. Must be either 'sst2', 'imdb' or 'rotten'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        help="Model name. Must be either 'bert', 'distilbert' or 'roberta'.",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "deep_lift",
            "discretized_integrated_gradients",
            "gradient_shap",
            "input_x_gradient",
            "integrated_gradients",
            "sequential_integrated_gradients",
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
        help="The algorithm to find the next anchor point.",
    )
    parser.add_argument(
        "--knns-path",
        type=str,
        default="knns/sst2_bert.pkl",
        help="Where the knns are stored. If not provided, compute them.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=500,
        help="Number of neighbors for the knns.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of jobs for the knns.",
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
        dataset_name=args.dataset,
        model_name=args.model,
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
