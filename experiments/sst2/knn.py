import os
import pickle as pkl

from argparse import ArgumentParser
from sklearn.neighbors import kneighbors_graph

from tint.models import Bert, DistilBert, Roberta

from experiments.sst2.utils import get_word_embeddings, model_dict


def knn(
    dataset_name: str = "sst2",
    model_name: str = "bert",
    n_neighbors: int = 500,
    n_jobs: int = None,
    save_path: str = "knns",
    tokenizer=None,
    model=None,
):
    # Load tokenizer and model
    if tokenizer is None or model is None:
        pretrained_model_name_or_path = model_dict[dataset_name][model_name]
        if model_name == "bert":
            tokenizer, model = Bert(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
            )
        elif model_name == "distilbert":
            tokenizer, model = DistilBert(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
            )
        elif model_name == "roberta":
            tokenizer, model = Roberta(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
            )
        else:
            raise NotImplementedError

    print(f"Starting KNN computation..")

    word_features = (
        get_word_embeddings(model, model_name).cpu().detach().numpy()
    )
    word_idx_map = tokenizer.get_vocab()
    knns = kneighbors_graph(
        word_features, n_neighbors=n_neighbors, mode="distance", n_jobs=n_jobs
    )

    if save_path is not None:
        knn_path = os.path.join(save_path, f"{dataset_name}_{model_name}.pkl")
        with open(knn_path, "wb") as fp:
            pkl.dump([word_idx_map, word_features, knns], fp)
        print(f"Written KNN data at {knn_path}")

    return word_idx_map, word_features, knns


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
        "--save-path",
        type=str,
        default="knns",
        help="Where to store the knns.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    knn(
        dataset_name=args.dataset,
        model_name=args.model,
        n_neighbors=args.n_neighbors,
        n_jobs=args.n_jobs,
        save_path=args.save_path,
    )
