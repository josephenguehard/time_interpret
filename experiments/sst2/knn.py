import os
import pickle as pkl

from argparse import ArgumentParser
from sklearn.neighbors import kneighbors_graph

from tint.models import Bert

from .utils import get_word_embeddings


def knn(
    tokenizer=None,
    model=None,
    n_neighbors: int = 500,
    n_jobs: int = None,
    save_path: str = "knns",
):
    # If model or tokenizer is None, load them
    if tokenizer is None or model is None:
        tokenizer, model = Bert(
            pretrained_model_name_or_path="textattack/bert-base-uncased-SST-2",
        )

    print(f"Starting KNN computation..")

    word_features = get_word_embeddings(model).cpu().detach().numpy()
    word_idx_map = tokenizer.get_vocab()
    knns = kneighbors_graph(
        word_features, n_neighbors=n_neighbors, mode="distance", n_jobs=n_jobs
    )

    if save_path is not None:
        knn_path = os.path.join(save_path, f"bert_sst2.pkl")
        with open(knn_path, "wb") as fp:
            pkl.dump([word_idx_map, word_features, knns], fp)
        print(f"Written KNN data at {knn_path}")

    return word_idx_map, word_features, knns


def parse_args():
    parser = ArgumentParser()
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
        "--save-path",
        type=str,
        default="knns",
        help="Where to store the knns.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    knn(
        n_neighbors=args.n_neighbors,
        n_jobs=args.n_jobs,
        save_path=args.save_path,
    )
