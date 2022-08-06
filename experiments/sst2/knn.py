import os
import pickle as pkl

from sklearn.neighbors import kneighbors_graph

from utils import get_word_embeddings


def knn(
    tokenizer,
    model,
    n_neighbors: int = 500,
    n_jobs: int = None,
    save_path: str = "knns",
):
    print(f"Starting KNN computation..")

    word_features = get_word_embeddings(model).cpu().detach().numpy()
    word_idx_map = tokenizer.get_vocab()
    knns = kneighbors_graph(
        word_features, n_neighbors=n_neighbors, mode="distance", n_jobs=n_jobs
    )

    if save_path is not None:
        knn_path = os.path.join(
            save_path, f"{args.nn}_{args.dataset}_{args.nbrs}.pkl"
        )
        with open(knn_path, "wb") as fp:
            pkl.dump([word_idx_map, word_features, knns], fp)
        print(f"Written KNN data at {knn_path}")

    return word_idx_map, word_features, knns
