import os
import torch as th
import torch.nn as nn

try:
    import fasttext
    fasttext.FastText.eprint = lambda x: None
except ImportError:
    fasttext = None


file_dir = os.path.dirname(__file__)


class Fasttext:
    """
    FastText model.

    Args:
        emb_dim (int): Size of the embeddings. Default to 100
        pre_trained (bool): Whether to load a pre-trained model or not.
            Default to ``False``
        data_dir (str): Where the text data and fasttext model are saved.

    References:
        https://fasttext.cc
    """

    def __init__(
        self,
        emb_dim: int = 100,
        pre_trained: bool = False,
        data_dir: str = os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "data",
            "biobank",
        ),
    ):
        self.emb_dim = emb_dim
        self.data_dir = data_dir

        assert fasttext is not None, "fasttext must be installed."

        self.text = os.path.join(self.data_dir, "text.txt")
        self.words = list()
        self.embedding = None

        if pre_trained:
            model = fasttext.load_model(
                os.path.join(self.data_dir, "fasttext.bin")
            )
            self._load_weights(model=model)

    def __getitem__(self, item):
        return self.embedding(item * th.ones(1).long())

    def fit(self):
        """
        Fit fasttext model.
        """
        assert self.text is not None, "Cannot fit the embedding, text missing."
        model = fasttext.train_unsupervised(
            self.text,
            model="skipgram",
            dim=self.emb_dim,
            minCount=1,
            maxn=0,
        )
        self._load_weights(model=model)

    def transform(self, x: th.Tensor):
        """
        Get fasttext embeddings given indexes.

        Args:
            x (th.Tensor): Input indexes.

        Returns:
            th.Tensor: Fasttext embeddings.
        """
        # Assert model pre-trained
        assert self.embedding is not None, "You need to pre-train the fasttext model first."

        # Reshape data
        data_shape = x.shape
        x = x.reshape(-1, data_shape[-1])

        # Get embeddings
        outputs = th.zeros((x.shape[0], self.emb_dim))
        outputs.index_add_(
            0,
            th.nonzero(x)[:, 0],
            self.embedding(x[th.nonzero(x, as_tuple=True)].long()),
        )

        # Reshape and register
        outputs = outputs.reshape(
            data_shape[:-1] + (self.emb_dim,)
        )

        return outputs

    def _load_weights(self, model):
        """
        Utility function for parallelisation of transform.

        Args:
            model (fasttext.Fasttext): A Fasttext model.
        """
        self.words = sorted([int(x) for x in model.words if x != "</s>"])
        weights = th.zeros((max(self.words) + 1, self.emb_dim))
        for i in self.words:
            weights[i] = th.from_numpy(model[str(i)])
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.embedding.weight.requires_grad = False
