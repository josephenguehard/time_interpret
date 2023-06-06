try:
    from transformers.models.distilbert import (
        DistilBertConfig,
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
    )
except ImportError:
    DistilBertConfig = None
    DistilBertTokenizer = None
    DistilBertForSequenceClassification = None


def DistilBert(
    pretrained_model_name_or_path: str = None,
    config=None,
    vocab_file=None,
    cache_dir=None,
    **kwargs,
):
    r"""
    Get DistilBert model for sentence classification, either as a pre-trained
    model or from scratch.

    Args:
        pretrained_model_name_or_path: Path of the pre-trained model.
            If ``None``, return an untrained DistilBert model.
            Default to ``None``
        config: Config of the DistilBert. Required when not loading a
            pre-trained model, otherwise unused. Default to ``None``
        vocab_file: Path to a vocab file for the tokenizer.
            Default to ``None``
        cache_dir: Where to save pretrained model. Default to ``None``
        kwargs: Additional arguments for the tokenizer if not pretrained.

    Returns:
        2-element tuple of **DistilBert Tokenizer**, **DistilBert Model**:
        - **DistilBert Tokenizer** (*DistilBertTokenizer*):
            DistilBert Tokenizer.
        - **DistilBert Model** (*DistilBertForSequenceClassification*):
            DistilBert model for sentence classification.

    References:
        https://huggingface.co/docs/transformers/main/en/model_doc/distilbert

    Examples:
        >>> from tint.models import DistilBert
        <BLANKLINE>
        >>> tokenizer, model = DistilBert("distilbert-base-uncased")
    """
    assert DistilBertConfig is not None, "transformers is not installed."

    # Return untrained bert model if path not provided
    if pretrained_model_name_or_path is None:
        assert config is not None, "DistilBert config must be provided."
        assert vocab_file is not None, "vocab file must be provided."
        return (
            DistilBertTokenizer(vocab_file, **kwargs),
            DistilBertForSequenceClassification(config=config),
        )

    # Otherwise load pretrained model
    return (
        DistilBertTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
        ),
        DistilBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            return_dict=False,
        ),
    )
