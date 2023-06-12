try:
    from transformers.models.roberta import (
        RobertaConfig,
        RobertaTokenizer,
        RobertaForSequenceClassification,
    )
except ImportError:
    RobertaConfig = None
    RobertaTokenizer = None
    RobertaForSequenceClassification = None


def Roberta(
    pretrained_model_name_or_path: str = None,
    config=None,
    vocab_file=None,
    cache_dir=None,
    **kwargs,
):
    r"""
    Get Roberta model for sentence classification, either as a pre-trained
    model or from scratch.

    Args:
        pretrained_model_name_or_path: Path of the pre-trained model.
            If ``None``, return an untrained Roberta model.
            Default to ``None``
        config: Config of the Roberta. Required when not loading a
            pre-trained model, otherwise unused. Default to ``None``
        vocab_file: Path to a vocab file for the tokenizer.
            Default to ``None``
        cache_dir: Where to save pretrained model. Default to ``None``
        kwargs: Additional arguments for the tokenizer if not pretrained.

    Returns:
        2-element tuple of **Roberta Tokenizer**, **Roberta Model**:
        - **Roberta Tokenizer** (*RobertaTokenizer*):
            Roberta Tokenizer.
        - **Roberta Model** (*RobertaForSequenceClassification*):
            Roberta model for sentence classification.

    References:
        https://huggingface.co/docs/transformers/main/en/model_doc/roberta

    Examples:
        >>> from tint.models import Roberta
         <BLANKLINE>
        >>> tokenizer, model = Roberta("roberta-base")
    """
    assert RobertaConfig is not None, "transformers is not installed."

    # Return untrained bert model if path not provided
    if pretrained_model_name_or_path is None:
        assert config is not None, "Roberta config must be provided."
        assert vocab_file is not None, "vocab file must be provided."
        return (
            RobertaTokenizer(vocab_file, **kwargs),
            RobertaForSequenceClassification(config=config),
        )

    # Otherwise load pretrained model
    return (
        RobertaTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
        ),
        RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            return_dict=False,
        ),
    )
