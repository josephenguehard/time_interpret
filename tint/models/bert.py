try:
    from transformers.models.bert import (
        BertConfig,
        BertTokenizer,
        BertForSequenceClassification,
    )
except ImportError:
    BertConfig = None
    BertTokenizer = None
    BertForSequenceClassification = None


def Bert(
    pretrained_model_name_or_path: str = None,
    config=None,
    vocab_file=None,
    **kwargs
):
    r"""
    Get Bert model for sentence classification, either as a pre-trained model
    or from scratch.

    Args:
        pretrained_model_name_or_path: Path of the pre-trained model.
            If ``None``, return an untrained Bert model. Default to ``None``
        config: Config of the Bert. Required when not loading a pre-trained
            model, otherwise unused. Default to ``None``
        vocab_file: Path to a vocab file for the tokenizer.
            Default to ``None``
        kwargs: Additional arguments for the tokenizer if not pretrained.

    Returns:
        2-element tuple of **Bert Tokenizer**, **Bert Model**:
        - **Bert Tokenizer** (*BertTokenizer*):
            Bert Tokenizer.
        - **Bert Model** (*BertForSequenceClassification*):
            Bert model for sentence classification.

    References:
        https://huggingface.co/docs/transformers/main/en/model_doc/bert

    Examples:
        >>> from tint.models import Bert
         <BLANKLINE>
         >>> tokenizer, model = Bert("bert-base-uncased")
    """
    assert BertConfig is not None, "transformers is not installed."

    # Load pretrained model if path provided
    if pretrained_model_name_or_path is None:
        assert config is not None, "Bert config must be provided."
        assert vocab_file is not None, "vocab file must be provided."
        return (
            BertTokenizer(vocab_file, **kwargs),
            BertForSequenceClassification(config=config),
        )

    # Otherwise return untrained bert model
    return (
        BertTokenizer.from_pretrained(pretrained_model_name_or_path),
        BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            return_dict=False,
        ),
    )
