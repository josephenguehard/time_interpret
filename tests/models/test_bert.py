from transformers.models.bert import (
    BertConfig,
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertTokenizer,
)

from tint.models import Bert


def test_init():
    tokenizer, model = Bert("bert-base-uncased")
    assert isinstance(tokenizer, BertTokenizer)
    assert isinstance(model, BertPreTrainedModel)

    tokenizer, model = Bert(config=BertConfig())
    assert isinstance(tokenizer, BertTokenizer)
    assert isinstance(model, BertForSequenceClassification)
