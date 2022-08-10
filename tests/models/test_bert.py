from transformers.models.bert import (
    BertPreTrainedModel,
    BertTokenizer,
)

from tint.models import Bert


def test_init():
    tokenizer, model = Bert("bert-base-uncased")
    assert isinstance(tokenizer, BertTokenizer)
    assert isinstance(model, BertPreTrainedModel)
