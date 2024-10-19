import torch.nn


def get_bert_embedding(bert_model, inputs):
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)