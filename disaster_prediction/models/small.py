from disaster_prediction.model_specs import ModelSpecs
import torch
from torch import nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from disaster_prediction.utils import KeyedDataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('google-bert/bert-base-uncased')
        self.head = nn.Sequential(nn.LazyLinear(2))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_layer = outputs['pooler_output']

        logits = self.head(hidden_layer)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}

def create_model():
    return Model()

def create_dataset(df: pd.DataFrame, include_labels=True) -> KeyedDataset:
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased', do_lower_case=True)
    encoded_dict = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    if include_labels:
        labels = torch.tensor(df['target'].tolist())
        return KeyedDataset(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
    else:
        return KeyedDataset(input_ids=input_ids,
                            attention_mask=attention_mask)

model_specs = ModelSpecs(
    model_name='small',
    model_creator=create_model,
    dataset_creator=create_dataset,
    learning_rate=5e-5,
    training_epochs=4,
    batch_size=32
)