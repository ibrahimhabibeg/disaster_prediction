from disaster_prediction.model_specs import ModelSpecs
import torch
import pandas as pd
from torch import nn
from transformers import BertTokenizer, BertModel
from disaster_prediction.utils import KeyedDataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_bert = BertModel.from_pretrained('google-bert/bert-large-uncased')
        self.text_bert_no_hidden = 1024

        self.keyword_bert = BertModel.from_pretrained('google-bert/bert-base-uncased')
        self.keyword_bert_no_hidden = 768

        self.head = nn.Sequential(nn.LazyLinear(512),
                                  nn.ReLU(),
                                  nn.LazyLinear(2))

    def forward(self, text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                keyword_input_ids: torch.Tensor,
                keyword_attention_mask: torch.Tensor,
                labels: torch.Tensor = None):
        text_outputs = self.text_bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_hidden_layer = text_outputs['pooler_output']

        keyword_outputs = self.keyword_bert(input_ids=keyword_input_ids, attention_mask=keyword_attention_mask)
        keyword_hidden_layer = keyword_outputs['pooler_output']

        full_hidden_layer = torch.cat((text_hidden_layer, keyword_hidden_layer), dim=1)

        logits = self.head(full_hidden_layer)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}

def create_model():
    return Model()

def create_dataset(df: pd.DataFrame, include_labels=True) -> KeyedDataset:
    df['keyword'] = df['keyword'].fillna('')

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-large-uncased', do_lower_case=True)
    encoded_dict = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
    text_input_ids = encoded_dict['input_ids']
    text_attention_mask = encoded_dict['attention_mask']

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased', do_lower_case=True)
    encoded_dict = tokenizer(df['keyword'].tolist(), padding=True, truncation=True, return_tensors='pt')
    keyword_input_ids = encoded_dict['input_ids']
    keyword_attention_mask = encoded_dict['attention_mask']

    if include_labels:
        labels = torch.tensor(df['target'].tolist())
        return KeyedDataset(text_input_ids=text_input_ids,
                            text_attention_mask=text_attention_mask,
                            keyword_input_ids=keyword_input_ids,
                            keyword_attention_mask=keyword_attention_mask,
                            labels=labels)
    else:
        return KeyedDataset(text_input_ids=text_input_ids,
                            text_attention_mask=text_attention_mask,
                            keyword_input_ids=keyword_input_ids,
                            keyword_attention_mask=keyword_attention_mask)


model_specs = ModelSpecs(
    model_name='large',
    model_creator=create_model,
    dataset_creator=create_dataset,
    learning_rate=5e-5,
    training_epochs=4,
    batch_size=32
)