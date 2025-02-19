from disaster_prediction.model_specs import ModelSpecs
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

def _get_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    return device

def _prepare_data_loader(df: pd.DataFrame,
                         model_spec: ModelSpecs,
                         include_labels: bool = True,
                         shuffle: bool = False):
    dataset = model_spec.dataset_creator(df, include_labels)
    return DataLoader(dataset, batch_size=model_spec.batch_size, shuffle=shuffle)


class ModelController:
    def __init__(self, model_spec: ModelSpecs, weights_path: str):
        self.model_spec = model_spec
        self.device = _get_device()
        self.weights_path = weights_path
        self.is_trained = os.path.exists(weights_path)

    def __save_model(self) -> None:
        torch.save(self.model_spec.model.state_dict(), self.weights_path)
        self.is_trained = True

    def __load_model(self) -> torch.nn.Module:
        model = self.model_spec.model
        model.load_state_dict(torch.load(self.weights_path, weights_only=True))
        return model

    def train(self, train_df: pd.DataFrame) -> None:
        train_dataloader = _prepare_data_loader(train_df, self.model_spec, shuffle=True, include_labels=True)

        model = self.model_spec.model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.model_spec.learning_rate)
        num_training_steps = len(train_dataloader) * self.model_spec.training_epochs
        lr_scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        progress = tqdm(range(num_training_steps))

        for epoch in range(self.model_spec.training_epochs):
            model.train()
            for batch in train_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress.update()
                progress.set_postfix({'loss': loss.item()})

        self.__save_model()

    def predict(self, df: pd.DataFrame) -> list[int]:
        if not self.is_trained:
            raise Exception("Model is not trained")
        model = self.__load_model().to(self.device)
        dataloader = _prepare_data_loader(df, self.model_spec, shuffle=False, include_labels=False)
        model.eval()
        predictions_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = model(**batch)
                predictions = outputs['logits'].argmax(dim=1)
                predictions_list.extend(predictions.tolist())
        return predictions_list

    def evaluate(self,
                 df: pd.DataFrame,
                 include_f1: bool = True,
                 include_accuracy: bool = True,
                 include_precision: bool = True,
                 include_recall: bool = True,
                 include_roc_auc: bool = True) -> dict[str, float]:
        if not self.is_trained:
            raise Exception("Model is not trained")
        predictions = self.predict(df)
        labels = df['target'].tolist()
        metrics = {}
        if include_f1:
            metrics['f1'] = f1_score(labels, predictions)
        if include_accuracy:
            metrics['accuracy'] = accuracy_score(labels, predictions)
        if include_precision:
            metrics['precision'] = precision_score(labels, predictions)
        if include_recall:
            metrics['recall'] = recall_score(labels, predictions)
        if include_roc_auc:
            metrics['roc_auc'] = roc_auc_score(labels, predictions)
        return metrics