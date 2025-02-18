from abc import ABC, abstractmethod
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
from disaster_prediction.dataset import load_raw_test_df, load_raw_train_df, load_raw_val_df
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(BASE_DIR, '../data/submissions')

class DisasterTweetsModel(ABC):

    def __init__(self):
        self.device = self.get_device()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the name of the specific model.
        """
        pass

    @abstractmethod
    def train(self, force: bool = False):
        """
        Trains the model.
        :param force: If True, forces the training even if the model is already trained.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> list[int]:
        """
        Predicts the target column for the given dataframe.
        The dataframe should have the same structure as the training data.
        :param df: Pandas DataFrame containing four columns: 'id', 'keyword', 'location', 'text'
        :return: List of predictions
        """
        pass

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            print('Using cuda')
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            print('Using mps')
            return torch.device('mps')
        else :
            print('Using cpu')
            return torch.device('cpu')

    def create_submission_file(self):
        df = load_raw_test_df()
        predictions = self.predict(df)
        result = pd.DataFrame({'id': df['id'], 'target': predictions})
        result.to_csv(os.path.join(SUBMISSION_DIR, f'{self.model_name}.csv'), index=False)
        print(f'Submission file created at {os.path.join(SUBMISSION_DIR, f"{self.model_name}.csv")}')

    def evaluate_on_train(self):
        raise NotImplementedError('This method is not implemented yet')

    def evaluate_on_val(self):
        raise NotImplementedError('This method is not implemented yet')

class KeyedDataset(Dataset):

    def __init__(self, **tensors):
        self.tensors = tensors
        assert all(len(t) == len(list(tensors.values())[0]) for t in tensors.values())

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.tensors.items()}

    def __len__(self):
        return len(next(iter(self.tensors.values())))

class BertBaseUncasedOnTextOnly(DisasterTweetsModel):

    def __init__(self):
        super().__init__()
        self.base_model_name = 'bert-base-uncased'
        self.trained_model_path = os.path.join(BASE_DIR, f'../models/{self.model_name}.pt')
        self.learning_rate = 2e-5
        self.num_epochs = 4

    @property
    def model_name(self) -> str:
        return 'bert-base-uncased-on-text-only'

    def __create_tokenized_dataset(self, df: pd.DataFrame):
        tokenizer = BertTokenizer.from_pretrained(self.base_model_name, do_lower_case=True)
        encoded_dict = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
        dataset = KeyedDataset(input_ids=encoded_dict['input_ids'], attention_mask=encoded_dict['attention_mask'], labels=torch.tensor(df['target'].tolist()))
        return dataset

    def __create_dataloader(self, dataset: Dataset, batch_size:int = 32, use_random_sampling:bool = False):
        sampler = RandomSampler(dataset) if use_random_sampling else SequentialSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def __is_model_trained(self):
        return os.path.exists(self.trained_model_path)

    def __load_pretrained_model(self):
        return BertForSequenceClassification.from_pretrained(self.base_model_name, num_labels=2)

    def __load_trained_model(self):
        model = BertForSequenceClassification.from_pretrained(self.base_model_name, num_labels=2)
        model.load_state_dict(torch.load(self.trained_model_path))
        return model

    def __save_trained_model(self, model):
        torch.save(model.state_dict(), self.trained_model_path)

    def train(self, force: bool = False):
        if not force and self.__is_model_trained():
            print('Model already trained. Skipping training.')
            return
        train_df = load_raw_train_df()
        train_dataset = self.__create_tokenized_dataset(train_df)
        train_dataloader = self.__create_dataloader(train_dataset, use_random_sampling=True)
        val_df = load_raw_val_df()
        val_dataset = self.__create_tokenized_dataset(val_df)
        val_dataloader = self.__create_dataloader(val_dataset, use_random_sampling=False)
        model = self.__load_pretrained_model()
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        num_training_steps = self.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        progress = tqdm(range(num_training_steps), desc='Training')

        for epoch in range(self.num_epochs):
            model.train()
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                model.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress.update(1)
                progress.set_postfix({'loss': loss.item()})

            model.eval()
            predictions = []
            targets = []
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())

            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions)
            recall = recall_score(targets, predictions)
            f1 = f1_score(targets, predictions)
            print(f'Epoch {epoch + 1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

        self.__save_trained_model(model)

    def predict(self, df: pd.DataFrame) -> list[int]:
        if not self.__is_model_trained():
            print('Model not trained. Training now...')
            self.train()
        model = self.__load_trained_model()
        model = model.to(self.device)
        dataset = self.__create_tokenized_dataset(df)
        dataloader = self.__create_dataloader(dataset, use_random_sampling=False)
        model.eval()
        predictions = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
        return predictions