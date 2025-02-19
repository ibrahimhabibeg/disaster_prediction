from typing import Callable
import pandas as pd
import torch
from torch import nn

class ModelSpecs:
    def __init__(self,
                 model_name: str,
                 model: nn.Module,
                 dataset_creator: Callable[[pd.DataFrame, bool], torch.utils.data.Dataset],
                 learning_rate: float = 5e-5,
                 training_epochs: int = 4,
                 batch_size: int = 32):
        self.model_name = model_name
        self.model = model
        self.dataset_creator = dataset_creator
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size

    def __str__(self):
        return f"ModelSpecs(model_name={self.model_name})"

    def __repr__(self):
        return str(self)