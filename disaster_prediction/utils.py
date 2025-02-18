import torch
from torch.utils.data import Dataset

class KeyedDataset(Dataset):
    def __init__(self, **tensor_dict):
        self.tensor_dict = tensor_dict

        for key, tensor in tensor_dict.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.size(0) == tensor_dict[next(iter(tensor_dict))].size(0)

    def __len__(self):
        return next(iter(self.tensor_dict.values())).size(0)

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.tensor_dict.items()}