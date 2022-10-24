from typing import Optional

import numpy as np
import torch
import torch.utils.data as data


class SequenceDataset(data.Dataset):
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 samples: list,
                 sequences: list,
                 in_features: list,
                 out_features: list,
                 sequence_size: Optional[int] = None,
                 cuda: bool = False):
        super(SequenceDataset, self).__init__()

        if len(x.shape) != len(y.shape) != 3:
            raise ValueError(
                "Input and output data shape length is not 3 (sample_len, sequence_len, feature_len)")
        if x.shape[0:2] != y.shape[0:2]:
            raise ValueError(
                "Input and output batch and sequence lengths are not identical")
        
        self.samples = samples
        self.sequences = sequences
        self.in_features = in_features
        self.out_features = out_features
        
        self.x = []
        self.y = []
        
        sequence_len = x.shape[1]
        if sequence_size is None:
            sequence_size = sequence_len
        
        for sequence_start in range(0, sequence_len, sequence_size):
            sequence_end = sequence_start + sequence_size

            if sequence_end > sequence_len:
                sequence_end = sequence_len

            batch_x = x
            batch_y = y
            batch_x = batch_x[:, sequence_start:sequence_end, :]
            batch_y = batch_y[:, sequence_start:sequence_end, :]
            batch_x = batch_x.copy()
            batch_y = batch_y.copy()
            batch_x = torch.from_numpy(batch_x).float()
            batch_y = torch.from_numpy(batch_y).float()

            self.x.append(batch_x)
            self.y.append(batch_y)
                
        if cuda and torch.cuda.is_available():
            self.to_device(device="cuda")

    def to_device(self,
                  device: str) -> None:
        self.x = [x.to(device=device) for x in self.x]
        self.y = [y.to(device=device) for y in self.y]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,
                    index: int):
        x_item = self.x[index]
        y_item = self.y[index]
        return x_item, y_item
    