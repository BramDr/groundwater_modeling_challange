import random
import numpy as np
import torch
import torch.utils.data as data
import sklearn.preprocessing as pp


class SequenceDataset(data.Dataset):
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 dates: list = None,
                 cuda: bool = False):
        super(SequenceDataset, self).__init__()

        if len(x.shape) != len(y.shape) != 3:
            raise ValueError(
                "Input and output data shape length is not 3 (sample_len, sequence_len, feature_len)")
        if x.shape[0:2] != y.shape[0:2]:
            raise ValueError(
                "Input and output batch and sequence lengths are not identical")
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.dates = dates
        self.sequence_size = x.shape[1]
                
        self.cuda = False
        if cuda and torch.cuda.is_available():
            self.to_device(device="cuda")
            self.cuda = cuda

    def set_sequence_size(self,
                          sequence_size: int):
        if sequence_size is None:
            self.sequence_size = self.x.shape[1]
        else:
            self.sequence_size = sequence_size
            
        
    def transform(self,
                  x_transformer,
                  y_transformer,
                  fit: bool):
        
        x = self.x.cpu().numpy()
        y = self.y.cpu().numpy()

        if fit:
            x_reshape = np.reshape(a=x, newshape=(-1, x.shape[2])).copy()
            x_transformer.fit(X=x_reshape)
            y_reshape = np.reshape(a=y, newshape=(-1, y.shape[2])).copy()
            y_transformer.fit(X=y_reshape)
        
        x_norm = np.reshape(a=x, newshape=(-1, x.shape[2])).copy()
        x_norm = x_transformer.transform(X = x_norm)
        x_norm = np.reshape(a=x_norm, newshape=x.shape)
        
        y_norm = np.reshape(a=y, newshape=(-1, y.shape[2])).copy()
        y_norm = y_transformer.transform(X = y_norm)
        y_norm = np.reshape(a=y_norm, newshape=y.shape)
        
        self.x = torch.from_numpy(x_norm).float()
        self.y = torch.from_numpy(y_norm).float()
        
        if self.cuda and torch.cuda.is_available():
            self.to_device(device="cuda")
        
        return x_transformer, y_transformer
        

    def to_device(self,
                  device: str) -> None:
        self.x = self.x.to(device=device)
        self.y = self.y.to(device=device)
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,
                    index: int):
        
        seq_sel_end = self.x.shape[1] - self.sequence_size + 1
        
        seq_start = random.randrange(start = 0, stop = seq_sel_end)
        seq_end = seq_start + self.sequence_size
        
        seq_slice = slice(seq_start, seq_end)
        
        x_item = self.x[:, seq_slice, :]
        y_item = self.y[:, seq_slice, :]
        
        return x_item, y_item
    