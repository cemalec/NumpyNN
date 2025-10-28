import numpy as np
import pandas as pd
from typing import List,Literal
import io
from PIL import Image
from abc import abstractmethod

class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray,split: Literal['train','test','validation']='train'):
        self.X = X
        self.y = y
        self.split = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def get_batches(self, batch_size: int, shuffle: bool = True):
        indices = np.arange(len(self.X))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield self.X[batch_indices], self.y[batch_indices]

    def split(self, train_ratio: float):
        split_idx = int(len(self.X) * train_ratio)
        X_train, y_train = self.X[:split_idx], self.y[:split_idx]
        X_val, y_val = self.X[split_idx:], self.y[split_idx:]
        return self.__class__(X_train, y_train, split='train'), self.__class__(X_val, y_val, split='validation')

    @abstractmethod
    def download(self):
        pass  # Placeholder for dataset download logic
    
class ParquetDataset(Dataset):
    def __init__(self, file_path: str, feature_cols: list, label_col: str):
        df = pd.read_parquet(file_path)
        X = df[feature_cols].values
        y = df[label_col].values
        super().__init__(X, y)
        
class MNISTDataset(ParquetDataset):
    def __init__(self, 
                 split: Literal['train','test'] = 'train',
                 feature_cols: List[str] = [ 'image' ],
                 label_col: str = 'label'):
        file_path: str = f'/workspaces/NumpyNN/data/mnist/{split}-00000-of-00001.parquet'
        super().__init__(file_path, 
                         feature_cols,
                         label_col)
        self.X = np.array([np.array(Image.open(io.BytesIO(img['bytes']))).reshape(-1) / 255.0 for img in self.X[:,0]])
        self.y = np.eye(10)[self.y]  # One-hot encode labels