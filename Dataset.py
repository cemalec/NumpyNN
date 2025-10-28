import logging
import numpy as np
import pandas as pd
from typing import List,Literal
import io
from PIL import Image
from abc import abstractmethod

logger = logging.getLogger(__name__)

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
            logger.debug(f"Yielding batch from index {start_idx} to {start_idx + batch_size}")
            yield self.X[batch_indices], self.y[batch_indices]

    @abstractmethod
    def preprocess_data(self):
        pass
    @abstractmethod
    def preprocess_labels(self):
        pass

    def preprocess(self):
        self.X = self.preprocess_data()
        self.y = self.preprocess_labels()

    def split(self, train_ratio: float):
        split_idx = int(len(self.X) * train_ratio)
        X_train, y_train = self.X[:split_idx], self.y[:split_idx]
        X_val, y_val = self.X[split_idx:], self.y[split_idx:]
        return self.__class__(X_train, y_train, split='train'), self.__class__(X_val, y_val, split='validation')
    
class ParquetDataset(Dataset):
    def __init__(self, file_path: str, feature_cols: list, label_col: str):
        df = pd.read_parquet(file_path)
        X = df[feature_cols].values
        y = df[label_col].values
        super().__init__(X, y)
        
class MNISTDataset(ParquetDataset):
    def __init__(self, 
                 split: Literal['train','test','validation'] = 'train',
                 feature_cols: List[str] = [ 'image' ],
                 label_col: str = 'label'):
        file_path: str = f'/workspaces/NumpyNN/data/mnist/{split}-00000-of-00001.parquet'
        super().__init__(file_path, 
                         feature_cols,
                         label_col)
        self.split = split
        self.preprocess()

    def preprocess_data(self):
        # Convert byte arrays to numpy arrays
        imgs = self.X[:,0]
        img_arrays = [Image.open(io.BytesIO(img['bytes'])) for img in imgs]
        # Flatten
        flattened_imgs = [np.array(img).reshape(-1) for img in img_arrays]
        # Normalize
        normalized_imgs = np.array(flattened_imgs) / 255.0
        # Return an array of shape (num_samples, 784)
        return np.array(normalized_imgs)
        
    def preprocess_labels(self):
        return np.eye(10)[self.y]  # One-hot encode labels