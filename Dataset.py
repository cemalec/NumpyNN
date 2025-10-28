import numpy as np
import pandas as pd
from typing import List,Literal,Optional,Tuple,Iterator
import os
from pathlib import Path
import io
from PIL import Image
from abc import abstractmethod

class Dataset:
    """
    A simple dataset class for handling features and labels.
    Parameters:
        X (np.ndarray): Feature data.
        y (np.ndarray): Labels corresponding to the feature data.
        batch_size (Optional[int]): Optional default batch size for iteration.
        cache_dir (Optional[str]): Directory to use for caching loaded/processed data.
    methods:
        __len__: Returns the number of samples in the dataset.
        __getitem__: Yields batches of data of specified size.
        split: Splits the dataset into training and validation sets based on a given ratio.
        download: Placeholder method for downloading the dataset.
        _cache_exists / _load_cache / _save_cache: helpers for caching.
    """
    def __init__(self, 
                 X: Optional[np.ndarray] = None, 
                 y: Optional[np.ndarray] = None,
                 name: Optional[str] = None,
                 file_path: Optional[str] = None,
                 batch_size: Optional[int] = 32, 
                 cache_dir: Optional[str] = None, 
                 auto_download: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.name = name
        self.file_path = file_path
        self.auto_download = auto_download
        # default cache dir: ./data/cache or $XDG_CACHE_HOME/NumpyNN
        if cache_dir is None:
            self.cache_dir = os.environ.get("CACHE_HOME")
            if self.cache_dir:
                self.cache_dir = Path(self.cache_dir) / "cache" / "default"
            else:
                self.cache_dir = Path.cwd() / "cache" / "default"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if (self.X is None or self.y is None) and (self.file_path is None) and auto_download:
            # subclasses are expected to implement download() and set self.X and self.y
            self.download()
        elif self.file_path is not None and (self.X is None or self.y is None):
            # load from provided file path
            self.download()
        if self.X is None or self.y is None:
            raise ValueError("Dataset requires X and y to be provided or download() to populate them")

    def _cache_exists(self, key: str) -> bool:
        return (self.cache_dir / f"{key}.npz").exists()

    def _load_cache(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        p = self.cache_dir / f"{key}.npz"
        with np.load(p, allow_pickle=True) as d:
            X = d["X"]
            y = d["y"]
        return X, y

    def _save_cache(self, key: str, X: np.ndarray, y: np.ndarray) -> None:
        p = self.cache_dir / f"{key}.npz"
        np.savez_compressed(p, X=X, y=y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, 
                    batch_size: int, 
                    shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
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
        return Dataset(X_train, y_train), Dataset(X_val, y_val)
    
    @abstractmethod
    def download(self):
        pass  # Placeholder for dataset download logic
    
    @abstractmethod
    def load(self):
        pass  # Placeholder for dataset loading logic
    
    @abstractmethod
    def preprocess(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Placeholder preprocessing hook.

        Subclasses may override to implement dataset-specific preprocessing
        (e.g. normalization, reshaping, one-hot encoding). If X and y are
        provided they will be processed and returned; otherwise self.X/self.y
        are returned unchanged.
        """
        if X is None and y is None:
            return self.X, self.y
        return X, y
    
class MNISTDataset(Dataset):
    """
    MNIST dataset loader that reads parquet files from ./data/mnist (or given file_path),
    preprocesses images to flattened pixel arrays (float32, scaled 0-1) and one-hot encodes labels.
    Caches processed arrays using Dataset._save_cache / _load_cache.
    """
    def __init__(self,
                 split: str = 'train',
                 file_path: Optional[str] = None,
                 batch_size: Optional[int] = 32,
                 cache_dir: Optional[str] = None,
                 auto_download: bool = True):
        # initialize base without auto download so we control when download/preprocess happens
        super().__init__(X=None, y=None, batch_size=batch_size, cache_dir=cache_dir, auto_download=False)
        self.split = split
        self._provided_file = file_path
        if auto_download:
            self.download()

    def download(self):
        """
        Load parquet file into a DataFrame, then preprocess into (X, y).
        Falls back to cache if available.
        """
        # determine file path
        if self._provided_file:
            p = Path(self._provided_file)
        else:
            p = Path.cwd() / 'data' / 'mnist' / f'{self.split}-00000-of-00001.parquet'

        cache_key = f"mnist_{p.stem}_{self.split}"
        # try cache first
        if self._cache_exists(cache_key):
            try:
                X_arr, y_arr = self._load_cache(cache_key)
                self.X, self.y = X_arr, y_arr
                return
            except Exception:
                pass

        if not p.exists():
            raise FileNotFoundError(f"MNIST parquet file not found at {p}")

        df = pd.read_parquet(p)
        X_arr, y_arr = self.preprocess(df)  # preprocess accepts DataFrame as X argument
        self.X, self.y = X_arr, y_arr
        try:
            self._save_cache(cache_key, self.X, self.y)
        except Exception:
            # non-fatal if caching fails
            pass

    def preprocess(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input DataFrame or arrays into (X_pixels, y_onehot).
        - If X is a pandas.DataFrame, expects an 'image' column (bytes or nested) and a 'label' column.
        - If X is already an ndarray, applies basic normalization and ensures y is one-hot.
        """
        # Handle DataFrame input (common when reading parquet)
        if isinstance(X, pd.DataFrame):
            df = X
            # image column may be 'image' or similar; prefer 'image'
            if 'image' in df.columns:
                imgs = df['image'].values
                processed_imgs = []
                for item in imgs:
                    # item may be raw bytes, dict-like with 'bytes', or already an array
                    try:
                        if isinstance(item, (bytes, bytearray)):
                            img = Image.open(io.BytesIO(item)).convert('L')
                            arr = np.array(img).reshape(-1)
                        elif hasattr(item, 'get') and 'bytes' in item:
                            b = item.get('bytes')
                            img = Image.open(io.BytesIO(b)).convert('L')
                            arr = np.array(img).reshape(-1)
                        else:
                            # fallback: try to coerce to numpy array
                            arr = np.asarray(item).reshape(-1)
                    except Exception:
                        # last resort: try direct array conversion
                        arr = np.asarray(item).reshape(-1)
                    processed_imgs.append(arr)
                X_arr = np.stack(processed_imgs).astype(np.float32)
            else:
                # If images are stored as multiple numeric columns, take all except 'label'
                feature_cols = [c for c in df.columns if c != 'label']
                X_arr = df[feature_cols].values.astype(np.float32)

            # labels
            if 'label' in df.columns:
                labels = df['label'].values
            elif y is not None:
                labels = y
            else:
                raise ValueError("No 'label' column found in DataFrame and no y provided")

        else:
            # X is ndarray-like
            if X is None or y is None:
                raise ValueError("When passing ndarray to preprocess, both X and y must be provided")
            X_arr = np.asarray(X)
            labels = np.asarray(y)

        # Normalize pixel values if in 0-255 range (heuristic)
        if X_arr.dtype == np.uint8 or X_arr.max() > 1.0:
            X_arr = X_arr.astype(np.float32) / 255.0
        else:
            X_arr = X_arr.astype(np.float32)

        # ensure flattened shape (n_samples, 784) when possible
        if X_arr.ndim == 3:  # (n, H, W)
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.ndim == 4:  # (n, C, H, W)
            X_arr = X_arr.reshape(X_arr.shape[0], -1)

        # One-hot encode labels if they are integer class indices
        labels = np.asarray(labels)
        if labels.ndim == 1 and labels.dtype.kind in 'iu':
            # assume classes in range [0, 9]
            num_classes = int(max(labels.max() + 1, 10))
            y_arr = np.eye(num_classes, dtype=np.float32)[labels]
        elif labels.ndim == 2:
            y_arr = labels.astype(np.float32)
        else:
            # fallback: try to coerce
            y_arr = labels.astype(np.float32)

        return X_arr, y_arr