import pytest
import numpy as np
import pandas as pd
import os
from Dataset import ParquetDataset

@pytest.fixture
def sample_parquet(tmp_path):
    # Create a sample DataFrame
    df = pd.DataFrame({
        'image': [np.random.rand(28*28) for _ in range(5)],
        'label': [0, 1, 2, 3, 4]
    })
    file_path = tmp_path / "sample.parquet"
    df.to_parquet(file_path)
    return str(file_path), df

def test_mnist_train_dataset_init(sample_parquet):
    file_path, df = sample_parquet
    dataset = ParquetDataset(file_path=file_path, feature_cols=['image'], label_col='label')
    assert isinstance(dataset.X, np.ndarray)
    assert isinstance(dataset.y, np.ndarray)
    assert dataset.X.shape[0] == len(df)
    assert dataset.y.shape[0] == len(df)
    # Check that the labels match
    np.testing.assert_array_equal(dataset.y, df['label'].values)