import pytest
import numpy as np
import pandas as pd
import os
from Dataset import ParquetDataset, MNISTDataset
from PIL import Image
import io


@pytest.fixture
def sample_parquet(tmp_path):
    """Parquet with PNG-encoded PIL images in 'image' column should be parsed by MNISTDataset."""

    images_bytes = []
    labels = []
    for i in range(5):
        img = Image.new("L", (28, 28), color=i * 10)  # simple grayscale image
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        images_bytes.append({"bytes": bio.getvalue()})
        labels.append(i)

    file_path = tmp_path / "mnist_pil.parquet"
    df = pd.DataFrame({"image": images_bytes, "label": labels})
    df.to_parquet(file_path)
    return str(file_path), df


def test_parquet_dataset(sample_parquet):
    file_path, df = sample_parquet
    dataset = ParquetDataset(
        file_path=file_path, feature_cols=["image"], label_col="label"
    )
    assert isinstance(dataset.X, np.ndarray)
    assert isinstance(dataset.y, np.ndarray)
    assert dataset.X.shape[0] == len(df)
    assert dataset.y.shape[0] == len(df)
    # Check that the labels match
    np.testing.assert_array_equal(dataset.y, df["label"].values)


def test_mnist_dataset_pil_images(sample_parquet):
    file_path, df = sample_parquet
    dataset = MNISTDataset(file_path=str(file_path), split="train")
    assert isinstance(dataset.X, np.ndarray)
    assert dataset.X.shape == (5, 28 * 28)
    assert np.issubdtype(dataset.X.dtype, np.floating)
    assert dataset.X.max() <= 1.0 and dataset.X.min() >= 0.0

    assert dataset.y.shape == (5, 10)
    np.testing.assert_array_equal(np.argmax(dataset.y, axis=1), df["label"].values)
