import pytest
from pathlib import Path
from configs.hydrate_yaml import hydrate_model


def _write_yaml(path: Path, content: str):
    path.write_text(content)


def test_hydrate_yaml_creates_model(tmp_path: Path):
    yaml_content = """
layers:
  - type: DenseLayer
    name: fc1
    input_size: 784
    output_size: 128
    activation_function: ReLU
  - type: DenseLayer
    name: fc2
    input_size: 128
    output_size: 64
    activation_function: ReLU
  - type: DenseLayer
    name: fc3
    input_size: 64
    output_size: 10
    activation_function: SoftMax
loss: CrossEntropyLoss
optimizer:
  type: SGD
  learning_rate: 0.001
"""
    cfg_path = tmp_path / "model_config.yaml"
    _write_yaml(cfg_path, yaml_content)

    model = hydrate_model(str(cfg_path))

    # Basic sanity checks for a valid model object
    assert model is not None
    assert hasattr(model, "layers"), "model should have a 'layers' attribute"
    assert isinstance(model.layers, (list, tuple))
    assert len(model.layers) == 3

    for layer in model.layers:
        # layer should expose at least input/output information or params
        assert hasattr(layer, "activation_function") or hasattr(layer, "type")
        assert (
            hasattr(layer, "weights")
            or hasattr(layer, "input_size")
            or hasattr(layer, "output_size")
        )

    # Model should implement forward and a loss/compute method
    assert callable(getattr(model, "forward", None))
    assert (
        callable(getattr(model, "backward", None))
        or callable(getattr(model, "compute_loss", None))
        or hasattr(model, "loss")
    )


def test_hydrate_yaml_invalid_config_raises(tmp_path: Path):
    from configs.hydrate_yaml import hydrate_model

    bad_yaml = "not_a_valid: [ : yaml :::"
    cfg_path = tmp_path / "bad_config.yaml"
    _write_yaml(cfg_path, bad_yaml)

    with pytest.raises(Exception):
        hydrate_model(str(cfg_path))
