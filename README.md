# NumPy Neural Net

This is a small, pedagogical implementation of neural-network building blocks with NumPy. It makes the data flow, gradients, parameters, and optimizer updates visible; it is not intended to reproduce a production framework such as PyTorch.

## Model And Training

A `Model` is an ordered sequence of layers, a differentiable loss, and an optimizer. Training performs:

1. A forward pass to produce predictions.
2. Mean loss computation.
3. A backward pass to produce each layer's input and parameter gradients.
4. An optimizer update for each trainable parameter.

`compute_loss()` reports the mean loss. A loss derivative must therefore already include the corresponding averaging factor. For example, `CrossEntropyLoss.derivative()` returns $(y_{pred} - y_{true}) / B$ for batch size $B$. Layers accumulate parameter gradients from that upstream gradient and do not divide by the batch size again.

## Differentiable Functions And Optimizers

`DifferentiableFunction` represents activations and losses through a function and derivative. Metrics do not need derivatives. Optimizers receive only a layer's `parameter_gradients` and update arrays exposed by `Layer.parameters()`.

Each `backward()` call returns:

```python
LayerGradients(
	input_gradient=...,       # passed to the preceding differentiable layer
	parameter_gradients=...,  # named gradients consumed by the optimizer
)
```

An embedding layer returns `None` for `input_gradient` because token IDs are discrete indices rather than differentiable values.

## Layer Shape Contracts

Let $B$ be batch size, $F$ feature count, $C$ channels, $H$ and $W$ spatial dimensions, $L$ sequence length, and $D$ embedding dimension.

| Layer | Forward input to output | Backward upstream to input gradient | Parameter gradients |
| --- | --- | --- | --- |
| `DenseLayer` | $(B, F) \rightarrow (B, O)$ | $(B, O) \rightarrow (B, F)$ | `weights`: $(F, O)$; `biases`: $(O)$ |
| `CNNLayer` | $(B, C, H, W) \rightarrow (B, C_{out}, H_{out}, W_{out})$ | output shape $\rightarrow$ input shape | `weights`: $(C_{out}, C, K, K)$; `biases`: $(C_{out})$ |
| `FlattenLayer` | $(B, \ldots) \rightarrow (B, N)$ | $(B, N) \rightarrow$ cached input shape | none |
| `ReshapeLayer` | $(B, N) \rightarrow (B, \ldots)$ | reshaped output gradient $\rightarrow$ cached input shape | none |
| `MaxPoolLayer` | $(B, C, H, W) \rightarrow (B, C, H_{out}, W_{out})$ | output shape $\rightarrow$ input shape | none |
| `BatchNormLayer` | $(B, F)$ or $(B, C, H, W) \rightarrow$ same shape | same shape $\rightarrow$ same shape | `gamma`, `beta`: feature shape |
| `LayerNormLayer` | $(\ldots, F) \rightarrow (\ldots, F)$ | same shape $\rightarrow$ same shape | `gamma`, `beta`: $(F)$ |
| `EmbeddingLayer` | integer IDs $(B, L) \rightarrow (B, L, D)$ | $(B, L, D) \rightarrow \texttt{None}$ | `weights`: $(V, D)$ |
| `PositionalEncodingLayer` | $(B, L, D) \rightarrow (B, L, D)$ | $(B, L, D) \rightarrow (B, L, D)$ | none |
| `DotProductAttentionLayer` | $(B, L, D) \rightarrow (B, L, D)$ | $(B, L, D) \rightarrow (B, L, D)$ | `query_weights`, `key_weights`, `value_weights`, `output_weights`: $(D, D)$ |

`DenseLayer`, `CNNLayer`, and `BatchNormLayer` validate their declared feature boundaries. CNN, pool, reshape, and layer-size configuration values are also checked before NumPy operations can fail ambiguously.

## Serialization

Layer, activation, loss, and optimizer registries intentionally accept only known types. Unsupported serialized names raise `ValueError` with the relevant type, rather than a low-level dictionary error.

## Dataset

`Dataset` owns loading, splitting, and preprocessing. The included MNIST dataset supplies batched inputs and labels for the training loop.
