import torch
import tensorflow as tf


from src.hyperx_torch import HyperXActivation as PyTorchHyperX
from src.hyperx_tf import HyperXActivation as TensorFlowHyperX


def test_edge_cases_pytorch():
    activation = PyTorchHyperX(k=1.0)
    x = torch.tensor([[float("inf"), -float("inf")], [0.0, float("nan")]])
    result = activation(x)
    assert not torch.any(torch.isnan(result)), "NaN values found in PyTorch output!"
    assert torch.all(torch.abs(result) < 1e10), "Extreme output values not handled properly!"


def test_edge_cases_tensorflow():
    activation = TensorFlowHyperX(k=1.0)
    x = tf.constant([[float("inf"), -float("inf")], [0.0, float("nan")]])
    result = activation(x)
    assert not tf.reduce_any(tf.math.is_nan(result)), "NaN values found in TensorFlow output!"
    assert tf.reduce_all(tf.math.abs(result) < 1e10), "Extreme output values not handled properly!"


# Test extreme values for TensorFlow
tensorflow_activation = TensorFlowHyperX(k=1.0)
extreme_values_tf = tf.constant([1e-10, -1e-10, 1e10, -1e10])
print("TensorFlow Extreme Values Output:", tensorflow_activation(extreme_values_tf).numpy())

# Test extreme values for TensorFlow
tensorflow_activation = TensorFlowHyperX(k=1.0)
extreme_values_tf = tf.constant([1e-10, -1e-10, 1e10, -1e10])
print("TensorFlow Extreme Values Output:", tensorflow_activation(extreme_values_tf).numpy())

# Test extreme values for PyTorch
pytorch_activation = PyTorchHyperX(k=1.0)
extreme_values_torch = torch.tensor([1e-10, -1e-10, 1e10, -1e10])
print("PyTorch Extreme Values Output:", pytorch_activation(extreme_values_torch))



def test_edge_cases_pytorch():
    activation = PyTorchHyperX(k=1.0)
    x = torch.tensor([[float("inf"), -float("inf")], [0.0, float("nan")]])
    result = activation(x)
    assert not torch.any(torch.isnan(result)), "NaN values found in PyTorch output!"
    assert torch.all(torch.abs(result) < 1e10), "Extreme output values not handled properly!"


def test_edge_cases_tensorflow():
    activation = TensorFlowHyperX(k=1.0)
    x = tf.constant([[float("inf"), -float("inf")], [0.0, float("nan")]], dtype=tf.float32)
    result = activation(x)
    assert not tf.reduce_any(tf.math.is_nan(result)), "NaN values found in TensorFlow output!"
    assert tf.reduce_all(tf.math.abs(result) < 1e10), "Extreme output values not handled properly!"
