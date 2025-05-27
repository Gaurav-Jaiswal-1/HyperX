import time
import torch
import tensorflow as tf
from src.hyperx_torch import HyperXActivation as PyTorchHyperX
from src.hyperx_tf import HyperXActivation as TensorFlowHyperX

def benchmark_pytorch():
    activation = PyTorchHyperX(k=1.0)
    x = torch.randn(1000000)

    start = time.time()
    for _ in range(100):
        y = activation(x)
    end = time.time()

    print(f"PyTorch HyperX Time: {end - start:.4f} seconds")

def benchmark_tensorflow():
    activation = TensorFlowHyperX(k=1.0)
    x = tf.random.normal([1000000])

    start = time.time()
    for _ in range(100):
        y = activation(x)
    end = time.time()

    print(f"TensorFlow HyperX Time: {end - start:.4f} seconds")


if __name__ == "__main__":
    benchmark_pytorch()
    benchmark_tensorflow()
