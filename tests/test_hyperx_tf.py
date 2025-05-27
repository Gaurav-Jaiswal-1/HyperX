import tensorflow as tf
from src.hyperx_tf import HyperXActivation

def test_hyperx_tensorflow():
    activation = HyperXActivation(k=1.0)
    x = tf.constant([1.0, -2.0, 3.0, -4.0], dtype=tf.float32)
    y = activation(x)

    assert y.shape == x.shape
    print("TensorFlow Test Passed")

test_hyperx_tensorflow()
