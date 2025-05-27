import tensorflow as tf
from src.hyperx_tf import HyperXActivation

# Instantiate the activation function
activation = HyperXActivation(k=1.0)

# Input tensor
x = tf.constant([1.0, -2.0, 3.0, -4.0], dtype=tf.float32)

# Forward pass
y = activation(x)
print("Output:", y)
