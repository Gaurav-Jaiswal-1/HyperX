import tensorflow as tf
from src.hyperx_tf import HyperXActivation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(10,)),
    HyperXActivation(k=1.0),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
