import tensorflow as tf


class HyperXActivation(tf.keras.layers.Layer):
    def __init__(self, k=1.0, **kwargs):
        """
        HyperX Activation Function for TensorFlow/Keras.
        Args:
            k (float): Initial value of the trainable parameter controlling scaling.
        """
        super(HyperXActivation, self).__init__(**kwargs)
        self.k = tf.Variable(k, trainable=True, dtype=tf.float32)

    def call(self, inputs):
        """
        Forward pass of the HyperX activation function.
        Args:
            inputs (Tensor): Input tensor.
        Returns:
            Tensor: Transformed tensor after applying the activation function.
        """
        return inputs * tf.math.tanh(self.k * inputs)
