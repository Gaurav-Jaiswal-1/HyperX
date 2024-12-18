import tensorflow as tf

@tf.function
def hyperx_tf(x, k=1.0):
    """
    TensorFlow implementation of the HyperX Activation Function: x * tanh(kx)
    
    Parameters:
        x (tf.Tensor): Input tensor.
        k (float): Scaling factor for the input (default is 1.0).
    
    Returns:
        tf.Tensor: Output of the activation function.
    """
    return x * tf.tanh(k * x)
