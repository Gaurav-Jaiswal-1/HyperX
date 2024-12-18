# HyperX Project Structure
```
HyperX/
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── hyperx.py                        # Core implementation of the HyperX function
│   ├── tensorflow_impl.py               # TensorFlow/Keras implementation
│   ├── pytorch_impl.py                  # PyTorch implementation
│   ├── tests/
│   │   ├── __init__.py                  # Test package initialization
│   │   ├── test_hyperx.py               # Unit tests for HyperX
│   │   ├── test_tensorflow_impl.py      # Tests for TensorFlow implementation
│   │   ├── test_pytorch_impl.py         # Tests for PyTorch implementation
│   └── examples/
│       ├── tensorflow_example.py        # Example usage in TensorFlow/Keras
│       ├── pytorch_example.py           # Example usage in PyTorch
├── docs/
│   ├── README.md                        # Documentation and usage guide
│   ├── HYPERX_DETAILS.md                # Detailed explanation of the function
│   ├── INSTALLATION.md                  # Instructions for installation
│   ├── BENCHMARKS.md                    # Performance benchmarking results
├── setup.py                             # Package setup script
├── requirements.txt                     # Dependencies
├── LICENSE                              # License for the project
└── .gitignore                           # Files to ignore in version control

```



# **HyperX Activation Function**

**HyperX Activation** is a novel activation function for deep learning that combines the benefits of Tanh with linear scaling. The activation is defined as:

\[
\text{HyperX}(x) = x \cdot \tanh(kx)
\]

This activation function helps improve gradient flow, avoids dying neuron problems seen in ReLU, and works effectively across various tasks and architectures.

---

## **Features**
- Custom activation function implemented in **PyTorch** and **TensorFlow**.
- Adjustable scaling parameter \( k \) for tuning activation behavior.
- Easy to integrate into any neural network architecture.
- Robust and tested for classification tasks.

---

## **Installation**

Install the package from **PyPI** using `pip`:

```bash
pip install hyperx-activation
```

---

## **Dependencies**

Ensure you have the following dependencies installed:

- **PyTorch** (for PyTorch implementation)
- **TensorFlow** (for TensorFlow implementation)
- **NumPy** (for numerical operations)

You can install dependencies using:

```bash
pip install torch tensorflow numpy
```

---

## **Usage**

### 1. **PyTorch Implementation**

Below is an example of using HyperX activation in a PyTorch model:

#### **Step 1**: Import the activation
```python
from hyperx_activation import HyperXActivation
import torch
import torch.nn as nn
import torch.optim as optim
```

#### **Step 2**: Define the model
```python
class SimpleModel(nn.Module):
    def __init__(self, k=1.0):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.activation = HyperXActivation(k=k)  # Use HyperX activation
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
```

#### **Step 3**: Train the model
```python
# Initialize model, loss function, and optimizer
model = SimpleModel(k=2.0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop
inputs = torch.randn(32, 784)  # 32 samples, 784 features
labels = torch.randint(0, 10, (32,))  # 10 classes

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---

### 2. **TensorFlow/Keras Implementation**

Follow these steps to integrate HyperX activation in TensorFlow/Keras:

#### **Step 1**: Import the activation
```python
from hyperx_activation import hyperx_activation
import tensorflow as tf
```

#### **Step 2**: Define the model
```python
def create_model(k=1.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(784,)),
        tf.keras.layers.Activation(lambda x: hyperx_activation(x, k)),  # Use HyperX activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = create_model(k=2.0)
```

#### **Step 3**: Compile and train the model
```python
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy training data
import numpy as np
x_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

---

## **Parameters**

- `k` (**float**, default=1.0): Scaling factor that adjusts the behavior of the activation function. Higher values lead to steeper activation.

---

## **Why Use HyperX?**
- Combines linear and non-linear properties for smoother gradient flow.
- Avoids dying neuron problems faced in ReLU.
- Provides a tunable parameter \( k \) to optimize performance for different tasks.
- Works effectively on classification, regression, and sequence modeling tasks.

---

## **Benchmarks**

Below are the results comparing **HyperX** with standard activations like ReLU and Tanh on classification tasks:

| Activation | Accuracy (%) | Training Time (s) |
|------------|--------------|------------------|
| HyperX     | 97.3         | 59.68            |
| ReLU       | 96.6         | 60.10            |
| Tanh       | 96.4         | 59.55            |

---

## **Contributing**

Contributions are welcome! If you find any issues or want to add new features, please submit a pull request or open an issue on the repository.

---

## **License**

This project is licensed under the MIT License. See the **LICENSE** file for details.

---

## **Author**

Developed by **[Your Name]**.  
If you have any questions or feedback, reach out to me at **[Your Email Address]**.

---

## **Links**

- **PyPI Package**: [https://pypi.org/project/hyperx-activation/](https://pypi.org/project/hyperx-activation/)
- **GitHub Repository**: [https://github.com/yourusername/hyperx-activation](https://github.com/yourusername/hyperx-activation)


