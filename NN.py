import random
import numpy as np
from abc import ABC, abstractmethod

"""Neural network parameter with value and gradient for backpropagation"""
class Parameter:
  def __init__(self, value):
    """Initialize parameter with value and zero gradient"""
    if not isinstance(value, (int, float, np.number)):
      raise ValueError("Value must be numeric")
    self._value = float(value)
    self._grad = 0.0
  
  @property
  def value(self):
    return self._value

  @property
  def grad(self):
    return self._grad
  
  def add_grad(self, amount):
    """Accumulate gradient from backpropagation"""
    if not isinstance(amount, (int, float, np.number)):
      raise ValueError("Amount must be numeric")
    self._grad += float(amount)

  def zero_grad(self):
    """Reset gradient to zero before next backward pass"""
    self._grad = 0.0
  
  def apply_update(self, lr):
    """Update parameter value using gradient descent"""
    if not isinstance(lr, (int, float, np.number)):
      raise ValueError("Learning rate must be numeric")
    if lr < 0:
      raise ValueError("Learning rate must be positive")
    self._value -= lr * self._grad

"""Abstract base class for neural network layers"""
class Layer(ABC):
  @abstractmethod
  def forward(self, x):
    """Forward pass through the layer"""
    pass
  @abstractmethod
  def backward(self, x, grad_output):
    """Backward pass computing gradients"""
    pass
  @abstractmethod
  def parameters(self):
    """Return list of trainable parameters"""
    pass

"""Fully connected (linear) layer with weights and biases"""
class Dense(Layer):
  def __init__(self, input_size, output_size):
    """Initialize weights randomly and biases to zero"""
    self._Weights = np.array([[Parameter(np.random.uniform(-0.1, 0.1)) for _ in range(input_size)] for _ in range(output_size)])
    self._biases = np.array([Parameter(0.0) for _ in range(output_size)])
  
  def forward(self, x):
    """Compute output = weights @ input + biases"""
    x = np.array(x, dtype=np.float64)
    output = np.zeros(len(self._Weights), dtype=np.float64)
    for i in range(len(self._Weights)):
      for j in range(len(self._Weights[0])):
        output[i] += self._Weights[i][j].value * x[j]
      output[i] += self._biases[i].value
    return output

  def parameters(self):
    """Return all weight and bias parameters"""
    return [p for row in self._Weights for p in row] + list(self._biases)

  def backward(self, x, grad_output):
    """Compute gradients for weights, biases, and input"""
    x = np.array(x, dtype=np.float64)
    grad_output = np.array(grad_output, dtype=np.float64)
    grad_input = np.zeros(len(x), dtype=np.float64)
    for i in range(len(self._Weights)):
      for j in range(len(self._Weights[0])):
        self._Weights[i][j].add_grad(grad_output[i] * x[j])
        grad_input[j] += self._Weights[i][j].value * grad_output[i]
      self._biases[i].add_grad(grad_output[i])
    return grad_input

"""Rectified Linear Unit activation function"""
class ReLU(Layer):
  def __init__(self):
    """Store input for backward pass"""
    self._input = None
  
  def forward(self, x):
    """Apply ReLU: max(0, x)"""
    x = np.array(x, dtype=np.float64)
    self._input = x.copy()
    return np.maximum(0, x)
  
  def parameters(self):
    """ReLU has no trainable parameters"""
    return []
  
  def backward(self, x, grad_output):
    """Gradient is 1 where input > 0, else 0"""
    grad_output = np.array(grad_output, dtype=np.float64)
    return grad_output * (self._input > 0).astype(np.float64)

"""Sigmoid activation function for binary classification"""
class Sigmoid(Layer):
  def __init__(self):
    """Store output for backward pass"""
    self._output = None
  
  def forward(self, x):
    """Apply sigmoid: 1 / (1 + exp(-x))"""
    x = np.array(x, dtype=np.float64)
    self._output = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return self._output
  
  def parameters(self):
    """Sigmoid has no trainable parameters"""
    return []

  def backward(self, x, grad_output):
    """Gradient: output * (1 - output)"""
    grad_output = np.array(grad_output, dtype=np.float64)
    return grad_output * self._output * (1.0 - self._output)

"""Mean Squared Error loss function"""
class MSE:
  def __init__(self):
    """Store predictions and targets for backward pass"""
    self._pred = None
    self._target = None
  
  def forward(self, pred, target):
    """Compute mean squared error"""
    pred = np.array(pred, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    self._pred = pred
    self._target = target
    return np.mean((pred - target) ** 2)
  
  def backward(self):
    """Gradient of MSE with respect to predictions"""
    batch_size = len(self._pred) if self._pred.ndim > 0 else 1
    return 2.0 * (self._pred - self._target) / batch_size

"""Binary Cross Entropy loss for binary classification"""
class BinaryCrossEntropy:
  def __init__(self):
    """Store predictions and targets for backward pass"""
    self._pred = None
    self._target = None
  
  def forward(self, pred, target):
    """Compute binary cross entropy loss"""
    pred = np.array(pred, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    self._pred = pred
    self._target = target
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
  
  def backward(self):
    """Gradient of BCE with respect to predictions"""
    pred = np.clip(self._pred, 1e-15, 1 - 1e-15)
    batch_size = len(self._pred) if self._pred.ndim > 0 else 1
    return -(self._target / pred - (1 - self._target) / (1 - pred)) / batch_size

"""Stochastic Gradient Descent optimizer"""
class SGD:
  def __init__(self, parameters, lr=0.01):
    """Initialize optimizer with parameters and learning rate"""
    self.parameters = parameters
    self.lr = lr
  
  def step(self):
    """Update all parameters using accumulated gradients"""
    for param in self.parameters:
      param.apply_update(self.lr)
  
  def zero_grad(self):
    """Reset all parameter gradients to zero"""
    for param in self.parameters:
      param.zero_grad()

"""Neural network model combining multiple layers"""
class Model:
  def __init__(self, layers):
    """Initialize model with list of layers"""
    self.layers = layers
  
  def forward(self, x):
    """Forward pass through all layers"""
    x = np.array(x, dtype=np.float64)
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def parameters(self):
    """Collect all parameters from all layers"""
    params = []
    for layer in self.layers:
      params += layer.parameters()
    return params

  def backward(self, x, grad_output):
    """Backward pass through all layers in reverse"""
    x = np.array(x, dtype=np.float64)
    activations = [x]
    for layer in self.layers:
      x = layer.forward(x)
      activations.append(x)
    grad = np.array(grad_output, dtype=np.float64)
    for i in reversed(range(len(self.layers))):
      grad = self.layers[i].backward(activations[i], grad)
    return grad

  def train_step(self, x, y, loss_fn, optimizer):
    """Single training step: forward, loss, backward, update"""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    optimizer.zero_grad()
    pred = self.forward(x)
    loss = loss_fn.forward(pred, y)
    grad_output = loss_fn.backward()
    self.backward(x, grad_output)
    optimizer.step()
    return loss

  def fit(self, X, y, loss_fn, optimizer, epochs=100, verbose=True):
    """Train model for specified number of epochs"""
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    for epoch in range(epochs):
      total_loss = 0.0
      for i in range(len(X)):
        loss = self.train_step(X[i], y[i], loss_fn, optimizer)
        total_loss += loss
      avg_loss = total_loss / len(X)
      if verbose and (epoch + 1) % (epochs // 10) == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
