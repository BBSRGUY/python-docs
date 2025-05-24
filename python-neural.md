# Python Neural Networks and Deep Learning

This document provides a comprehensive guide to neural network implementations, frameworks, and operations in Python with syntax and usage examples.

## Neural Network Fundamentals (Pure Python)

### Basic Neuron Implementation
```python
import math
import random

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights randomly
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def forward(self, inputs):
        """Forward pass through neuron"""
        if len(inputs) != len(self.weights):
            raise ValueError("Input size doesn't match weight size")
        
        # Weighted sum + bias
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return weighted_sum + self.bias
    
    def activate(self, x, activation='sigmoid'):
        """Apply activation function"""
        if activation == 'sigmoid':
            return 1 / (1 + math.exp(-max(-500, min(500, x))))  # Prevent overflow
        elif activation == 'tanh':
            return math.tanh(x)
        elif activation == 'relu':
            return max(0, x)
        elif activation == 'leaky_relu':
            return x if x > 0 else 0.01 * x
        else:
            return x  # Linear activation

# Example usage
neuron = Neuron(3)
inputs = [0.5, -0.2, 0.8]
output = neuron.forward(inputs)
activated_output = neuron.activate(output, 'sigmoid')
print(f"Raw output: {output:.4f}, Activated: {activated_output:.4f}")
```

### Multi-Layer Perceptron (MLP) from Scratch
```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, weight) + bias
            self.z_values.append(z)
            
            # Apply activation (sigmoid for all layers)
            if i == len(self.weights) - 1:  # Output layer
                activation = self.sigmoid(z)
            else:  # Hidden layers
                activation = self.sigmoid(z)
            
            self.activations.append(activation)
            current_input = activation
        
        return self.activations[-1]
    
    def backward(self, X, y):
        """Backward propagation"""
        m = X.shape[0]
        
        # Calculate output layer error
        output_error = self.activations[-1] - y
        deltas = [output_error]
        
        # Backpropagate errors
        for i in range(len(self.weights) - 1, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights and biases
        for i in range(len(self.weights)):
            weight_gradient = self.activations[i].T.dot(deltas[i]) / m
            bias_gradient = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            self.weights[i] -= self.learning_rate * weight_gradient
            self.biases[i] -= self.learning_rate * bias_gradient
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass
            self.backward(X, y)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

# Example: XOR problem
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    mlp = MLP([2, 4, 1], learning_rate=0.1)
    mlp.train(X, y, epochs=5000, verbose=True)
    
    # Test predictions
    predictions = mlp.predict(X)
    print("\nPredictions:")
    for i, (input_val, target, pred) in enumerate(zip(X, y, predictions)):
        print(f"Input: {input_val}, Target: {target[0]}, Prediction: {pred[0]:.4f}")
```

## TensorFlow and Keras

### Basic Neural Network with TensorFlow/Keras
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)

# Simple feedforward network
def create_simple_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load and preprocess MNIST data
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

# Train the model
def train_mnist_classifier():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    model = create_simple_model()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

# Example usage
# model, history = train_mnist_classifier()
```

### Convolutional Neural Networks (CNN)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a CNN for image classification"""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Advanced CNN with residual connections
def create_residual_block(x, filters, kernel_size=3, stride=1):
    """Create a residual block"""
    # Shortcut connection
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if necessary
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_resnet_model(input_shape=(32, 32, 3), num_classes=10):
    """Create a simple ResNet-like model"""
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = create_residual_block(x, 64)
    x = create_residual_block(x, 64)
    x = create_residual_block(x, 128, stride=2)
    x = create_residual_block(x, 128)
    x = create_residual_block(x, 256, stride=2)
    x = create_residual_block(x, 256)
    
    # Global average pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, x)
    return model

# Transfer learning example
def create_transfer_learning_model(num_classes=10):
    """Create a model using transfer learning with VGG16"""
    base_model = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### Recurrent Neural Networks (RNN)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_lstm_model(vocab_size, embedding_dim=128, lstm_units=64, max_length=100):
    """Create an LSTM model for text processing"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2),
        layers.LSTM(lstm_units, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_bidirectional_lstm(vocab_size, embedding_dim=128, lstm_units=64, max_length=100):
    """Create a bidirectional LSTM model"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=0.2)),
        layers.Bidirectional(layers.LSTM(lstm_units, dropout=0.2)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_gru_model(vocab_size, embedding_dim=128, gru_units=64, max_length=100):
    """Create a GRU model"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.GRU(gru_units, return_sequences=True, dropout=0.2),
        layers.GRU(gru_units, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Sequence-to-sequence model
def create_seq2seq_model(input_vocab_size, output_vocab_size, embedding_dim=256, lstm_units=512):
    """Create a sequence-to-sequence model for translation"""
    # Encoder
    encoder_inputs = keras.Input(shape=(None,))
    encoder_embedding = layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = layers.LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = keras.Input(shape=(None,))
    decoder_embedding = layers.Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = layers.Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
```

### Attention Mechanisms and Transformers
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        attn_output, _ = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

def create_transformer_model(vocab_size, d_model=128, num_heads=8, dff=512, num_blocks=4, max_length=100):
    """Create a transformer model for classification"""
    inputs = keras.Input(shape=(max_length,))
    
    # Embedding and positional encoding
    embedding = layers.Embedding(vocab_size, d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    # Add positional encoding
    pos_encoding = positional_encoding(max_length, d_model)
    embedding += pos_encoding[:, :max_length, :]
    
    x = layers.Dropout(0.1)(embedding)
    
    # Transformer blocks
    for i in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, dff)(x)
    
    # Global average pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def positional_encoding(position, d_model):
    """Create positional encoding"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
```

## PyTorch Neural Networks

### Basic PyTorch Neural Network
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.6f}')

# Example usage
if __name__ == "__main__":
    # Create sample data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 3, (1000,))
    
    # Create data loader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleNN(10, 64, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_model(model, train_loader, criterion, optimizer)
```

### PyTorch CNN Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ResNet block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### PyTorch RNN/LSTM Implementation
```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=0.3 if n_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last output for classification
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        
        return output

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, bidirectional=True,
                           dropout=0.3 if n_layers > 1 else 0)
        
        # Fully connected layer (hidden_dim * 2 because of bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last output for classification
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        
        return output

# Attention mechanism for LSTM
class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.dropout(context_vector)
        output = self.fc(output)
        
        return output
```

## Scikit-learn Neural Networks

### Multi-layer Perceptron with Scikit-learn
```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.datasets import make_classification, make_regression
import numpy as np

# Classification example
def sklearn_mlp_classification():
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                              n_informative=15, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = mlp.predict(X_test_scaled)
    
    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {mlp.score(X_test_scaled, y_test):.4f}")
    
    return mlp

# Regression example
def sklearn_mlp_regression():
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = mlp.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {mlp.score(X_test_scaled, y_test):.4f}")
    
    return mlp

# Hyperparameter tuning
def tune_mlp_hyperparameters():
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Grid search
    mlp = MLPClassifier(max_iter=500, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_score:.4f}")
    
    return best_model

# Example usage
# classifier = sklearn_mlp_classification()
# regressor = sklearn_mlp_regression()
# best_model = tune_mlp_hyperparameters()
```

## Activation Functions

### Custom Activation Functions
```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid function"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh function"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Derivative of Leaky ReLU function"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def swish(x):
        """Swish activation function"""
        return x * ActivationFunctions.sigmoid(x)
    
    @staticmethod
    def swish_derivative(x):
        """Derivative of Swish function"""
        s = ActivationFunctions.sigmoid(x)
        return s + x * s * (1 - s)
    
    @staticmethod
    def mish(x):
        """Mish activation function"""
        return x * np.tanh(ActivationFunctions.softplus(x))
    
    @staticmethod
    def softplus(x):
        """Softplus activation function"""
        return np.log(1 + np.exp(np.clip(x, -500, 500)))
    
    @staticmethod
    def gelu(x):
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x, axis=-1):
        """Softmax activation function"""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Visualization of activation functions
def plot_activation_functions():
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'Sigmoid': ActivationFunctions.sigmoid(x),
        'Tanh': ActivationFunctions.tanh(x),
        'ReLU': ActivationFunctions.relu(x),
        'Leaky ReLU': ActivationFunctions.leaky_relu(x),
        'Swish': ActivationFunctions.swish(x),
        'GELU': ActivationFunctions.gelu(x)
    }
    
    plt.figure(figsize=(12, 8))
    for i, (name, y) in enumerate(activations.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(x, y, label=name)
        plt.title(name)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# plot_activation_functions()
```

## Loss Functions

### Custom Loss Functions
```python
import numpy as np
import tensorflow as tf

class LossFunctions:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """Mean Absolute Error loss"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
        """Binary cross-entropy loss"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
        """Categorical cross-entropy loss"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred), axis=1).mean()
    
    @staticmethod
    def sparse_categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
        """Sparse categorical cross-entropy loss"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """Huber loss (smooth L1 loss)"""
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        small_error_loss = 0.5 * error ** 2
        large_error_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, small_error_loss, large_error_loss))
    
    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, epsilon=1e-8):
        """Focal loss for addressing class imbalance"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        pt = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        focal_weight = alpha_t * (1 - pt) ** gamma
        return -np.mean(focal_weight * np.log(pt))

# TensorFlow/Keras custom loss functions
def custom_keras_losses():
    def focal_loss_keras(alpha=0.25, gamma=2.0):
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            focal_weight = alpha_t * tf.pow(1 - pt, gamma)
            focal_loss = -focal_weight * tf.log(pt)
            
            return tf.reduce_mean(focal_loss)
        return focal_loss_fn
    
    def dice_loss(y_true, y_pred, smooth=1):
        """Dice loss for segmentation tasks"""
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        dice_coeff = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_coeff
    
    def contrastive_loss(margin=1.0):
        """Contrastive loss for siamese networks"""
        def contrastive_loss_fn(y_true, y_pred):
            square_pred = tf.square(y_pred)
            margin_square = tf.square(tf.maximum(margin - y_pred, 0))
            return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
        return contrastive_loss_fn
    
    return focal_loss_keras, dice_loss, contrastive_loss
```

## Optimization Algorithms

### Custom Optimizers
```python
import numpy as np

class CustomOptimizers:
    class SGD:
        def __init__(self, learning_rate=0.01, momentum=0.0):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.velocity = {}
        
        def update(self, params, grads, layer_name):
            if layer_name not in self.velocity:
                self.velocity[layer_name] = np.zeros_like(params)
            
            self.velocity[layer_name] = (self.momentum * self.velocity[layer_name] - 
                                       self.learning_rate * grads)
            return params + self.velocity[layer_name]
    
    class Adam:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = {}  # First moment
            self.v = {}  # Second moment
            self.t = 0   # Time step
        
        def update(self, params, grads, layer_name):
            self.t += 1
            
            if layer_name not in self.m:
                self.m[layer_name] = np.zeros_like(params)
                self.v[layer_name] = np.zeros_like(params)
            
            # Update biased first moment estimate
            self.m[layer_name] = self.beta1 * self.m[layer_name] + (1 - self.beta1) * grads
            
            # Update biased second raw moment estimate
            self.v[layer_name] = self.beta2 * self.v[layer_name] + (1 - self.beta2) * (grads ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[layer_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[layer_name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            update = self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            return params - update
    
    class RMSprop:
        def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.rho = rho
            self.epsilon = epsilon
            self.cache = {}
        
        def update(self, params, grads, layer_name):
            if layer_name not in self.cache:
                self.cache[layer_name] = np.zeros_like(params)
            
            self.cache[layer_name] = (self.rho * self.cache[layer_name] + 
                                    (1 - self.rho) * grads ** 2)
            
            update = self.learning_rate * grads / (np.sqrt(self.cache[layer_name]) + self.epsilon)
            return params - update

# Learning rate scheduling
class LearningRateScheduler:
    @staticmethod
    def exponential_decay(initial_lr, decay_rate, step):
        """Exponential decay: lr = initial_lr * decay_rate^step"""
        return initial_lr * (decay_rate ** step)
    
    @staticmethod
    def step_decay(initial_lr, drop_rate, epochs_drop, epoch):
        """Step decay: lr drops by drop_rate every epochs_drop epochs"""
        return initial_lr * (drop_rate ** np.floor(epoch / epochs_drop))
    
    @staticmethod
    def cosine_annealing(initial_lr, min_lr, T_max, epoch):
        """Cosine annealing: lr follows cosine curve"""
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    
    @staticmethod
    def warm_restart(initial_lr, T_0, T_mult, epoch):
        """Cosine annealing with warm restarts"""
        T_cur = epoch
        T_i = T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= T_mult
        return initial_lr * (1 + np.cos(np.pi * T_cur / T_i)) / 2
```

## Regularization Techniques

### Regularization Implementation
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dropout implementation
class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
    
    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask

# Batch normalization implementation
class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
    
    def forward(self, x, training=True):
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1:])
            self.running_var = np.ones(x.shape[1:])
            self.gamma = np.ones(x.shape[1:])
            self.beta = np.zeros(x.shape[1:])
        
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            # Use running statistics
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta

# Layer normalization
class LayerNormalization:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
    
    def forward(self, x):
        if self.gamma is None:
            self.gamma = np.ones(x.shape[-1])
            self.beta = np.zeros(x.shape[-1])
        
        # Calculate statistics along the last axis
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta

# Weight regularization
def l1_regularization(weights, lambda_l1):
    """L1 regularization (Lasso)"""
    return lambda_l1 * np.sum(np.abs(weights))

def l2_regularization(weights, lambda_l2):
    """L2 regularization (Ridge)"""
    return lambda_l2 * np.sum(weights ** 2)

def elastic_net_regularization(weights, lambda_l1, lambda_l2):
    """Elastic Net regularization (L1 + L2)"""
    return l1_regularization(weights, lambda_l1) + l2_regularization(weights, lambda_l2)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model_weights):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model_weights.copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

## Model Evaluation and Metrics

### Neural Network Metrics
```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

class NeuralNetworkMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, average='macro'):
        """Calculate precision"""
        return precision_score(y_true, y_pred, average=average)
    
    @staticmethod
    def recall(y_true, y_pred, average='macro'):
        """Calculate recall"""
        return recall_score(y_true, y_pred, average=average)
    
    @staticmethod
    def f1_score_metric(y_true, y_pred, average='macro'):
        """Calculate F1 score"""
        return f1_score(y_true, y_pred, average=average)
    
    @staticmethod
    def top_k_accuracy(y_true, y_pred_proba, k=5):
        """Calculate top-k accuracy"""
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_pred[i] for i in range(len(y_true))])
    
    @staticmethod
    def perplexity(y_true, y_pred_proba, epsilon=1e-15):
        """Calculate perplexity for language models"""
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        cross_entropy = -np.mean([np.log(y_pred_proba[i, y_true[i]]) for i in range(len(y_true))])
        return np.exp(cross_entropy)
    
    @staticmethod
    def bleu_score(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
        """Simplified BLEU score calculation"""
        from collections import Counter
        
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        scores = []
        for n, weight in enumerate(weights, 1):
            ref_ngrams = Counter(get_ngrams(reference, n))
            cand_ngrams = Counter(get_ngrams(candidate, n))
            
            overlap = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            
            if total == 0:
                scores.append(0)
            else:
                scores.append(overlap / total)
        
        # Brevity penalty
        bp = min(1, len(candidate) / len(reference)) if len(reference) > 0 else 0
        
        # Geometric mean of n-gram precisions
        if all(score > 0 for score in scores):
            bleu = bp * np.exp(sum(w * np.log(s) for w, s in zip(weights, scores)))
        else:
            bleu = 0
        
        return bleu

# Model evaluation utilities
class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.history = {}
    
    def evaluate_classification(self, X_test, y_test, class_names=None):
        """Comprehensive classification evaluation"""
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        # ROC AUC (for multi-class)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': auc,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        return results
    
    def plot_training_history(self, history):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Plot training & validation loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
```

## Advanced Neural Network Architectures

### Generative Adversarial Networks (GANs)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SimpleGAN:
    def __init__(self, latent_dim=100, img_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Build the combined model (for training generator)
        z = keras.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        
        self.combined = keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')
    
    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(256, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(np.prod(self.img_shape), activation='tanh'),
            layers.Reshape(self.img_shape)
        ])
        
        return model
    
    def build_discriminator(self):
        model = keras.Sequential([
            layers.Flatten(input_shape=self.img_shape),
            
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def train(self, X_train, epochs, batch_size=128, sample_interval=50):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Print progress
            if epoch % sample_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
```

### Variational Autoencoders (VAE)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE:
    def __init__(self, input_shape, latent_dim=20):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build encoder, decoder, and VAE
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()
    
    def build_encoder(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Mean and log variance for latent distribution
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        return keras.Model(inputs, [z_mean, z_log_var], name='encoder')
    
    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        outputs = layers.Reshape(self.input_shape)(x)
        
        return keras.Model(latent_inputs, outputs, name='decoder')
    
    def sampling(self, args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def build_vae(self):
        inputs = keras.Input(shape=self.input_shape)
        z_mean, z_log_var = self.encoder(inputs)
        z = layers.Lambda(self.sampling, name='z')([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        
        vae = keras.Model(inputs, reconstructed, name='vae')
        
        # Add KL divergence regularization loss
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        vae.add_loss(kl_loss)
        
        return vae
    
    def compile_and_fit(self, x_train, epochs=50, batch_size=128):
        self.vae.compile(optimizer='adam', loss='mse')
        history = self.vae.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history
```

---

*This document covers comprehensive neural network implementations and operations in Python including fundamental concepts, major frameworks (TensorFlow/Keras, PyTorch, Scikit-learn), activation functions, loss functions, optimization, regularization, evaluation metrics, and advanced architectures. For the most up-to-date information, refer to the official documentation of the respective frameworks.*