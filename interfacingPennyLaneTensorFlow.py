import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

# Suppress warnings related to casting complex to float
warnings.filterwarnings('ignore', category=UserWarning)

# Define a quantum device
dev = qml.device("default.qubit", wires=2)

# Create a quantum circuit using PennyLane
@qml.qnode(dev, interface="tf", diff_method="backprop")
def quantum_circuit(params):
    qml.Hadamard(wires=0)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RY(params[3], wires=1)
    qml.CNOT(wires=[1, 0])  # Add more entanglement
    qml.RZ(params[4], wires=0)  # Additional gate with trainable parameter
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define a QuantumLayer using TensorFlow
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Quantum parameters (trainable variables)
        self.params = tf.Variable(initial_value=tf.random.uniform([5], 0, np.pi), trainable=True)

    @tf.autograph.experimental.do_not_convert
    @tf.function
    def call(self, inputs):
        # Get the batch size from the inputs
        batch_size = tf.shape(inputs)[0]
        
        # Quantum circuit call
        result = quantum_circuit(self.params)
        result = tf.stack(result)
        result = tf.cast(tf.math.real(result), tf.float32)  # Ensure result is real and float32
        
        # Repeat the result for each item in the batch
        return tf.tile(tf.reshape(result, (1, 2)), [batch_size, 1])

# Define the hybrid quantum-classical model
def hybrid_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(4, activation="relu"),
        QuantumLayer(),  # Quantum layer
        tf.keras.layers.Dense(4, activation="relu"),  # Additional layer
        tf.keras.layers.Dense(2, activation="softmax")  # Output layer for classification
    ])
    return model

# Compile the model
model = hybrid_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Create more varied dummy data
X = np.random.rand(100, 2)  # Generate 100 random 2D inputs
X = (X - np.min(X)) / (np.max(X) - np.min(X)) * np.pi  # Normalize data between 0 and pi
Y = np.random.randint(0, 2, (100, 2))  # Generate corresponding binary labels

# Train the model and capture history
history = model.fit(X, Y, epochs=30, batch_size=32)

# Plot accuracy and loss with higher precision
plt.figure(figsize=(12, 6), dpi=120)  # Higher DPI for better resolution

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', marker='o')
plt.title('Training Accuracy over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.5, 1)  # Adjust this range if needed based on your accuracy values
plt.xticks(range(0, len(history.history['accuracy']), 1))  # Exact number of epochs
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='orange', marker='o')
plt.title('Training Loss over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(range(0, len(history.history['loss']), 1))  # Exact number of epochs
plt.legend()

plt.tight_layout()
plt.show()

