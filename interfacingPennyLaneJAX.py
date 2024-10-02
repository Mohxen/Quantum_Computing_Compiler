import pennylane as qml
from pennylane import numpy as np
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import optax

# Define a quantum device
dev = qml.device("default.qubit", wires=2)

# Create a quantum circuit using PennyLane
@qml.qnode(dev, interface="jax", diff_method="backprop")
def quantum_circuit(params):
    qml.Hadamard(wires=0)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RY(params[3], wires=1)
    qml.CNOT(wires=[1, 0])  # Add more entanglement
    qml.RZ(params[4], wires=0)  # Additional gate with trainable parameter
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

# Define the hybrid quantum-classical model using JAX
def hybrid_model(params, x):
    dense_1 = jax.nn.relu(jnp.dot(x, params['dense_1_w']) + params['dense_1_b'])
    
    # Quantum output from the circuit (make sure it's a 2D array for batch processing)
    quantum_output = jnp.array([quantum_circuit(params['quantum'])])
    
    dense_2 = jax.nn.relu(jnp.dot(quantum_output, params['dense_2_w']) + params['dense_2_b'])
    output = jax.nn.softmax(jnp.dot(dense_2, params['output_w']) + params['output_b'])
    return output

# Initialize parameters for the hybrid model
def init_params():
    key = jax.random.PRNGKey(0)
    params = {
        'dense_1_w': jax.random.normal(key, (2, 4)),
        'dense_1_b': jnp.zeros(4),
        'quantum': jnp.array(jax.random.uniform(key, (5,), minval=0, maxval=np.pi)),
        'dense_2_w': jax.random.normal(key, (2, 4)),
        'dense_2_b': jnp.zeros(4),
        'output_w': jax.random.normal(key, (4, 2)),
        'output_b': jnp.zeros(2)
    }
    return params

# Loss function using binary cross-entropy
def loss_fn(params, x, y):
    predictions = hybrid_model(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(predictions + 1e-8), axis=1))

# Accuracy function
def accuracy_fn(params, x, y):
    predictions = hybrid_model(params, x)
    pred_labels = jnp.argmax(predictions, axis=1)  # Shape should be (batch_size, 2)
    true_labels = jnp.argmax(y, axis=1)  # Ensure true labels are in the same format
    accuracy = jnp.mean(pred_labels == true_labels)
    return accuracy

# Define the optimizer
optimizer = optax.adam(learning_rate=0.01)

# Train the model
@jax.jit
def train_step(params, x, y, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Create more varied dummy data
X = np.random.rand(100, 2)  # Generate 100 random 2D inputs
X = (X - np.min(X)) / (np.max(X) - np.min(X)) * np.pi  # Normalize data between 0 and pi

# One-hot encode the labels
Y = np.eye(2)[np.random.randint(0, 2, 100)]  # Ensure Y is one-hot encoded

# Initialize parameters and optimizer state
params = init_params()
opt_state = optimizer.init(params)

# Training loop
num_epochs = 30
batch_size = 32
loss_history = []
accuracy_history = []

for epoch in range(num_epochs):
    perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(X))
    X = X[perm]
    Y = Y[perm]
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        params, opt_state, loss = train_step(params, x_batch, y_batch, opt_state)
    loss_history.append(loss)
    
    # Calculate accuracy after each epoch
    accuracy = accuracy_fn(params, X, Y)
    accuracy_history.append(accuracy)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 6), dpi=120)

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(accuracy_history, label='Training Accuracy', color='blue', marker='o')
plt.title('Training Accuracy over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.xticks(range(0, num_epochs, 1))
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Training Loss', color='orange', marker='o')
plt.title('Training Loss over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(range(0, num_epochs, 1))
plt.legend()

plt.tight_layout()
plt.show()
