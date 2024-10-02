import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a quantum device
dev = qml.device("default.qubit", wires=2)

# Define the quantum circuit using PennyLane
@qml.qnode(dev, interface="torch", diff_method="backprop")
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

# Define the QuantumLayer using PyTorch
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Quantum parameters (trainable variables)
        self.params = nn.Parameter(torch.rand(5, dtype=torch.float32) * np.pi)  # Use float32

    def forward(self, inputs):
        # Call the quantum circuit
        result = quantum_circuit(self.params)
        # Convert the result to a PyTorch tensor and repeat for batch size
        result = torch.stack(result)  # Combine results from PauliZ(0) and PauliZ(1)
        result = result.real  # Ensure the result is real
        # Repeat the result for each item in the batch
        batch_size = inputs.size(0)
        return result.unsqueeze(0).expand(batch_size, -1)

# Define the hybrid quantum-classical model using PyTorch
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(2, 4, dtype=torch.float32)  # Ensure float32 dtype
        self.relu = nn.ReLU()
        self.quantum_layer = QuantumLayer()  # Quantum layer
        self.fc2 = nn.Linear(2, 4, dtype=torch.float32)  # Classical layer after quantum layer
        self.fc3 = nn.Linear(4, 2, dtype=torch.float32)  # Output layer for classification

    def forward(self, x):
        x = x.float()  # Ensure input is float32
        x = self.relu(self.fc1(x))  # Pass through classical layer
        x = self.quantum_layer(x)   # Pass through quantum layer
        x = self.relu(self.fc2(x))  # Another classical layer
        x = torch.softmax(self.fc3(x), dim=1)  # Softmax for classification
        return x

# Instantiate the model
model = HybridModel()

# Loss and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create more varied dummy data, ensuring it's of dtype float32
X = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)  # Ensure data is float32
X = (X - X.min()) / (X.max() - X.min()) * np.pi  # Normalize data between 0 and pi
Y = torch.tensor(np.random.randint(0, 2, (100, 2)), dtype=torch.float32)  # Ensure labels are float32

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, Y)  # Compute loss
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights

    # Print progress
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: Gradient - {param.grad}")

# Specifically, check the gradients of the quantum layer
quantum_layer_params = model.quantum_layer.params
print(f"QuantumLayer params gradients: {quantum_layer_params.grad}")
