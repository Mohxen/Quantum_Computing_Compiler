import pennylane as qml
from pennylane import numpy as np

# Define a device: the default qubit simulator with 2 qubits
dev = qml.device('default.qubit', wires=2)

# Define a QNode with more operations
@qml.qnode(dev)
def expanded_circuit(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # Additional operations
    qml.RZ(params[2], wires=0)
    qml.RX(params[3], wires=1)
    
    # Measure the expectation values of PauliZ on both qubits
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define parameters for the circuit
params = np.array([0.5, 0.1, 0.3, 0.2])

# Run the quantum circuit
result = expanded_circuit(params)
print(f"Result: {result}")
