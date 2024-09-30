import pennylane as qml
from pennylane import numpy as np

# Define a device: the default qubit simulator with 2 qubits
dev = qml.device('default.qubit', wires=2)


