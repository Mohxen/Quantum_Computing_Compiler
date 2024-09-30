import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define a device with 2 qubits (you can increase the number of qubits for more complex circuits)
dev = qml.device('default.qubit', wires=2)

# Define the expanded quantum circuit with parameters
@qml.qnode(dev)
def expanded_circuit(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=0)
    qml.RX(params[3], wires=1)
    qml.RY(params[4], wires=0)  # Additional parameterized rotation
    qml.RZ(params[5], wires=1)  # Additional parameterized rotation
    qml.RY(0.4, wires=0)        # Additional gate for added complexity
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define the simpler quantum circuit
@qml.qnode(dev)
def simpler_circuit(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define the cost function to optimize
def cost(params):
    # Evaluate the circuit to get numerical values
    exp0, exp1 = expanded_circuit(params)  # You can switch this with simpler_circuit(params) if desired
    
    # Convert the outputs to numerical values using qml.math
    exp0_value = qml.math.squeeze(exp0)
    exp1_value = qml.math.squeeze(exp1)
    
    # Perform the cost function with numerical values
    return (exp0_value - 0.3)**2 + (exp1_value - 0.7)**2

# Initialize the parameters with random values
params = np.random.uniform(low=0, high=2 * np.pi, size=6, requires_grad=True)

# Set up the optimizer
opt = qml.AdamOptimizer(stepsize=0.1)

# Variables to store the optimization progress
costs = []

# Optimization loop for 100 steps
for step in range(100):
    params, cost_val = opt.step_and_cost(cost, params)
    costs.append(cost_val)
    print(f"Step {step}: Cost = {cost_val}, Params = {params}")

# Compute the gradients
grad_fn = qml.grad(cost)
gradients = grad_fn(params)
print("Gradients: ", gradients)

# Output the optimized parameters
print(f"Optimized parameters: {params}")

# Plot the cost function over the steps to visualize optimization progress
plt.plot(costs)
plt.xlabel("Step")
plt.ylabel("Cost")
plt.title("Cost Function Optimization")
plt.show()
