import pennylane as qml
from pennylane import numpy as np

# Define a device
dev = qml.device('default.qubit', wires=2)

# Define the expanded quantum circuit with parameters
# @qml.qnode(dev)
# def expanded_circuit(params):
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 1])
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.RZ(params[2], wires=0)
#     qml.RX(params[3], wires=1)
#     qml.RY(params[4], wires=0)  # Additional parameterized rotation
#     qml.RZ(params[5], wires=1)  # Additional parameterized rotation
#     qml.RY(0.4, wires=0)  # Additional gate for added complexity
#     return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

@qml.qnode(dev)
def simpler_circuit(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define the cost function to optimize
# def cost(params):
#     return expanded_circuit(params)[0]  # Optimizing based on the expectation value of qubit 0
# def cost(params):
#     # exp0, exp1 = expanded_circuit(params)
#     exp0, exp1 = simpler_circuit(params)
#     # This cost encourages qubit 0 to be in state |0⟩ (+1) and qubit 1 in state |1⟩ (-1)
#     # return (exp0 - 0.5)**2 + (exp1 - 0.5)**2  # Relax the desired state
#     return (exp0 - 0.3)**2 + (exp1 - 0.7)**2
def cost(params):
    # Evaluate the circuit to get numerical values
    exp0, exp1 = simpler_circuit(params)
    
    # Convert the outputs to numerical values using qml.math
    exp0_value = qml.math.squeeze(exp0)
    exp1_value = qml.math.squeeze(exp1)
    
    # Perform the cost function with numerical values
    return (exp0_value - 0.3)**2 + (exp1_value - 0.7)**2


# Initialize the parameters
# params = np.array([0.5, 0.1, 0.3, 0.2], requires_grad=True)
# params = np.array([1.0, 0.5, 0.7, 0.9], requires_grad=True)
# params = np.array([0.2, 0.8, 0.3, 0.7], requires_grad=True)
# params = np.array([np.pi / 4, np.pi / 2, np.pi / 3, np.pi / 6, np.pi / 7, np.pi / 5], requires_grad=True)
params = np.random.uniform(low=0, high=2 * np.pi, size=6, requires_grad=True)


# print("Initial result: ", expanded_circuit(params))

# The step size (learning rate) might be too small, meaning the optimizer is not making significant updates to the parameters. 
# You could try increasing the step size in the GradientDescentOptimizer.
# Set up the gradient descent optimizer
# opt = qml.GradientDescentOptimizer(stepsize=0.1)
# opt = qml.GradientDescentOptimizer(stepsize=0.5)
opt = qml.AdamOptimizer(stepsize=0.1)

# Optimization loop for 100 steps
for step in range(100):
    params, cost_val = opt.step_and_cost(cost, params)
    print(f"Step {step}: Cost = {cost_val}, Params = {params}")

grad_fn = qml.grad(cost)
gradients = grad_fn(params)
print("Gradients: ", gradients)

# Output the optimized parameters
print(f"Optimized parameters: {params}")
