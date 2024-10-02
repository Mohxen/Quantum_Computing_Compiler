import pennylane as qml
from catalyst import qjit
from jax import numpy as jnp
import matplotlib.pyplot as plt
import optax
import jax

# Define a simpler quantum device
dev = qml.device("lightning.qubit", wires=2)

# Extremely simplified quantum circuit with supported operations
@qjit
@qml.qnode(dev)
def quantum_circuit(params):
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

# Hybrid model
@qjit
def hybrid_model(params, x):
    dense_1 = jnp.tanh(jnp.dot(x, params['dense_1_w']) + params['dense_1_b'])
    quantum_output = quantum_circuit(params['quantum'])
    dense_2 = jnp.tanh(jnp.dot(jnp.array([quantum_output]), params['dense_2_w']) + params['dense_2_b'])
    output = jnp.exp(jnp.dot(dense_2, params['output_w']) + params['output_b'])
    return output

# Initialize parameters
def init_params():
    key = jax.random.PRNGKey(0)
    params = {
        'dense_1_w': jax.random.normal(key, (2, 4)),
        'dense_1_b': jnp.zeros(4),
        'quantum': jnp.array([jnp.pi / 4]),  # Simplified quantum parameter
        'dense_2_w': jax.random.normal(key, (1, 4)),
        'dense_2_b': jnp.zeros(4),
        'output_w': jax.random.normal(key, (4, 2)),
        'output_b': jnp.zeros(2)
    }
    return params

# Define loss function
@qjit
def loss_fn(params, x, y):
    predictions = hybrid_model(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(predictions + 1e-8), axis=1))

# Define the optimizer and training function
optimizer = optax.adam(learning_rate=0.01)

@qjit
def train_step(params, x, y, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Generate dataset
key = jax.random.PRNGKey(0)
X = jax.random.uniform(key, (100, 2)) * jnp.pi
Y = jnp.eye(2)[jax.random.randint(key, (100,), minval=0, maxval=2)]

# Initialize parameters and optimizer state
params = init_params()
opt_state = optimizer.init(params)

# Training loop
num_epochs = 30
batch_size = 32
for epoch in range(num_epochs):
    perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(X))
    X = X[perm]
    Y = Y[perm]
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        params, opt_state, loss = train_step(params, x_batch, y_batch, opt_state)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')
