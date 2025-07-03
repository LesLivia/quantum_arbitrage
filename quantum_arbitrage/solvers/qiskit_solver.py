import numpy as np

r = 0.03
T = [3 / 12]

Ki = [6, 9, 12, 15]

# Define S and delta_S for the matrix A
S = [9, 12, 15, 18]  # Terminal stock prices
delta_S = 3  # (uniform spacing for simplicity)

A = np.zeros((len(S), len(Ki)))  # Initialize the matrix A for terminal stock prices and strike prices
# A is a matrix of size (len(K), len(S)), where each element A[j,i] corresponds to the payoff of the call option

# Calculate the matrix A using the given formula positive[(s_i - k_j)] * delta_S
for i in range(len(S)):
    for j in range(len(Ki)):
        if S[i] > Ki[j]:
            A[j, i] = max(0, (S[i] - Ki[j]) * delta_S)  # Payoff should be non-negative, if S[i] < K[j], it will be 0

print("Terminal stock prices:", S)
print("Strike prices:", Ki)
print("Matrix A:", A)

A_inv = np.linalg.inv(A)  # Inverse of A
print("Inverse of A:\n", A_inv)

# Pretend C_vector is the observed call prices in the market
# C_vector = [26.35, 17.42, 8.93, 2.23]
C_vector = [27.45, 18.45, 10.35, 4.05]
print("Call options prices observed in market:", C_vector)
discount_factor = np.exp(r * T[0])
solution = np.matrix(A_inv) @ np.matrix(C_vector).T
discounted_solution = np.matrix(solution) * discount_factor

print("Discounted solution of density matrix:\n", discounted_solution)
print("Sum of discounted solution:", np.sum(discounted_solution))

# Check if A is Hermitian
is_hermitian = np.allclose(A, A.conj().T)
# print("Is A Hermitian?:", is_hermitian)

# Construct the Hermitian matrix A'
A_dagger = A.conj().T  # Hermitian transpose of A
zero_block = np.zeros_like(A)  # Create zero block of same shape as A
A_prime = np.block([[zero_block, A], [A_dagger, zero_block]])
A_prime.shape
print("Hermitian Matrix A':\n", A_prime)

# Check if A' is Hermitian
is_hermitian_prime = np.allclose(A_prime, A_prime.conj().T)
print("Is A' Hermitian?:", is_hermitian_prime)

# Calculate the eigenvalues of A'
eigenvalues, eigenvectors = np.linalg.eig(A_prime)
print("Eigenvalues of A':", eigenvalues)

# Compute C' = [[C], [0]]
C_prime = np.concatenate((C_vector, np.zeros(len(C_vector))))

# Print C_prime as a column vector
print("C':\n", C_prime.reshape(-1, 1))

# Compute f = A'^{-1} * C'
A_prime_inv = np.linalg.inv(A_prime)
f = A_prime_inv @ C_prime * discount_factor
print(['{:.2f}'.format(v) for v in f])

# Print f (the risk-neutral density)
f = f.flatten()
f = np.round(f, 8)
print("f:\n", f.reshape(-1, 1))

# Imports for Qiskit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Initialize, UnitaryGate, QFT, RYGate
from qiskit.visualization import plot_histogram
from qiskit import *

# Various imports
import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt


def apply_controlled_ry(circ, control_qubits, ancilla, rotation):
    # Create the RY rotation gate
    mry_gate = RYGate(rotation).control(len(control_qubits))

    # Append to the circuit (controls followed by target)
    circ.append(mry_gate, control_qubits + [ancilla])


def postselect(statevector, qubit_number, value: bool):
    """Given a quantum state which is a tensor product of multiple qubits,
    compute the quantum state that would result if a specified qubit was found
    to be 1 or 0.
    (Inspired by https://github.com/nelimee/quantum-hhl-4x4/blob/master/hhl4x4/4x4.py,
    but slightly changed so that it better works)

    :statevector: The quantum state
    :qubit_number: The index of the qubit to be set to 1 or 0
    :value: The value to which the specified qubit is set
    """
    # print("state vector: ", statevector)
    # Define a mask depending on the specified qubit
    mask = 1 << qubit_number

    # Depending on the desired value of the qubit, update the quantum state
    if value:
        array_mask = np.arange(len(statevector)) & mask
        array_mask = (array_mask != 0)
    else:
        array_mask = np.logical_not(np.arange(len(statevector)) & mask)

    # print("array mask: ", array_mask)

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    # print("statevector[array_mask]: ", statevector[array_mask])

    # Return a normalized quantum state
    return normalize(statevector[array_mask])


def hhl(circ, ancilla_reg, clock_reg, input_reg, input_matrix, measurement_reg):
    """Create the HHL circuit for our specific problem.

    :circ: The circuit in which the QPE is performed
    :ancilla_reg: The quantum register holding the ancilla qubit
    :clock_reg: The quantum register holding the eigenvalues
    :input_reg: The quantum register holding the input (what we call |b>)
    :input_matrix: The matrix of the input, we use this to compute normalized input and initialize input_reg's state
    :measurement_reg: The classical register holding the results of the mesurements"""

    # Normalize the input matrix and initialize the normalized input state
    norm_input = input_matrix / np.linalg.norm(input_matrix)
    init_gate = Initialize(norm_input)
    circ.append(init_gate, input_reg)

    circ.barrier()

    # Perform the QPE designed specifically for our problem
    circ.h(clock_reg)

    # Choose t such that \phi \in [0, 1) with t = 2pi * tilde{lambda_max} / (2^n) * lambda_max
    t = np.pi / 42  # t = 0.0752153 (tilde{lambda_max} = 24) ~ np.pi / 42

    # Compute U = e^{i A' t}
    U = expm(1j * A_prime * t)

    for i in range(len(clock_reg)):
        pow_U = np.linalg.matrix_power(U, 2 ** i)  # U^{2^i}
        # Check if the matrix is unitary
        if not np.allclose(np.eye(U.shape[0]), U @ U.conj().T):
            print(f"Matrix U^{2 ** i} is not unitary!")
            break
        label = f"U^{2 ** i}"
        CUGate = UnitaryGate(pow_U, label=label).control(1)
        circ.append(CUGate, [clock_reg[i]] + input_reg[:])

    # Perform Inverse QFT on the clock register
    circ.append(QFT(5).inverse(), clock_reg[:])

    # neg_eigenvalues = [-2.579, -4.022, -10.095, -62.652]
    # pos_eigenvalues = [2.579, 4.022, 10.095, 62.652]
    tilde_lambda = [1, 2, 4, 24]

    circ.barrier()

    # Choose a constant C such that C ≤ tilde_λ_min
    C = 1

    # Calculate rotation angle theta for each encoded eigenvalue
    theta = np.zeros(len(tilde_lambda))

    for i in range(len(tilde_lambda)):
        # For each eigenvalue, calculate phi and multiply it by 2^n to get the basis state representation (in decimal)
        # We assume that the eigenvalues are positive, so we only consider the positive eigenvalues
        # phi = ((pos_eigenvalues[i] * t) / (2 * np.pi))
        # print(f"Decimal state of j for λ{i}={pos_eigenvalues[i]}: {phi* 32}")
        # basis_state = phi * (2 ** len(clock_reg))

        # Calculate the angle for the controlled RY gate based on the encoded eigenvalues
        theta[i] = 2 * np.arcsin(C / np.abs(tilde_lambda[i]))

    # Apply controlled RY gates based on the derived basis states for positive eigenvalues
    # apply_controlled_ry(circ, basis_state, ancilla_reg[0], theta[0])
    apply_controlled_ry(circ, clock_reg[:1], ancilla_reg[0], theta[0])
    apply_controlled_ry(circ, clock_reg[1:2], ancilla_reg[0], theta[1])
    apply_controlled_ry(circ, clock_reg[2:3], ancilla_reg[0], theta[2])
    apply_controlled_ry(circ, clock_reg[3:], ancilla_reg[0], theta[3])

    circ.barrier()

    circ.append(QFT(5), clock_reg[:])

    # Perform the inverse QPE specifically designed for our problem
    U_inv = expm(-1j * A_prime * t)
    for i in reversed(range(len(clock_reg))):
        # Compute U^2^i
        pow_U = np.linalg.matrix_power(U_inv, 2 ** i)  # U^{2^i}
        label = f"U_inv^{2 ** i}"
        CUGate = UnitaryGate(pow_U, label=label).control(1)
        circ.append(CUGate, [clock_reg[i]] + input_reg[:])

    circ.h(clock_reg)

    circ.barrier()


# Create the various registers needed
input_reg = QuantumRegister(3, name='input')
clock_reg = QuantumRegister(5, name='clock')
ancilla_reg = QuantumRegister(1, name='ancilla')
measurement_reg = ClassicalRegister(4, name='c')

# Create an empty circuit with the specified registers
circuit = QuantumCircuit(ancilla_reg, clock_reg, input_reg, measurement_reg)

# Add the HHL circuit to the circuit
hhl(circuit, ancilla_reg, clock_reg, input_reg, C_prime, measurement_reg)

# Add measurements to the original circuit (for running on quantum computers)
circuit.measure(ancilla_reg, measurement_reg[0])
circuit.measure(input_reg[0], measurement_reg[1])
circuit.measure(input_reg[1], measurement_reg[2])
circuit.measure(input_reg[2], measurement_reg[3])

circuit.draw(output='mpl', fold=-1)

sv = circuit.remove_final_measurements(inplace=False)  # no measurements allowed
statevector = Statevector(sv)

full_state = statevector.data
# Get the output vector conditioned on ancilla value being 1
statevector = postselect(full_state, 0, value=1)

# This is hardcoded, in our case, need to select
# element of the statevector to recover the solution
# These are the amplitudes for the value of x being 0 or 1,
# and values all 0 for the clock register, which in the final
# vector is on coordinates 0 and every 32th element after that
# (since we have 5 qubits for clock, and 3 qubits for input,
# the total number of qubits is 8, and the clock qubits
# are the first 5 qubits, while the input qubits are the last 3 qubits).
selector = np.zeros(2 ** 8)  # 2 ^ (clock + input qubits)
selector[0] = 1
selector[32] = 1  # 2 ^ len(clock qubits)
selector[32 * 2] = 1
selector[32 * 3] = 1
selector[32 * 4] = 1
selector[32 * 5] = 1
selector[32 * 6] = 1
selector[32 * 7] = 1
selector = (selector != 0)

# Get the output for qubit containing solution
x_experimental = statevector[selector]

# Multiply the quantum state of solution by the norm of expected solution (hardcoded in our case)
# to get the right answer (recall HHL gives a solution proportional to the true one)
solution = 0.561249 * x_experimental
exp_solution = discount_factor * np.matrix(solution)

amplitudes = np.absolute(full_state) ** 2

print("Exact solution: {}".format(f))
print("Experimental solution: {}".format(exp_solution))
print("Error in found solution: {}".format(np.linalg.norm(exp_solution - f)))

n_states = len(full_state)
x_axis_values = np.arange(n_states)
x_axis_binary = [' '.join(bin(n)[3:])[::-1] for n in np.arange(n_states, 2 * n_states)]

plt.figure(figsize=(12, 5))
plt.grid(color='gray', linewidth=0.2)
plt.bar(x_axis_values, np.real(full_state))
plt.xticks(x_axis_values, x_axis_binary, rotation='vertical')
plt.ylabel('Real amplitude')
plt.xlabel('State')
plt.show()

# Execute the circuit using the simulator
simulator = AerSimulator()
qct = transpile(circuit, simulator)
job = simulator.run(qct, shots=65536)

# Get the result of the execution
result = job.result()

# Get the counts, the frequency of each answer
counts = result.get_counts()

# Divide all counts by the number of shots to get the probabilities
for i in counts:
    counts[i] = counts[i] / 65536

# Display the results
plot_histogram(counts, figsize=(12, 5))

# Extract the probabilities of the solution states in the simulated circuit
# As we added zeros to the end of the C_vector, we need to discard the first
# 4 entries, 000 -> 011, when the ancilla is measured 1
sim_f = [0, 0, 0, 0, 0.001, 0.004, 0.009, 0.02]

# Multiply the simulated probabilities by the discount factor
adjusted_sim_f = discount_factor * np.matrix(sim_f)

# Normalize sim_f to ensure it sums to 1
adjusted_sim_f_norm = adjusted_sim_f / np.sum(adjusted_sim_f)
print("Normalized f:\n", np.round(adjusted_sim_f_norm.reshape(-1, 1), 5))

# Calculate the exact normalized solution
f_norm = f / np.linalg.norm(f)
f_norm_squared = f_norm ** 2
print("Exact solution: \n{}".format(f_norm_squared.reshape(-1, 1)))
