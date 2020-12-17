import numpy as np
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus, SuperOp, Operator
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram
import qiskit

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error

p1 = 0.01
p2 = 0.001
flip1 = pauli_error([('X', p2), ('Y', p2), ('Z', p2), ('I', 1-3*p2)])
flip2 = pauli_error([('XX', p1), ('XY', p1), ('XZ', p1), ('YX', p1), ('YY', p1), ('YZ', p1), ('ZX', p1), ('ZY', p1), ('ZZ', p1), ('IX', p2), ('IY', p2), ('IZ', p2), ('XI', p2), ('YI', p2), ('ZI', p2), ('II', 1-9*p1-6*p2)])

noise_flip = NoiseModel()
noise_flip.add_all_qubit_quantum_error(flip1, ["u1", "u2", "u3"])
noise_flip.add_all_qubit_quantum_error(flip2, ["cx"])
print(noise_flip)
