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

p1 = 0.1
p2 = 0.1
p3 = 0.1
flip = pauli_error([('X', p1), ('Y', p2), ('Z', p3), ('I', 1-p1-p2-p3)])

noise_flip = NoiseModel()
noise_flip.add_all_qubit_quantum_error(flip, ["u1", "u2"])#, "u3"])#, "u2", "u3"])

def noisy_outcome(circ, shots):
    noisy_simulator = QasmSimulator(noise_model=noise_flip)
    job = execute(circ, noisy_simulator, shots=shots)
    result_flip = job.result()
    counts_flip = result_flip.get_counts(0)
    return counts_flip
def ideal_outcome(circ, shots):
    ideal_simulator = QasmSimulator()
    job = execute(circ, ideal_simulator, shots=shots)
    result_flip = job.result()
    counts_flip = result_flip.get_counts(0)
    return counts_flip

# calibrate noisy u1 gate
from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA

import warnings
warnings.filterwarnings("ignore")

NUM_SHOTS = 10000

iters = 10
params = np.array([0.67370068, 0.37532489, 0.05474931, 0.81144705, 0.81721155, 0.46643128])

c1 = 0
c2 = 0

def get_probability_distribution(counts):
    output_distr = []
    for i in ['00', '01', '10', '11']:
        if i in counts.keys():
            output_distr.append(counts[i] / NUM_SHOTS)
        else:
            output_distr.append(0)
    return output_distr

def cost1(output_distr, target_distr):
    return sum([np.abs(output_distr[i] - target_distr[i]) for i in range(2)])
    
def circ_ideal(ra):
    qc = QuantumCircuit(2)
    qc.initialize([k1, np.sqrt(1-k1*k1)*1j], [0])
    qc.initialize([1-k1, np.sqrt(1-(1-k1)*(1-k1))*1j], [1])
    qc.cu1(ra, 0, 1)
    qc.cu1(ra, 0, 1)
    qc.cu1(ra, 0, 1)
    qc.cu1(ra, 0, 1)
    return qc

for k1 in [0., 0.2, 0.5, 0.8, 1.]:
#for i in range(iters):
#    k1 = np.random.random(1)[0]
    qc = QuantumCircuit(1)
    for ra in np.linspace(np.pi, 0, 5):
        print('----------------------------------------')
        print('Current k1:', k1)
        print('Current angle:', ra)
        qc = circ_ideal(ra)
        qc.measure_all()
        cl = ideal_outcome(qc, NUM_SHOTS)
        target_distr = get_probability_distribution(cl)
        print("Ideal:", target_distr)
        cn = noisy_outcome(qc, NUM_SHOTS)
        noisy_distr = get_probability_distribution(cn)
        print("Noisy:", noisy_distr)
        c1 += cost1(noisy_distr, target_distr)
        print("Noisy cost:", cost1(noisy_distr, target_distr))
        backend = QasmSimulator(noise_model=noise_flip)
        def get_var_form(params):
            qr = QuantumRegister(2, name="q")
            cr = ClassicalRegister(2, name='c')
            qc = QuantumCircuit(qr, cr)
            qc.initialize([k1, np.sqrt(1-k1*k1)*1j], [0])
            qc.initialize([1-k1, np.sqrt(1-(1-k1)*(1-k1))*1j], [1])
            qc.cu1(ra, 0, 1)
            qc.u3(params[0], params[1], params[2], qr[0])
            qc.u3(params[3], params[4], params[5], qr[1])
            qc.cu1(ra, 0, 1)
            qc.u3(params[0], params[1], params[2], qr[0])
            qc.u3(params[3], params[4], params[5], qr[1])
            qc.cu1(ra, 0, 1)
            qc.u3(params[0], params[1], params[2], qr[0])
            qc.u3(params[3], params[4], params[5], qr[1])
            qc.cu1(ra, 0, 1)
            qc.u3(params[0], params[1], params[2], qr[0])
            qc.u3(params[3], params[4], params[5], qr[1])
            qc.measure(qr, cr)
            return qc
        
        qc = get_var_form(params)
        counts = execute(qc, backend, shots=NUM_SHOTS).result().get_counts(qc)
        output_distr = get_probability_distribution(counts)
        print("Cali:", output_distr)
        c2 += cost1(output_distr, target_distr)
        print("Cali cost:", cost1(output_distr, target_distr))
        print("Current parameter:", params)
        print('----------------------------------------')
#print(qiskit.circuit.library.U3Gate(params[0], params[1], params[2]).to_matrix())
print('noisy total cost:',  c1)
print('Cali total cost:',  c2)
