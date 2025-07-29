from qiskit import QuantumCircuit
from ConstantPropagation import ConstantPropagation

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.barrier()
qc.cx(0, 2)

table, new_qc = ConstantPropagation.propagate(qc)
print(table)
print(new_qc.draw())