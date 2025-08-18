from qiskit import QuantumCircuit
from ConstantPropagation import ConstantPropagation

qc = QuantumCircuit(5)
qc.h(0)
# qc.x(2)
# qc.x(1)
qc.cx(0, 1)
qc.x(1)
# qc.cx(0, 1)
# qc.cx(0, 2)
# qc.reset(1)

# qc.x(2)
# qc.reset(2)

table, new_qc = ConstantPropagation.propagate(qc)
print(table)
print(new_qc.draw())