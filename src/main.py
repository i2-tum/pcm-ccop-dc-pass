from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from ConstantPropagation import ConstantPropagation

qr = QuantumRegister(5)
cr = ClassicalRegister(5)
qc = QuantumCircuit(qr, cr)
# qc.h(0)
# qc.swap(0, 1)
# qc.x(2)
# qc.x(1)
# qc.cx(0, 1)
# qc.h(1)
# qc.cx(1, 0)
qc.x(1)
qc.h(2)
qc.x(3)
# qc.cx(0, 1)
# qc.h(3)
# qc.cx(0, 2)
# qc.cx(0, 2)
# qc.h(2)
qc.measure(qr, cr)

with qc.if_test((cr, 10)):
    qc.h(4)

# qc.x(2)
# qc.reset(2)

# print("===================== INITIAL CIRC")
# print(qc.draw())
table, new_qc = ConstantPropagation.optimize(qc)
print(table)
# print("===================== PROB CIRC")
print(new_qc.draw())

# print("===================== ISTANCE")
# istnc = ConstantPropagation.generate_istance(new_qc)
# print(istnc)