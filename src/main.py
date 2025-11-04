from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from ConstantPropagation import ConstantPropagation
from qiskit.circuit.classical import expr

qr = QuantumRegister(6)
cr = ClassicalRegister(6)
qc = QuantumCircuit(qr, cr)

qc.h(0)
qc.cx(0, 1)

qc.measure(qr[0], cr[0])

with qc.if_test((cr[0], 1)) as else_:
    qc.x(5)
    qc.h(2)

with else_:
    qc.x(3)
    qc.h(4)

qc.reset(qr[1])
qc.measure(qr[1], cr[1])

with qc.if_test((cr[1], 0)) as else_:
    qc.h(3)
with else_:
    qc.x(4)

# qc.swap(0, 1)
# qc.x(2)
# qc.x(1)
# qc.cx(0, 1)
# qc.h(1)
# qc.cx(1, 0)
# qc.x(1)
# qc.h(2)
# qc.cx(2,3)
# qc.cx(0, 1)
# qc.h(3)
# qc.cx(0, 2)
# qc.cx(0, 2)
# qc.h(2)
# qc.measure(qr[1], cr[1])
# qc.measure(qr[2], cr[2])
# qc.measure(qr[3], cr[3])

# cond = expr.bit_and(cr[1], (cr[2]))

# with qc.if_test(cond):
#     qc.h(4)
# with qc.if_test((cr[2], 1)):
#     qc.h(4)
#     qc.x(5)

# qc.x(2)
# qc.reset(2)

# qc.h(qr[0])
# # qc.cx(qr[0], qr[1])
# qc.measure(qr[0], cr[0])
# with qc.if_test((cr[0], 1)):
#     qc.z(qr[3])

print("######################### INITIAL CIRC")
print(qc.draw())
table, new_qc = ConstantPropagation.optimize(qc, max_ent_group_size=10)
print(table)
print("######################### PROB CIRC")
print(new_qc.draw())

print("######################### ISTANCE")
istnc = ConstantPropagation.generate_istance(new_qc)
print(istnc)