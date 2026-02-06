from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from ConstantPropagation import ConstantPropagation
from qiskit.circuit.classical import expr

qr = QuantumRegister(6)
cr = ClassicalRegister(6)
qc = QuantumCircuit(qr, cr)

# Case where I noticed that I have to fix the generation of the instance
# With BigProbabilisticGate
qc.h(0)
qc.cx(0, 1)

qc.h(2)
qc.cx(2, 3)

qc.measure(0, 0)
qc.measure(1, 1)
qc.measure(2, 2)
qc.measure(3, 3)

with qc.if_test(expr.bit_and(cr[0], expr.bit_and(cr[1], expr.bit_and(cr[2], cr[3])))):
    qc.x(3)
    qc.h(4)

qc.reset(1)


## Case where I am checking whether the disentangle algorithm is working well
# qc.h(0)
# qc.cx(0, 1)
# qc.h(1)
# qc.cx(1,2)
# qc.h(2)

# qc.measure(qr[0], cr[0])


# qc.measure(qr[0], cr[0])
# qc.measure(qr[1], cr[1])

# with qc.if_test((cr, 3)):
#     qc.x(5)
#     qc.h(2)

# TODO: do a test on conditions with XOR to test the part:
# if l.expr == r.expr:
#   return CondResult(always_false=True)


# qc.reset(qr[1])
# qc.measure(qr[1], cr[1])

# with qc.if_test((cr[1], 0)) as else_:
#     qc.h(3)
# with else_:
#     qc.x(4)

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
table, new_qc, info = ConstantPropagation.optimize(qc, max_ent_group_size=10)
print(table)
print("######################### PROB CIRC")
print(new_qc.draw())

print("######################### ISTANCE")
instnc = ConstantPropagation.generate_instance(new_qc)
print(instnc)

print("######################### INFO")
print(info)