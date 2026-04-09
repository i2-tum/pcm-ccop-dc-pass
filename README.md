# Optimization pass for simplifying Classically Controlled Operations in Dynamic Circuits 

This method is based on the manuscript ([preprint link coming soon]).

This framework takes a Qiskit dynamic circuit containing mid-circuit measurements, resets, and classical `if_test` controls, and rewrites it into a **probabilistic circuit**.

In short, it:
- simplifies deterministic classical controls whenever possible
- replaces non-deterministic mid-circuit measurement/reset behavior with probabilistic constructs (`ProbabilisticGate`)

Each call to `generate_instance(...)` samples those probabilistic constructs and returns a concrete circuit instance.

## Quick Start

```python
from src.ConstantPropagation import ConstantPropagation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

qr = QuantumRegister(2, "q")
cr = ClassicalRegister(1, "c")
qc = QuantumCircuit(qr, cr)

qc.h(0)
qc.measure(0, 0)
with qc.if_test((cr[0], 1)):
    qc.x(1)
qc.reset(0)

# 1) Transform dynamic circuit -> probabilistic circuit
prob_circ = ConstantPropagation.optimize(qc)

# 2) Instantiate (sample) the probabilistic circuit
inst_1 = ConstantPropagation.generate_instance(prob_circ)
inst_2 = ConstantPropagation.generate_instance(prob_circ)
```

`inst_1` and `inst_2` may differ, because sampling is independent at each call.


