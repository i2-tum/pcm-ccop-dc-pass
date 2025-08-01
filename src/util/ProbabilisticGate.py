from qiskit.circuit import Gate

class ProbabilisticGate(Gate):
    def __init__(self, base_gate : Gate, prob):
        super().__init__(f"prb_{base_gate.name}", base_gate.num_qubits, [prob, *base_gate.params], label = f"prb({prob})_{base_gate.name}")
        self.base_gate = base_gate
        self.prob = prob
