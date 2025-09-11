from qiskit.circuit import Gate, Clbit

class ProbabilisticGate(Gate):
    def __init__(self, base_gate : Gate, prob: float, creg_from_meas: Clbit | None = None):
        super().__init__(f"prb_{base_gate.name}", base_gate.num_qubits, [prob, *base_gate.params], label = f"prb({prob})_{base_gate.name}")
        self.base_gate = base_gate
        self.prob = prob
        self.creg_from_meas = creg_from_meas
    
    def get_probability(self) -> float:
        return self.prob
    
    def get_creg_from_meas(self) -> Clbit | None:
        return self.creg_from_meas
    
    def get_base_gate(self) -> Gate:
        return self.base_gate

class BigProbabilisticGate(Gate):
    def __init__(self, probs, gates_and_ind, num_qubits, creg_from_meas: Clbit | None = None):
        if len(probs) != len(gates_and_ind):
            raise ValueError("Length of probs and gates must be the same.")
        super().__init__("big_prob", num_qubits, [], label="big_prob")

        # store as list of (prob, (gate, indices))
        self.probs_and_gates = [(probs[i], gates_and_ind[i]) for i in range(len(probs))]
        self.creg_from_meas = creg_from_meas
    
    def get_probabilities(self) -> list[float]:
        return [p for p, _ in self.probs_and_gates]
    
    def get_gates(self) -> list[Gate]:
        return [g for _, g in self.probs_and_gates]
    
    def get_probs_and_gates(self) -> list[tuple[float, Gate]]:
        return list(self.probs_and_gates)
    
    def get_creg_from_meas(self) -> Clbit | None:
        return self.creg_from_meas
