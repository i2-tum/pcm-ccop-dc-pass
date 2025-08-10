from qiskit.circuit import Gate, ClassicalRegister

class ProbabilisticGate(Gate):
    def __init__(self, base_gate : Gate, prob: float, creg_from_meas: ClassicalRegister = None):
        super().__init__(f"prb_{base_gate.name}", base_gate.num_qubits, [prob, *base_gate.params], label = f"prb({prob})_{base_gate.name}")
        self.base_gate = base_gate
        self.prob = prob
        self.creg_from_meas = creg_from_meas
    
    def get_probability(self) -> float:
        return self.prob
    
    def get_creg_from_meas(self) -> ClassicalRegister:
        return self.creg_from_meas
    
    def get_base_gate(self) -> Gate:
        return self.base_gate


# TODO: complete this
class BigBrobabilisticGate():
    def __init__(self, probs):

        ()
