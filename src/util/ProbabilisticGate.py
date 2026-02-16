from qiskit.circuit import Gate, Clbit
    
# The probabilistic gate takes as input
# - the number of qubits it acts on
# - the index i of the qubit that has been measured to get the probability
# - the probability value p indicating the probability that the measured qubit is in state |1>
# - a sequence of instruction (gate, indices) to be applied with probability p
# - a sequence of instruction (gate, indices) to be applied with probability 1-p
class ProbabilisticGate(Gate):
    def __init__(self, prob: float, inst_if_one: list[tuple[Gate, list[int]]], inst_if_zero: list[tuple[Gate, list[int]]], num_qubits: int, creg_from_meas: Clbit | None = None):
        super().__init__("prob_gate", num_qubits, [prob], label="prob_gate")
        self.prob = prob
        self.inst_if_one = inst_if_one
        self.inst_if_zero = inst_if_zero
        self.creg_from_meas = creg_from_meas
    
    def get_probability(self) -> float:
        return self.prob

    def get_inst_if_one(self) -> list[tuple[Gate, list[int]]]:
        return self.inst_if_one

    def get_inst_if_zero(self) -> list[tuple[Gate, list[int]]]:
        return self.inst_if_zero

    def get_creg_from_meas(self) -> Clbit | None:
        return self.creg_from_meas
