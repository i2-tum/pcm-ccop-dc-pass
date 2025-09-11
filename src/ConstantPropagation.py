from __future__ import annotations

from typing import List, Tuple, Sequence, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ControlledGate, Gate, Qubit, Clbit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import StatePreparation, XGate
from qiskit.circuit.classical import expr
from functools import reduce

import numpy as np

from UnionTable import UnionTable
from util.ActivationState import ActivationState
from util.BitState import BitState
from util.ProbabilisticGate import ProbabilisticGate, BigProbabilisticGate
from SimplifyCondition import SimplifyCondition
import random

__all__ = ["ConstantPropagation"]

def _single_qubit_matrix(instr: Instruction) -> List[complex]:
    """Return a flat list *[a, b, c, d]* representing a 2x2 unitary."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr

    mat = Operator(base).data
    if mat.shape != (2, 2):
        raise ValueError("Instruction is not a single-qubit unitary")
    
    return [complex(mat[0, 0]), complex(mat[0, 1]), complex(mat[1, 0]), complex(mat[1, 1])]


def _two_qubit_matrix(instr: Instruction) -> List[List[complex]]:
    """Return a 4x4 nested list for two-qubit *instr*."""
    mat = Operator(instr).data

    if mat.shape != (4, 4):
        raise ValueError("Instruction is not a two-qubit unitary")
    
    return [[complex(mat[r, c]) for c in range(4)] for r in range(4)]

IGNORED_GATES: set[str] = {
    "barrier",
    "delay",
    "id"
}

UNSUPPORTED_GATES: set[str] = {
    "peres", "peresdg",
    "atru", "afalse", "multi_atru", "multi_afalse",
}

RESET_NAME = "reset"
MEASURE_NAME = "measure"
IF_ELSE_NAME = "if_else"

# ConstantPropagation main class
class ConstantPropagation:
    """Run the *constant-propagation* analysis/optimisation on a circuit."""

    MAX_AMPLITUDES: int = 32
    
    @classmethod
    def _propagate(cls, circuit: QuantumCircuit, max_amplitudes: int | None = None, max_ent_group_size = 1, table: UnionTable | None = None) -> Tuple[UnionTable, QuantumCircuit]:
        max_amplitudes = max_amplitudes or cls.MAX_AMPLITUDES

        clbit_states: dict[Clbit, BitState] = {}
        table = table or UnionTable(circuit.num_qubits)

        # Prepare new circuit
        new_circ = QuantumCircuit(circuit.qubits, circuit.clbits)

        # Walk through instructions
        for inst in circuit.data:
            instr = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits

            q_indices = [q._index for q in qargs]
            name_lc = instr.name.lower()

            cls._check_amplitudes(table, max_amplitudes)

            if table.all_top() and name_lc != RESET_NAME:
                new_circ.append(instr, qargs, cargs)
                continue

            # Ignored gates in the optimization
            if name_lc in IGNORED_GATES:
                new_circ.append(instr, qargs, cargs)
                continue

            # Unsupported or classically-controlled operations
            if name_lc in UNSUPPORTED_GATES:
                for t in q_indices:
                    table.set_top(t)
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == IF_ELSE_NAME:
                cls._optimize_classic_controlled_operation(new_circ, clbit_states, inst, table, max_amplitudes)
                continue
                
            if name_lc == MEASURE_NAME: # Single measurement
                ind = q_indices[0]
                if table.purity_test(ind):
                    _prob_meas_0 = table[ind].get_qubit_state().probability_measure_zero(table.index_in_state(ind))
                    _prob_meas_1 = table[ind].get_qubit_state().probability_measure_one(table.index_in_state(ind))
                    if _prob_meas_0 != 1.0 and _prob_meas_1 != 1.0:
                        state_vector =  table[ind].get_qubit_state().to_state_vector()
                        rot = cls._synthesize_rotation(state_vector, True)
                        inst = rot.to_instruction()
                        new_circ.append(inst, qargs)
                        # Append probabilistic gate
                        prb_gate = ProbabilisticGate(XGate(), _prob_meas_1, cargs[0])
                        new_circ.append(prb_gate, qargs)

                        clbit_states[cargs[0]] = BitState(_prob_meas_1)
                        table.separate(ind)   
                        table.set_top(ind)
                    elif _prob_meas_0 == 1.0:
                        clbit_states[cargs[0]] = BitState.ZERO
                    else:
                        clbit_states[cargs[0]] = BitState.ONE
                elif table[ind].is_qubit_state() and table[ind].get_qubit_state().get_n_qubits() <= max_ent_group_size:
                    state_vector =  table[ind].get_qubit_state().to_state_vector()
                    rot = cls._synthesize_rotation(state_vector, True)
                    inst = rot.to_instruction()

                    targets = table.qubits_in_state(table[ind].get_qubit_state())
                    new_circ.append(inst, targets)
                    
                    state = table[ind].get_qubit_state()
                    probs_big_prob_gate = []
                    gates_ind_big_prob_gate = []
                    for k, v in state:
                        gates_ind_big_prob_gate_curr = []
                        i = 0
                        probs_big_prob_gate.append(abs(v)**2)
                        for kk in k:
                            if kk:
                                gates_ind_big_prob_gate_curr.append((XGate(), [targets[i]]))
                            i += 1
                        gates_ind_big_prob_gate.append(gates_ind_big_prob_gate_curr)

                    big_prob_gate = BigProbabilisticGate(probs_big_prob_gate, gates_ind_big_prob_gate, len(targets), cargs[0])
                    new_circ.append(big_prob_gate, targets)

                    table.set_top(ind)
                    clbit_states[cargs[0]] = BitState.NOT_KNOWN
                    new_circ.append(instr, qargs, cargs)
                else:
                    table.set_top(ind)
                    clbit_states[cargs[0]] = BitState.NOT_KNOWN
                    new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == RESET_NAME:
                ind = q_indices[0]
                if not table.purity_test(ind):
                    if table[ind].is_qubit_state() and table[ind].get_qubit_state().get_n_qubits() <= max_ent_group_size:
                        # Perform rotation from the current state to |0...0>
                        state_vector =  table[ind].get_qubit_state().to_state_vector()
                        rot = cls._synthesize_rotation(state_vector, True)
                        inst = rot.to_instruction()
                        new_circ.append(inst, qargs)
                        
                        # Reset ind-th qubit
                        table.reset_state(ind)

                        # Add gates to perform rotation from state |0...0> to the state before reset where the ind-qubit is |0>
                        state_vector =  table[ind].get_qubit_state().to_state_vector()
                        rot = cls._synthesize_rotation(state_vector, False)
                        inst = rot.to_instruction()
                        new_circ.append(inst, qargs)
                        
                        for q in table.qubits_in_state(table[ind].get_qubit_state()):
                            table.separate(q)
                    else:
                        table.reset_state(ind)
                        for q in table.qubits_in_state(table[ind].get_qubit_state()):
                            table.separate(q)
                        new_circ.append(instr, qargs, cargs)

                elif table[ind].is_qubit_state():
                    _prob_meas_0 = table[ind].get_qubit_state().probability_measure_zero(table.index_in_state(ind))
                    _prob_meas_1 = table[ind].get_qubit_state().probability_measure_one(table.index_in_state(ind))
                    if _prob_meas_1 == 1.0: # Qubit is in the state |1>
                        new_circ.append(XGate(), qargs, cargs) # Apply X(ind) -> |0>
                    elif _prob_meas_0 != 1.0 and _prob_meas_1 != 1.0: 
                        state_vector =  table[ind].get_qubit_state().to_state_vector()
                        rot = cls._synthesize_rotation(state_vector, True)
                        inst = rot.to_instruction()
                        new_circ.append(inst, qargs)
                
                table.reset_state(ind)
                continue
            
            min_contr = cls._minimize_controls(table, instr, qargs)
            if min_contr is not None:
                instr_min_contr, qargs_min_contr = min_contr
                cls._apply_gate(table, instr_min_contr, qargs_min_contr, max_amplitudes)
                new_circ.append(instr_min_contr, qargs_min_contr, cargs)

        return table, new_circ

    @classmethod
    def optimize(cls, circuit: QuantumCircuit, max_amplitudes: int | None = None, max_ent_group_size = 1) -> QuantumCircuit:
        """Perform constant-propagation in-place on *circuit."""
        table, new_circ = cls._propagate(circuit, max_amplitudes, max_ent_group_size)

        return table, new_circ
    
    @classmethod
    def generate_istance(cls, circuit: QuantumCircuit) -> QuantumCircuit:
        new_circ = QuantumCircuit(circuit.qubits, circuit.clbits)
        clbit_states: dict[Clbit, BitState] = {}

        # Walk through instructions
        for inst in circuit.data:
            instr = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits

            name_lc = instr.name.lower()

            if isinstance(instr, ProbabilisticGate):
                creg_from_meas = instr.get_creg_from_meas()
                prob = instr.get_probability()

                # Compiles the probabilistic gate
                if random.random() < prob:
                    new_circ.append(instr.get_base_gate(), qargs, cargs)
                    clbit_states[creg_from_meas] = BitState.ONE
                else:
                    clbit_states[creg_from_meas] = BitState.ZERO
            elif isinstance(instr, BigProbabilisticGate):
                creg_from_meas = instr.get_creg_from_meas()
                creg_from_meas_state = BitState.ZERO # Default value
                probs = instr.get_probabilities()
                gates_and_ind = instr.get_gates()
                r = random.random()
                # Choose one of the element of probs_and_gates according to the probabilities
                weights = [p for p in probs]
                # Use random.choices to pick one sequence based on weights
                selected_seq = random.choices(gates_and_ind, weights=weights, k=1)[0]
                for g, indices in selected_seq:
                    new_circ.append(g, indices)
                    if indices[0] == creg_from_meas._index:
                        creg_from_meas_state = BitState.ONE
                clbit_states[creg_from_meas] = creg_from_meas_state
                    

            elif name_lc == MEASURE_NAME:
                # When a measurement is performed the bit states of the corresponding measurement operation are set to NOT_KNOWN
                for c in cargs:
                    clbit_states[c] = BitState.NOT_KNOWN
            elif name_lc == IF_ELSE_NAME:
                cls._optimize_classic_controlled_operation(new_circ, clbit_states, inst)

            else: # Appends all the other operations
                new_circ.append(instr, qargs, cargs)
        return new_circ
    
    @classmethod
    def _optimize_classic_controlled_operation(cls, new_circ: QuantumCircuit, clbit_states: dict[Clbit, BitState], inst: Instruction, table: UnionTable | None = None, max_amplitudes = 1) -> None:
        instr = inst.operation
        cargs = inst.clbits
        instr_cond = instr.condition
        
        qc_then = []

        for inner_inst in inst.params[0]:
            qc_then_instr = inner_inst.operation
            qc_then_qargs = inner_inst.qubits
            qc_then_cargs = inner_inst.clbits
        
            if table is not None:
                min_contr = cls._minimize_controls(table, qc_then_instr, qc_then_qargs)
                if min_contr is None:
                    continue # The inner operation will never be applied
                qc_then_instr, qc_then_qargs = min_contr
            
            qc_then.append((qc_then_instr, qc_then_qargs, qc_then_cargs))
            
        
        if len(qc_then) == 0:
            return # No operation in the then branch

        if isinstance(instr_cond, tuple): # Case where the condition is a comparison between a register and an integer
            if all(clbit_states.get(c, BitState.ZERO) in (BitState.ZERO, BitState.ONE) for c in cargs):
                # Compare bit states in 'clbit_states' with the value in the expression
                _, val_exp = instr_cond
                val_state = sum((1 if clbit_states.get(c, BitState.ZERO) == BitState.ONE else 0) << c._index for c in cargs)

                if val_exp == val_state: # We know that the gate will always be applied
                    for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                        if table is not None: # Add the instruction without the classical control
                            cls._apply_gate(table, qc_then_instr, qc_then_qargs, max_amplitudes)
                        new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)
                else: 
                    pass # The operation will not be applied at runtime
            else:
                # Check if the already determined bit satisfies the condition
                mask = 0
                expected = 0
                not_determined_bits = []
                for c in cargs:
                    i = c._index
                    st = clbit_states.get(c, BitState.ZERO)
                    if st in (BitState.ZERO, BitState.ONE):
                        mask |= (1 << i)
                        if st == BitState.ONE:
                            expected |= (1 << i)
                    else:
                        not_determined_bits.append(c)
                _, val_exp = instr_cond
                if (val_exp & mask) == expected: # Append the classical controlled operation
                    if len(not_determined_bits) == 1: # Only one bit as control register
                        c = not_determined_bits[0]
                        with new_circ.if_test((c, 0 if (1 << c._index) & val_exp == 0 else 1)):
                            for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                                new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)
                    else: # Multiple bits as control register
                        # Builds the new condition for the classical controlled operation
                        bits = []
                        for c in not_determined_bits:
                            bit = c
                            bit_val_exp = 0 if (1 << bit._index) & val_exp == 0 else 1
                            if bit_val_exp == 0:
                                bit = expr.bit_not(bit)
                            bits.append(bit)
                        cond = reduce(expr.bit_and, bits)
                        with new_circ.if_test(cond):
                            for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                                new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)
                else:
                    pass # The operation will not be applied at runtime
                if table is not None:
                    for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:   
                        q_ind = [q._index for q in qc_then_qargs]
                        for ind in q_ind:
                            table.set_top(ind)
        else: # Case where the condition is an expression
            cond = instr_cond
            res = SimplifyCondition.simplify(cond, clbit_states)
            if res.always_true: # The inner operation will always be applied
                for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                    if table is not None: # Add the instruction without the classical control
                        cls._apply_gate(table, qc_then_instr, qc_then_qargs, max_amplitudes)
                    new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)
            elif res.always_false:
                pass # The inner operation will never be applied
            else: # The inner operation may be applied
                with new_circ.if_test(res.expr):
                    for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                        new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)

                if table is not None:
                    for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:   
                        q_ind = [q._index for q in qc_then_qargs]
                        for ind in q_ind:
                            table.set_top(ind)
    @classmethod
    def _minimize_controls(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit]):
        q_indices = [q._index for q in qargs]
        # name_lc = instr.name.lower()
        # Determine control and target qubits
        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
            controls: List[int] = q_indices[:nctrl]
            # targets: List[int] = q_indices[nctrl:]
        else:
            controls = []
            # targets = q_indices

        # Minimize control set
        activation, min_controls = table.minimize_controls(controls)
        if activation == ActivationState.NEVER:
            return None

        instr_eff, qargs_eff = cls._rebuild_instruction(instr, qargs, min_controls)
        return (instr_eff, qargs_eff)

    @classmethod
    def _apply_gate(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit], max_amplitudes: int) -> None:
        q_indices = [q._index for q in qargs]
        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
            controls: List[int] = q_indices[:nctrl]
            targets: List[int] = q_indices[nctrl:]
        else:
            controls = []
            targets = q_indices
            
        # Update UnionTable
        if len(targets) == 1:
            cls._apply_single_qubit_gate(table, targets[0], controls, instr)
        elif len(targets) == 2:
            cls._apply_two_qubit_gate(table, targets[0], targets[1], controls, instr)
        else:
            # Multi‑qubit gates currently unsupported
            for t in q_indices:
                table.set_top(t)

        cls._check_amplitude(table, max_amplitudes, targets[0])

    # Checks if the number of amplitudes exceeds 'max_amplitudes'
    @staticmethod
    def _check_amplitude(table: UnionTable, max_amplitudes: int, index: int) -> bool:
        reg = table[index]
        if reg.is_qubit_state() and reg.get_qubit_state().size() > max_amplitudes:
            table.set_top(index)
            return True
        return False

    @classmethod
    def _check_amplitudes(cls, table: UnionTable, max_amplitudes: int) -> None:
        for i in range(table.size()):
            cls._check_amplitude(table, max_amplitudes, i)

    @staticmethod
    def _apply_single_qubit_gate(table: UnionTable, target: int, controls: Sequence[int], instr: Instruction) -> None:
        table.combine(target, list(controls))
        if table.is_top(target):
            return
        idx_t = table.index_in_state(target)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _single_qubit_matrix(instr)
        table[target].get_qubit_state().apply_gate(idx_t, matrix, idx_ctrl)

        # Separates states of disentangled qubits after gate application
        table.separate(target)

        for c in controls:
            # Separates states of disentangled qubits after gate application
            table.separate(c)

    @staticmethod
    def _apply_two_qubit_gate(table: UnionTable, t1: int, t2: int, controls: Sequence[int], instr: Instruction) -> None:
        table.combine(t1, list(controls))
        table.combine(t1, t2)
        if table.is_top(t1):
            return
        idx1 = table.index_in_state(t1)
        idx2 = table.index_in_state(t2)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _two_qubit_matrix(instr)
        table[t1].get_qubit_state().apply_two_qubit_gate(idx1, idx2, matrix, idx_ctrl)

        # Separates states of disentangled qubits after gate application
        table.separate(t1)
        table.separate(t2)
        for c in controls:
            table.separate(c)

    # Prunes useless controls from the instruction
    @staticmethod
    def _rebuild_instruction(instr: Instruction, qargs: List[Qubit], min_controls: List[int]) -> Tuple[Instruction, List[Qubit]]:
        """Return *(instruction, qargs)* with the pruned control set."""
        if not isinstance(instr, ControlledGate):
            return instr, qargs

        original_ctrls = instr.num_ctrl_qubits
        if len(min_controls) == original_ctrls:
            return instr, qargs

        base_gate: Gate = instr.base_gate
        new_ctrl_count = len(min_controls)
        new_gate: Gate = base_gate if new_ctrl_count == 0 else base_gate.control(new_ctrl_count)

        # Reflect the order expected by Qiskit: controls first, then targets
        ctrl_qubits = [q for q in qargs[:original_ctrls] if q._index in min_controls]
        target_qubits = qargs[original_ctrls:]
        new_qargs = ctrl_qubits + list(target_qubits)

        return new_gate, new_qargs
    
    
    @staticmethod
    def _synthesize_rotation(state_vector, inverse = False) -> QuantumCircuit:
        # Ensure the input state is normalized
        state_vector = state_vector / np.linalg.norm(state_vector)
        # Get the number of qubits needed (log2 of the length of state_vector)
        n = int(np.log2(len(state_vector)))
        # Create a QuantumCircuit with n qubits
        qc = QuantumCircuit(n)
        state_preparation = StatePreparation(state_vector)
        # Append the state preparation to the quantum circuit
        qc.append(state_preparation, range(n))
        # Decompose the state preparation into individual gates
        #qc = transpile(qc, basis_gates=['h', 'cx', 'rz', 'ry'])

        if inverse:
            return qc.inverse()
        else:
            return qc
