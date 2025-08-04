from __future__ import annotations

from typing import List, Tuple, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction, ControlledGate, Gate, Qubit, Clbit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import StatePreparation, XGate

import numpy as np

from UnionTable import UnionTable
from util.ActivationState import ActivationState
from QuState import EPS

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
    "if_else"
}

RESET_NAME = "reset"
MEASURE_NAME = "measure"
SWAP_NAME = "swap"

# ConstantPropagation main class
class ConstantPropagation:
    """Run the *constant-propagation* analysis/optimisation on a circuit."""

    MAX_AMPLITUDES: int = 32
    
    @classmethod
    def propagate(cls, circuit: QuantumCircuit, max_amplitudes: int | None = None, max_ent_group_size = 1, table: UnionTable | None = None) -> Tuple[UnionTable, QuantumCircuit]:
        max_amplitudes = max_amplitudes or cls.MAX_AMPLITUDES
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

            # No more information is tracked
            if table.all_top():
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

            if name_lc == MEASURE_NAME:
                c_indices = [c._index for c in cargs]
                if q_indices:
                    table.set_top(q_indices[0]) # Meassurement operations involve only one qubit
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == RESET_NAME:
                ind = q_indices[0]
                if not table.purity_test(ind):
                    if table[ind].is_qubit_state() and table[ind].get_qubit_state().get_n_qubits() <= max_ent_group_size:
                        # Perform rotation from the current state to |0...0>
                        state_vecor =  table[ind].get_qubit_state().to_state_vector()
                        for x in cls._prepare_state(state_vecor, True).data:
                            new_circ.append(x)
                        
                        # Reset ind-th qubit
                        table.reset_state(ind)

                        # Add gates to perform rotation from state |0...0> to the state before reset where the ind-qubit is |0>

                        for q in table[ind].get_qubit_state():
                            table.separate(q)
                    else:
                        if table[ind].is_qubit_state():
                            table.set_top(ind)
                        new_circ.append(instr, qargs, cargs)

                elif table[ind].is_qubit_state():
                    _prob_meas_0 = table[ind].get_qubit_state().probability_measure_zero()
                    _prob_meas_1 = table[ind].get_qubit_state().probability_measure_one()
                    if _prob_meas_1 == 1.0: # Qubit is in the state |1>
                        new_circ.append(XGate(), [ind]) # Apply X(ind) -> |0>
                    elif _prob_meas_0 != 1.0 and _prob_meas_1 != 1.0: 
                        state_vecor =  table[ind].get_qubit_state().to_state_vector()
                        for x in cls._prepare_state(state_vecor, True).data:
                            new_circ.append(x)
                
                
                table.reset_state(ind) # Set to |0> the state of the resetted qubit
                continue

            # Determine control and target qubits
            if isinstance(instr, ControlledGate):
                nctrl = instr.num_ctrl_qubits
                controls: List[int] = q_indices[:nctrl]
                targets: List[int] = q_indices[nctrl:]
            else:
                controls = []
                targets = q_indices

            # Minimise control set
            activation, min_controls = table.minimize_controls(controls)
            if activation == ActivationState.NEVER:
                continue

            if name_lc == SWAP_NAME:
                t1, t2 = targets
                if activation == ActivationState.ALWAYS and not min_controls:
                    table.swap(t1, t2)
                    cls._check_amplitude(table, max_amplitudes, t1)
                    new_circ.append(instr, qargs, cargs)
                    continue


            instr_eff, qargs_eff = cls._rebuild_instruction(instr, qargs, min_controls)

            # Update UnionTable
            if len(targets) == 1:
                cls._apply_single_qubit_gate(table, targets[0], min_controls, instr)
            elif len(targets) == 2:
                cls._apply_two_qubit_gate(table, targets[0], targets[1], min_controls, instr)
            else:
                # Multi‑qubit gates currently unsupported
                for t in targets:
                    table.set_top(t)

            cls._check_amplitude(table, max_amplitudes, targets[0])

            new_circ.append(instr_eff, qargs_eff, cargs)

        return table, new_circ

    @classmethod
    def optimise(cls, circuit: QuantumCircuit, max_amplitudes: int | None = None, max_ent_group_size = 1) -> None:
        """Perform constant-propagation in-place on *circuit."""
        _, new_circ = cls.propagate(circuit, max_amplitudes, max_ent_group_size, emit_optimised_circuit=True)

        circuit.data.clear()
        circuit.append(new_circ.data)


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
        ctrl_qubits = [q for q in qargs[:original_ctrls] if q.index in min_controls]
        target_qubits = qargs[original_ctrls:]
        new_qargs = ctrl_qubits + target_qubits

        return new_gate, new_qargs
    
    
    @staticmethod
    def _prepare_state(state_vector, inverse = False) -> QuantumCircuit:
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