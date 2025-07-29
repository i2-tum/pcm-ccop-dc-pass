from __future__ import annotations

from typing import List, Tuple, Optional, Sequence, Union

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ControlledGate, Gate, Qubit, Clbit
from qiskit.quantum_info import Operator

from UnionTable import UnionTable
from util.ActivationState import ActivationState
from QuState import EPS

__all__ = ["ConstantPropagation"]

def _single_qubit_matrix(instr: Instruction) -> List[complex]:
    """Return a flat list *[a, b, c, d]* representing a 2x2 unitary."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr

    mat = Operator(base).data
    if mat.shape != (2, 2):
        raise ValueError("Instruction is not a single‑qubit unitary")
    
    return [complex(mat[0, 0]), complex(mat[0, 1]), complex(mat[1, 0]), complex(mat[1, 1])]


def _two_qubit_matrix(instr: Instruction) -> List[List[complex]]:
    """Return a 4x4 nested list for two-qubit *instr*."""
    mat = Operator(instr).data
    if mat.shape != (4, 4):
        raise ValueError("Instruction is not a two-qubit unitary")
    return [[complex(mat[r, c]) for c in range(4)] for r in range(4)]


# ---------------------------------------------------------------------------
# Gate classification helpers (rough mapping of the C++ enums)
# ---------------------------------------------------------------------------

IGNORED_GATES: set[str] = {
    "barrier",
    "snapshot",  # legacy qasm2 directive
    "delay",  # timing only
    "id",
    "showprobabilities",  # alike to snapshot in qfr
    "teleportation",  # placeholder – not in core Qiskit
    "opcount",  # diagnostic, unsupported in Qiskit
}

UNSUPPORTED_GATES: set[str] = {
    "peres", "peresdg",
    "atru", "afalse", "multi_atru", "multi_afalse",
}

RESET_NAME = "reset"
MEASURE_NAME = "measure"
SWAP_NAME = "swap"

# ---------------------------------------------------------------------------
# ConstantPropagation main class
# ---------------------------------------------------------------------------

class ConstantPropagation:
    """Run the *constant‑propagation* analysis/optimisation on a circuit."""

    MAX_AMPLITUDES: int = 32  # sensible default, tunable

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Public API
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def propagate(
        cls,
        circuit: QuantumCircuit,
        max_amplitudes: int | None = None,
        table: UnionTable | None = None,
        *,
        emit_optimised_circuit: bool = True,
    ) -> Tuple[UnionTable, Optional[QuantumCircuit]]:
        """Analyse *circuit* and (optionally) emit a simplified version.

        Parameters
        ----------
        circuit
            Circuit to analyse.  **Is *not* modified in‑place.**
        max_amplitudes
            Soft cap for the number of amplitudes we keep explicitly per
            qubit; exceeding it collapses that qubit to *TOP*.
        table
            Existing :class:`UnionTable` to update.  A new table is
            created if *None*.
        emit_optimised_circuit
            ``True`` → also build a copy of *circuit* with redundant
            controls stripped; ``False`` → perform analysis only.
        """
        max_amplitudes = max_amplitudes or cls.MAX_AMPLITUDES
        table = table or UnionTable(circuit.num_qubits)

        # --- 1) Flatten composite gates (two decomposition rounds cover
        #        most nested definitions) --------------------------------
        flat = circuit.decompose(reps=2)

        # Prepare new circuit (if requested)
        new_circ: Optional[QuantumCircuit]
        if emit_optimised_circuit:
            new_circ = QuantumCircuit(flat.qubits, flat.clbits)
        else:
            new_circ = None

        # --- 2) Walk through instructions ----------------------------------
        for instr, qargs, cargs in flat.data:
            q_indices = [q.index for q in qargs]
            name_lc = instr.name.lower()

            # Pre‑flight: global amplitude guard / bail‑out fast‑path -----
            cls._check_amplitudes(table, max_amplitudes)
            if table.all_top():
                if new_circ is not None:
                    new_circ.append(instr, qargs, cargs)
                continue

            # 1) Copy‑through gates (barrier, delay, …) -------------------
            if name_lc in IGNORED_GATES:
                if new_circ is not None:
                    new_circ.append(instr, qargs, cargs)
                continue

            # 2) Unsupported or classically‑controlled → mark targets TOP--
            if name_lc in UNSUPPORTED_GATES or instr.condition is not None:
                for t in q_indices:
                    table.set_top(t)
                if new_circ is not None:
                    new_circ.append(instr, qargs, cargs)
                continue

            # 3) Measurement ---------------------------------------------
            if name_lc == MEASURE_NAME:
                # Only the *first* qubit is measured; that one collapses
                if q_indices:
                    table.set_top(q_indices[0])
                if new_circ is not None:
                    new_circ.append(instr, qargs, cargs)
                continue

            # 4) Reset ----------------------------------------------------
            if name_lc == RESET_NAME:
                if q_indices:
                    table.reset_state(q_indices[0])
                if new_circ is not None:
                    new_circ.append(instr, qargs, cargs)
                continue

            # 5) Determine controls / targets ----------------------------
            if isinstance(instr, ControlledGate):
                nctrl = instr.num_ctrl_qubits
                controls: List[int] = q_indices[:nctrl]
                targets: List[int] = q_indices[nctrl:]
            else:
                controls = []
                targets = q_indices

            # Minimise control set ---------------------------------------
            activation, min_controls = table.minimize_controls(controls)
            if activation == ActivationState.NEVER:
                # Gate will never fire – no effect – skip!
                continue

            # 6) Special‑case: SWAP --------------------------------------
            if name_lc == SWAP_NAME and len(targets) == 2:
                t1, t2 = targets
                if activation == ActivationState.ALWAYS and not min_controls:
                    # Full swap without remaining controls – purely
                    # structural, can use UnionTable.swap.
                    table.swap(t1, t2)
                    cls._check_amplitude(table, max_amplitudes, t1)
                    if new_circ is not None:
                        new_circ.append(instr, qargs, cargs)  # identical op
                    continue
                # Otherwise treat as generic controlled two‑qubit gate
                # (Fredkin / controlled‑SWAP).

            # 7) Rebuild instruction with pruned controls ----------------
            if new_circ is not None:
                instr_eff, qargs_eff = cls._rebuild_instruction(instr, qargs, min_controls)

            # 8) Update UnionTable ---------------------------------------
            if len(targets) == 1:
                cls._apply_single_qubit_gate(table, targets[0], min_controls, instr)
            elif len(targets) == 2:
                cls._apply_two_qubit_gate(table, targets[0], targets[1], min_controls, instr)
            else:
                # Multi‑qubit unitaries currently unsupported – collapse
                for t in targets:
                    table.set_top(t)

            # Immediate amplitude guard for main target ------------------
            cls._check_amplitude(table, max_amplitudes, targets[0])

            # 9) Append to new circuit -----------------------------------
            if new_circ is not None:
                new_circ.append(instr_eff, qargs_eff, cargs)

        return table, new_circ

    # ------------------------------------------------------------------
    # Convenience wrapper: in‑place optimisation (mutates *circuit*)
    # ------------------------------------------------------------------

    @classmethod
    def optimise(cls, circuit: QuantumCircuit, max_amplitudes: int | None = None) -> None:
        """Perform constant‑propagation **in‑place** on *circuit*."""
        _, new_circ = cls.propagate(circuit, max_amplitudes, emit_optimised_circuit=True)
        # Qiskit lacks a true "replace all data" helper; rebuild manually
        circuit.data.clear()
        circuit.append(new_circ.data)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    # ---------- Amplitude guards --------------------------------------

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

    # ---------- Gate applications -------------------------------------

    @staticmethod
    def _apply_single_qubit_gate(
        table: UnionTable,
        target: int,
        controls: Sequence[int],
        instr: Instruction,
    ) -> None:
        table.combine(target, list(controls))
        if table.is_top(target):
            return
        idx_t = table.index_in_state(target)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _single_qubit_matrix(instr)
        table[target].get_qubit_state().apply_gate(idx_t, matrix, idx_ctrl)

    @staticmethod
    def _apply_two_qubit_gate(
        table: UnionTable,
        t1: int,
        t2: int,
        controls: Sequence[int],
        instr: Instruction,
    ) -> None:
        table.combine(t1, list(controls))
        table.combine(t1, t2)
        if table.is_top(t1):
            return
        idx1 = table.index_in_state(t1)
        idx2 = table.index_in_state(t2)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _two_qubit_matrix(instr)
        table[t1].get_qubit_state().apply_two_qubit_gate(idx1, idx2, matrix, idx_ctrl)

    # ---------- Instruction rebuilding (control pruning) --------------

    @staticmethod
    def _rebuild_instruction(
        instr: Instruction,
        qargs: List[Qubit],
        min_controls: List[int],
    ) -> Tuple[Instruction, List[Qubit]]:
        """Return *(instruction, qargs)* with the pruned control set."""
        if not isinstance(instr, ControlledGate):
            # Nothing to strip
            return instr, qargs

        original_ctrls = instr.num_ctrl_qubits
        if len(min_controls) == original_ctrls:
            # No change
            return instr, qargs

        base_gate: Gate = instr.base_gate
        new_ctrl_count = len(min_controls)
        new_gate: Gate = base_gate if new_ctrl_count == 0 else base_gate.control(new_ctrl_count)

        # Reflect the order expected by Qiskit: controls first, then targets
        ctrl_qubits = [q for q in qargs[:original_ctrls] if q.index in min_controls]
        target_qubits = qargs[original_ctrls:]
        new_qargs = ctrl_qubits + target_qubits

        return new_gate, new_qargs
