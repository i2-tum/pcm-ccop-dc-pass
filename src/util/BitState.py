"""
This class is used to represent the state of classical bits in the propagation of the classical registers
"""

class BitState:
    ZERO = 0
    ONE = 1
    NOT_KNOWN = 2

    def __init__(self, prob: float | None = None):
        if prob is None:
            raise ValueError("Use BitState.ZERO, ONE, NOT_KNOWN for fixed states.")
        if not (0.0 <= prob <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        self._prob = prob

    def __repr__(self):
        if self is BitState.ZERO:
            return "ZERO"
        elif self is BitState.ONE:
            return "ONE"
        elif self is BitState.NOT_KNOWN:
            return "NOT_KNOWN"
        else:
            return f"ONE_PROB({self._prob})"

    def is_probabilistic(self):
        return self._prob is not None

    def probability(self):
        if not self.is_probabilistic():
            raise TypeError("Not a probabilistic state.")
        return self._prob