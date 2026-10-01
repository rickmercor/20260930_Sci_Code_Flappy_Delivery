"""
Map a bipartite operator through the detector operators of the protocol, returning the stack of

operators that describes the joint system once Alice's raw-key symbol has been recorded.

Alice's raw key is the symbol her detector reports, and the detector operators are indexed by

that symbol in increasing order. The joint space is ordered so that the basis label of the pair

(a, b) is a * d_local + b, with a on Alice's side, and every detector operator acts on that

joint space.



The returned stack is indexed by the key symbol in the same order as the supplied detector

operators, and entry a is the operator obtained from the input by the detector operator of

symbol a. Each entry has the shape of the input. The map is linear in the input and is applied

entry by entry, so it is defined for any square input of compatible size, Hermitian or not.

Summed over the key symbol the traces of the entries reproduce the trace of the input whenever

the detector operators resolve the identity, and each entry inherits Hermiticity and positive

semidefiniteness from the input.

Returns
-------
return blocks
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def key_map_blocks(operator: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    '''Return the stack of key-conditioned operators produced by the detector operators.

    Parameters
    ----------
    operator : np.ndarray
        Square array of shape (d, d) acting on the ordered bipartite space. Real or complex
        input is accepted.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol.

    Returns
    -------
    blocks : np.ndarray
        Complex array of shape (m, d, d) whose entry a is the operator conditioned on key
        symbol a.

    Raises
    ------
    ValueError
        If operator is not a two-dimensional square array with at least one row, if its
        entries are not all finite, if detector_ops is not a three-dimensional array with at
        least one entry whose blocks are square of the same dimension as operator, or if the
        detector entries are not all finite.
    '''
    return blocks

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_detector_ops(operator: "np.ndarray", detector_ops: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return the operator and detector stack as validated complex arrays."""
    matrix = np.asarray(operator, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("operator must be a square 2D array with at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator must have finite entries")
    stack = np.asarray(detector_ops, dtype=complex)
    dim = matrix.shape[0]
    if stack.ndim != 3 or stack.shape[0] < 1 or stack.shape[1:] != (dim, dim):
        raise ValueError("detector_ops must have shape (m, d, d) with m >= 1 matching operator")
    if not np.all(np.isfinite(stack)):
        raise ValueError("detector_ops must have finite entries")
    return matrix, stack


def _oracle_key_map_blocks(operator: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    matrix, stack = _validate_detector_ops(operator, detector_ops)
    return np.array([element @ matrix @ element.conj().T for element in stack], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    header = "import numpy as np\n"

    detector = """
def local_detector(d_local, d_bob, epsilon):
    out = []
    for a in range(d_local):
        element = (epsilon / d_local) * np.eye(d_local, dtype=complex)
        element[a, a] += 1.0 - epsilon
        w, V = np.linalg.eigh(element)
        root = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T
        out.append(np.kron(root, np.eye(d_bob, dtype=complex)))
    return np.array(out, dtype=complex)
"""

    state_setup = header + detector + """
rng = np.random.default_rng(11)
root = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
rho = root @ root.conj().T
rho = rho / np.trace(rho).real
detector_ops = local_detector(3, 3, 0.04)
"""

    invariance_probe = """
def invariance_probe(fn, matrix, detector_ops):
    blocks = fn(matrix, detector_ops)
    traces = np.array([float(np.trace(b).real) for b in blocks])
    hermitian = max(float(np.abs(b - b.conj().T).max()) for b in blocks)
    smallest = min(float(np.linalg.eigvalsh(0.5 * (b + b.conj().T))[0]) for b in blocks)
    return np.array([float(traces.sum()), hermitian, smallest,
                     float(traces.min()), float(traces.max())])
"""

    invalid = """
def probe(fn, args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
"""

    return [
        # --- Normal: a full-rank qutrit-qutrit state at the benchmark detector noise ---
        {
            "setup": state_setup,
            "call": "key_map_blocks(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_key_map_blocks(rho, detector_ops)",
        },
        # --- Normal: trace resolution, Hermiticity and positivity of the conditioned blocks ---
        {
            "setup": state_setup + invariance_probe,
            "call": "invariance_probe(key_map_blocks, rho.copy(), detector_ops.copy())",
            "gold_call": "invariance_probe(_oracle_key_map_blocks, rho, detector_ops)",
        },
        # --- Normal: non-Hermitian input, so the map is exercised as a linear operation ---
        {
            "setup": header + detector + """
matrix = (np.arange(36, dtype=float) + 1.0).reshape(6, 6) + 1j * np.eye(6)
detector_ops = local_detector(2, 3, 0.12)
""",
            "call": "key_map_blocks(matrix.copy(), detector_ops.copy())",
            "gold_call": "_oracle_key_map_blocks(matrix, detector_ops)",
        },
        # --- Boundary: projective detector limit, where the blocks become rank deficient ---
        {
            "setup": state_setup + """
detector_ops = local_detector(3, 3, 0.0)
""",
            "call": "key_map_blocks(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_key_map_blocks(rho, detector_ops)",
        },
        # --- Boundary: a single detector operator equal to the identity ---
        {
            "setup": header + """
rng = np.random.default_rng(5)
matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
detector_ops = np.eye(4, dtype=complex).reshape(1, 4, 4)
""",
            "call": "key_map_blocks(matrix.copy(), detector_ops.copy())",
            "gold_call": "_oracle_key_map_blocks(matrix, detector_ops)",
        },
        # --- Edge: maximally entangled qutrit pair, whose blocks keep the full coherence ---
        {
            "setup": header + detector + """
vec = np.zeros(9, dtype=complex)
for k in range(3):
    vec[3 * k + k] = 1.0 / np.sqrt(3.0)
rho = np.outer(vec, vec.conj())
detector_ops = local_detector(3, 3, 0.04)
""",
            "call": "key_map_blocks(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_key_map_blocks(rho, detector_ops)",
        },
        # --- Edge: fully randomising detector, where every block is a scaled copy of the input ---
        {
            "setup": state_setup + """
detector_ops = local_detector(3, 3, 1.0)
""",
            "call": "key_map_blocks(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_key_map_blocks(rho, detector_ops)",
        },
        # --- Invalid: detector blocks of the wrong dimension ---
        {
            "setup": header + invalid + """
matrix = np.eye(6, dtype=complex)
detector_ops = np.zeros((2, 4, 4), dtype=complex)
""",
            "call": "probe(key_map_blocks, (matrix.copy(), detector_ops.copy()))",
            "gold_call": "probe(_oracle_key_map_blocks, (matrix, detector_ops))",
        },
        # --- Invalid: non-square input ---
        {
            "setup": header + invalid + """
matrix = np.zeros((2, 6), dtype=complex)
detector_ops = np.zeros((2, 2, 2), dtype=complex)
""",
            "call": "probe(key_map_blocks, (matrix.copy(), detector_ops.copy()))",
            "gold_call": "probe(_oracle_key_map_blocks, (matrix, detector_ops))",
        },
    ]
