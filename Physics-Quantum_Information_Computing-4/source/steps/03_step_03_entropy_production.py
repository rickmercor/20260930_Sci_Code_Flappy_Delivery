"""
Evaluate the entropy produced by the raw-key measurement process on a given bipartite state,

which is the objective whose constrained minimum bounds Eve's uncertainty about the raw key.

The state acts on the ordered bipartite space of Alice and Bob, with the basis label of the

pair (a, b) equal to a * d_local + b. Alice's raw key is the symbol her detector reports, and

the detector is supplied through its detector operators, indexed by that symbol in increasing

order.



All entropies are von Neumann entropies taken with the natural logarithm, so the returned

value is in nats. Eigenvalues that are not strictly positive contribute nothing, following the

convention that zero times the logarithm of zero is zero, which makes the value finite for

states of any rank and for a detector whose operators are not invertible.

Returns
-------
return production
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def entropy_production(rho: "np.ndarray", detector_ops: "np.ndarray") -> float:
    '''Return the entropy production of the raw-key measurement process, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive semidefinite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space. Hermiticity is required within 1e-9 and the smallest
        eigenvalue within -1e-9.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol.

    Returns
    -------
    production : float
        Entropy produced by the process, in nats.

    Raises
    ------
    ValueError
        If rho is not a two-dimensional square array with at least one row, if its entries are
        not all finite, if it is not Hermitian within 1e-9, if its smallest eigenvalue is below
        -1e-9, if its trace differs from one by more than 1e-9, or if detector_ops is not a
        three-dimensional array of finite (d, d) blocks matching rho with at least one entry.
    '''
    return production

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_density_operator(rho: "np.ndarray") -> "np.ndarray":
    """Return rho as a complex array after checking it is a valid density operator."""
    matrix = np.asarray(rho, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("rho must be a square 2D array with at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rho must have finite entries")
    if not np.allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-9):
        raise ValueError("rho must be Hermitian within 1e-9")
    matrix = 0.5 * (matrix + matrix.conj().T)
    if float(np.linalg.eigvalsh(matrix)[0]) < -1e-9:
        raise ValueError("rho must be positive semidefinite within 1e-9")
    if abs(float(np.trace(matrix).real) - 1.0) > 1e-9:
        raise ValueError("rho must have unit trace within 1e-9")
    return matrix


def _von_neumann_entropy(matrix: "np.ndarray") -> float:
    """Return the von Neumann entropy in nats, discarding non-positive eigenvalues."""
    spectrum = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    spectrum = spectrum[spectrum > 0.0]
    return float(-np.sum(spectrum * np.log(spectrum)))


def _oracle_entropy_production(rho: "np.ndarray", detector_ops: "np.ndarray") -> float:
    matrix = _validate_density_operator(rho)
    blocks = _oracle_key_map_blocks(matrix, detector_ops)
    recorded = sum(_von_neumann_entropy(block) for block in blocks)
    return float(recorded - _von_neumann_entropy(matrix))

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

    benchmark = header + detector + """
rng = np.random.default_rng(11)
root = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
rho = root @ root.conj().T
rho = rho / np.trace(rho).real
detector_ops = local_detector(3, 3, 0.04)
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
        # --- Normal: a generic full-rank qutrit-qutrit state at the benchmark detector noise ---
        {
            "setup": benchmark,
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Normal: near-benchmark state built from the maximally entangled pair and noise ---
        {
            "setup": header + detector + """
vec = np.zeros(9, dtype=complex)
for k in range(3):
    vec[3 * k + k] = 1.0 / np.sqrt(3.0)
rho = 0.88 * np.outer(vec, vec.conj()) + 0.12 * np.eye(9, dtype=complex) / 9.0
detector_ops = local_detector(3, 3, 0.04)
""",
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Normal: the same state read by a much noisier detector ---
        {
            "setup": benchmark + """
detector_ops = local_detector(3, 3, 0.35)
""",
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Boundary: fully randomising detector, where the record carries no key ---
        {
            "setup": benchmark + """
detector_ops = local_detector(3, 3, 1.0)
""",
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Boundary: projective detector limit on the maximally mixed state ---
        {
            "setup": header + detector + """
rho = np.eye(9, dtype=complex) / 9.0
detector_ops = local_detector(3, 3, 0.0)
""",
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Edge: rank-one maximally entangled pair read projectively ---
        {
            "setup": header + detector + """
vec = np.zeros(9, dtype=complex)
for k in range(3):
    vec[3 * k + k] = 1.0 / np.sqrt(3.0)
rho = np.outer(vec, vec.conj())
detector_ops = local_detector(3, 3, 0.0)
""",
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Edge: singular qubit-qubit state with a coherence across Alice's labels ---
        {
            "setup": header + detector + """
vec = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
rho = np.outer(vec, vec.conj())
detector_ops = local_detector(2, 2, 0.09)
""",
            "call": "entropy_production(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_entropy_production(rho, detector_ops)",
        },
        # --- Invalid: non-Hermitian input ---
        {
            "setup": header + detector + invalid + """
rho = np.eye(4, dtype=complex) / 4.0
rho[0, 1] = 0.05
detector_ops = local_detector(2, 2, 0.05)
""",
            "call": "probe(entropy_production, (rho.copy(), detector_ops.copy()))",
            "gold_call": "probe(_oracle_entropy_production, (rho, detector_ops))",
        },
        # --- Invalid: trace different from one ---
        {
            "setup": header + detector + invalid + """
rho = np.eye(4, dtype=complex) / 2.0
detector_ops = local_detector(2, 2, 0.05)
""",
            "call": "probe(entropy_production, (rho.copy(), detector_ops.copy()))",
            "gold_call": "probe(_oracle_entropy_production, (rho, detector_ops))",
        },
    ]
