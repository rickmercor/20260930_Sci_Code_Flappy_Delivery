"""
Evaluate the Hermitian operator that the protocol's data-matching step uses as its reference

logarithm at a given bipartite state and detector.

The state acts on the ordered bipartite space of Alice and Bob, with the basis label of the pair

(a, b) equal to a * d_local + b, and Alice's raw key is the symbol her detector reports. The

detector is supplied through its detector operators, indexed by that symbol in increasing order.

The returned operator is Hermitian, has the shape of the state, and is expressed with natural

logarithms, so it is in nats. It carries no free additive constant and no normalising constant is subtracted: pairing it with the state, as the trace of their product, gives the sum of tr(B log B) over the key-conditioned operators B of Step 02 at that state. Every operator the detector conditions from the state must be strictly

positive definite, which for a detector whose operators are invertible holds whenever the state

is, so the state and the detector are both restricted accordingly.

Returns
-------
return reference
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moving_reference_logarithm(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    '''Return the reference logarithm of the data-matching step at rho, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.

    Returns
    -------
    reference : np.ndarray
        Complex Hermitian array of shape (d, d) holding the reference logarithm.

    Raises
    ------
    ValueError
        If rho is not a Hermitian positive semidefinite square two-dimensional array of unit
        trace with finite entries, if detector_ops is not a three-dimensional array of finite
        (d, d) blocks matching rho with at least one entry, or if any operator the detector
        conditions from rho has smallest eigenvalue at or below 1e-12.
    '''
    return reference

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _positive_definite_logarithm(matrix: "np.ndarray") -> "np.ndarray":
    """Return the matrix logarithm of a Hermitian operator with eigenvalues above 1e-12."""
    spectrum, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    if float(spectrum[0]) <= 1e-12:
        raise ValueError("operator must have smallest eigenvalue above 1e-12")
    return (vectors * np.log(spectrum)) @ vectors.conj().T


def _oracle_moving_reference_logarithm(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    matrix = _validate_density_operator(rho)
    blocks = _oracle_key_map_blocks(matrix, detector_ops)
    stack = np.asarray(detector_ops, dtype=complex)
    reference = np.zeros_like(matrix)
    for element, block in zip(stack, blocks):
        reference = reference + element.conj().T @ _positive_definite_logarithm(block) @ element
    return 0.5 * (reference + reference.conj().T)

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
rho = 0.7 * rho + 0.3 * np.eye(9, dtype=complex) / 9.0
detector_ops = local_detector(3, 3, 0.04)
"""
    pairing_probe = """
def pairing_probe(fn, production_fn, rho, detector_ops):
    reference = fn(rho, detector_ops)
    spectrum, vectors = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    log_rho = (vectors * np.log(spectrum)) @ vectors.conj().T
    gradient = log_rho - reference
    paired = float(np.trace(rho @ gradient).real)
    return np.array([paired, paired - production_fn(rho, detector_ops),
                     float(np.abs(reference - reference.conj().T).max())])
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
        # --- Normal: well-conditioned qutrit-qutrit state at the benchmark detector noise ---
        {
            "setup": benchmark,
            "call": "moving_reference_logarithm(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_moving_reference_logarithm(rho, detector_ops)",
        },
        # --- Normal: the reference pairs with the state to reproduce the entropy production ---
        {
            "setup": benchmark + pairing_probe,
            "call": "pairing_probe(moving_reference_logarithm, entropy_production, rho.copy(), detector_ops.copy())",
            "gold_call": "pairing_probe(_oracle_moving_reference_logarithm, _oracle_entropy_production, rho, detector_ops)",
            "tol": 1e-8,
        },
        # --- Normal: maximally mixed state, where the reference is a multiple of the identity ---
        {
            "setup": header + detector + """
rho = np.eye(9, dtype=complex) / 9.0
detector_ops = local_detector(3, 3, 0.04)
""",
            "call": "moving_reference_logarithm(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_moving_reference_logarithm(rho, detector_ops)",
        },
        # --- Boundary: fully randomising detector, where every block is a scaled copy ---
        {
            "setup": benchmark + """
detector_ops = local_detector(3, 3, 1.0)
""",
            "call": "moving_reference_logarithm(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_moving_reference_logarithm(rho, detector_ops)",
        },
        # --- Boundary: a single detector operator equal to the identity ---
        {
            "setup": benchmark + """
detector_ops = np.eye(9, dtype=complex).reshape(1, 9, 9)
""",
            "call": "moving_reference_logarithm(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_moving_reference_logarithm(rho, detector_ops)",
        },
        # --- Edge: strongly correlated qutrit pair close to the boundary of positivity ---
        {
            "setup": header + detector + """
vec = np.zeros(9, dtype=complex)
for k in range(3):
    vec[3 * k + k] = 1.0 / np.sqrt(3.0)
rho = 0.995 * np.outer(vec, vec.conj()) + 0.005 * np.eye(9, dtype=complex) / 9.0
detector_ops = local_detector(3, 3, 0.04)
""",
            "call": "moving_reference_logarithm(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_moving_reference_logarithm(rho, detector_ops)",
        },
        # --- Edge: qubit-qubit state with an asymmetric Alice marginal and a quiet detector ---
        {
            "setup": header + detector + """
rho = np.array([[0.40, 0.05, 0.02, 0.12],
                [0.05, 0.14, 0.01, 0.03],
                [0.02, 0.01, 0.18, 0.04],
                [0.12, 0.03, 0.04, 0.28]], dtype=complex)
rho = rho / np.trace(rho).real
detector_ops = local_detector(2, 2, 0.01)
""",
            "call": "moving_reference_logarithm(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_moving_reference_logarithm(rho, detector_ops)",
        },
        # --- Invalid: projective detector, whose conditioned blocks are rank deficient ---
        {
            "setup": benchmark + invalid + """
projective = local_detector(3, 3, 0.0)
""",
            "call": "probe(moving_reference_logarithm, (rho.copy(), projective.copy()))",
            "gold_call": "probe(_oracle_moving_reference_logarithm, (rho, projective))",
        },
        # --- Invalid: singular state ---
        {
            "setup": header + detector + invalid + """
rho = np.diag([0.5, 0.5, 0.0, 0.0]).astype(complex)
detector_ops = local_detector(2, 2, 0.05)
""",
            "call": "probe(moving_reference_logarithm, (rho.copy(), detector_ops.copy()))",
            "gold_call": "probe(_oracle_moving_reference_logarithm, (rho, detector_ops))",
        },
    ]
