"""
Evaluate the Hermitian operator that represents the derivative of the entropy production of the

raw-key measurement process at a given bipartite state and detector.

The state acts on the ordered bipartite space of Alice and Bob, with the basis label of the pair

(a, b) equal to a * d_local + b, and Alice's raw key is the symbol her detector reports. The

detector is supplied through its detector operators, indexed by that symbol in increasing order.



The derivative is taken with respect to the state in the Hilbert-Schmidt pairing, so pairing the

returned operator with a traceless Hermitian direction reproduces the directional derivative of

the entropy production along that direction, and pairing it with the state itself reproduces the

entropy production. All logarithms are natural, so the returned operator is in nats. The state

must be strictly positive definite and every operator the detector conditions from it must be

too.

Returns
-------
return gradient
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def objective_gradient(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    '''Return the Hermitian derivative of the entropy production at rho, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space. Its smallest eigenvalue must exceed 1e-12.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.

    Returns
    -------
    gradient : np.ndarray
        Complex Hermitian array of shape (d, d) holding the derivative operator.

    Raises
    ------
    ValueError
        If rho is not a Hermitian positive semidefinite square two-dimensional array of unit
        trace with finite entries, if its smallest eigenvalue does not exceed 1e-12, if
        detector_ops is not a three-dimensional array of finite (d, d) blocks matching rho with
        at least one entry, or if any operator the detector conditions from rho has smallest
        eigenvalue at or below 1e-12.
    '''
    return gradient

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_objective_gradient(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    matrix = _validate_density_operator(rho)
    if float(np.linalg.eigvalsh(matrix)[0]) <= 1e-12:
        raise ValueError("rho must have smallest eigenvalue above 1e-12")
    reference = _oracle_moving_reference_logarithm(matrix, detector_ops)
    gradient = _positive_definite_logarithm(matrix) - reference
    return 0.5 * (gradient + gradient.conj().T)

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
    directional_probe = """
def directional_probe(fn, rho, detector_ops, seed):
    gradient = fn(rho, detector_ops)
    d = rho.shape[0]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(3):
        direction = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        direction = 0.5 * (direction + direction.conj().T)
        direction = direction - np.trace(direction).real * np.eye(d) / d
        direction = direction / np.abs(direction).max()
        values.append(float(np.trace(gradient @ direction).real))
    values.append(float(np.trace(gradient @ rho).real))
    return np.array(values)
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
        # --- Normal: a well-conditioned qutrit-qutrit state at the benchmark detector noise ---
        {
            "setup": benchmark,
            "call": "objective_gradient(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_objective_gradient(rho, detector_ops)",
        },
        # --- Normal: directional pairings, including the pairing with the state itself ---
        {
            "setup": benchmark + directional_probe,
            "call": "directional_probe(objective_gradient, rho.copy(), detector_ops.copy(), 23)",
            "gold_call": "directional_probe(_oracle_objective_gradient, rho, detector_ops, 23)",
        },
        # --- Normal: the same state read by a much noisier detector ---
        {
            "setup": benchmark + """
detector_ops = local_detector(3, 3, 0.35)
""",
            "call": "objective_gradient(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_objective_gradient(rho, detector_ops)",
        },
        # --- Boundary: maximally mixed state, where the derivative is a multiple of the identity ---
        {
            "setup": header + detector + """
rho = np.eye(9, dtype=complex) / 9.0
detector_ops = local_detector(3, 3, 0.04)
""",
            "call": "objective_gradient(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_objective_gradient(rho, detector_ops)",
        },
        # --- Boundary: fully randomising detector, where the derivative vanishes ---
        {
            "setup": benchmark + """
detector_ops = local_detector(3, 3, 1.0)
""",
            "call": "objective_gradient(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_objective_gradient(rho, detector_ops)",
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
            "call": "objective_gradient(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_objective_gradient(rho, detector_ops)",
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
            "call": "objective_gradient(rho.copy(), detector_ops.copy())",
            "gold_call": "_oracle_objective_gradient(rho, detector_ops)",
        },
        # --- Invalid: singular state, so the logarithm is not defined ---
        {
            "setup": header + detector + invalid + """
rho = np.diag([0.5, 0.5, 0.0, 0.0]).astype(complex)
detector_ops = local_detector(2, 2, 0.05)
""",
            "call": "probe(objective_gradient, (rho.copy(), detector_ops.copy()))",
            "gold_call": "probe(_oracle_objective_gradient, (rho, detector_ops))",
        },
        # --- Invalid: projective detector, whose conditioned blocks are rank deficient ---
        {
            "setup": benchmark + invalid + """
projective = local_detector(3, 3, 0.0)
""",
            "call": "probe(objective_gradient, (rho.copy(), projective.copy()))",
            "gold_call": "probe(_oracle_objective_gradient, (rho, projective))",
        },
    ]
