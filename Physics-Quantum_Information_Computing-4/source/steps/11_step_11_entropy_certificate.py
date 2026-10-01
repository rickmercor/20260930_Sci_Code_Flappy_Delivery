"""
Evaluate the rigorous lower bound on the constrained minimum of the raw-key entropy production

that a given linearisation state together with a given real weight vector certifies.

The weights shift the derivative operator G of Step 05 with a plus sign, as G + sum_i w_i M_i,

with M_i the constraint observables and w_i the weights.

The linearisation state acts on the ordered bipartite space of Alice and Bob, with the basis

label of the pair (a, b) equal to a * d_local + b, and Alice's raw key is the symbol her detector

reports. The detector is supplied through its detector operators, indexed by that symbol in

increasing order. The constraint observables, the observed moment vector and the weight vector

share one ordering.



The returned value is in nats. It is a valid lower bound for every admissible linearisation state

and every real weight vector, so it may be negative and it may be far below the constrained

minimum when either argument is poorly chosen; its quality is what improves as the pipeline

advances. The linearisation state need not reproduce the observed moments.

Returns
-------
return bound
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def entropy_certificate(rho: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", detector_ops: "np.ndarray", weights: "np.ndarray") -> float:
    '''Return the certified lower bound on the constrained minimum, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space, with smallest eigenvalue above 1e-12.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables matching the dimension of rho.
    moments : np.ndarray
        Real array of shape (n, ) with finite entries, holding the observed value of each
        observable in the order of constraint_ops.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in the
        order of constraint_ops.

    Returns
    -------
    bound : float
        Certified lower bound on the constrained minimum of the entropy production, in nats.

    Raises
    ------
    ValueError
        If rho fails the requirements above, if constraint_ops is not a three-dimensional array
        of finite Hermitian blocks matching rho, if moments or weights is not a real finite
        array of shape (n, ), or if detector_ops fails the requirements above.
    '''
    return bound

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_entropy_certificate(rho: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", detector_ops: "np.ndarray", weights: "np.ndarray") -> float:
    gradient = _oracle_objective_gradient(rho, detector_ops)
    _, operators, multipliers = _validate_exponential_family(gradient, constraint_ops, weights)
    observed = _validate_moment_vector(moments, operators.shape[0])

    shift = np.tensordot(multipliers, operators, axes=(0, 0)) if multipliers.size else 0.0
    pencil = gradient + shift
    pencil = 0.5 * (pencil + pencil.conj().T)
    return float(np.linalg.eigvalsh(pencil)[0] - float(multipliers @ observed))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    header = "import numpy as np\n"
    shared = """
def local_instance(d_local, p_z, p_x):
    omega = np.exp(2j * np.pi / d_local)
    fourier = omega ** np.outer(np.arange(d_local), np.arange(d_local)) / np.sqrt(d_local)
    stack, values = [], []
    for basis_a, basis_b, table in ((np.eye(d_local, dtype=complex), np.eye(d_local, dtype=complex), p_z),
                                    (fourier, fourier.conj(), p_x)):
        for a in range(d_local):
            for b in range(d_local):
                if a == d_local - 1 and b == d_local - 1:
                    continue
                va = basis_a[:, a].reshape(-1, 1)
                vb = basis_b[:, b].reshape(-1, 1)
                stack.append(np.kron(va @ va.conj().T, vb @ vb.conj().T))
                values.append(table[a][b])
    return np.array(stack, dtype=complex), np.array(values, dtype=float)

def local_detector(d_local, d_bob, epsilon):
    out = []
    for a in range(d_local):
        element = (epsilon / d_local) * np.eye(d_local, dtype=complex)
        element[a, a] += 1.0 - epsilon
        w, V = np.linalg.eigh(element)
        root = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T
        out.append(np.kron(root, np.eye(d_bob, dtype=complex)))
    return np.array(out, dtype=complex)

p_z = [[0.297, 0.021, 0.014],
       [0.018, 0.284, 0.026],
       [0.011, 0.023, 0.306]]
p_x = [[0.281, 0.029, 0.019],
       [0.024, 0.292, 0.031],
       [0.017, 0.026, 0.281]]
"""
    instance = header + shared + """
operators, moments = local_instance(3, p_z, p_x)
detector_ops = local_detector(3, 3, 0.04)
rho = np.eye(9, dtype=complex) / 9.0
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
        # --- Normal: benchmark linearisation state with a moderate weight vector ---
        {
            "setup": instance + """
weights = np.linspace(-0.8, 1.6, 16)
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Normal: a Gibbs-like linearisation state such as later iterations supply ---
        {
            "setup": instance + """
weights = np.linspace(-1.2, 2.4, 16)
generator = -np.tensordot(weights, operators, axes=(0, 0))
generator = 0.5 * (generator + generator.conj().T)
spectrum, vectors = np.linalg.eigh(generator)
populations = np.exp(spectrum - spectrum.max())
populations = populations / populations.sum()
rho = (vectors * populations) @ vectors.conj().T
rho = 0.5 * (rho + rho.conj().T)
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Normal: a halved weight vector, which slackens the same supporting hyperplane ---
        {
            "setup": instance + """
rng = np.random.default_rng(57)
weights = 0.5 * rng.normal(size=16) * 2.0
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Boundary: all weights zero at the maximally mixed linearisation state ---
        {
            "setup": instance + """
weights = np.zeros(16)
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Boundary: fully randomising detector, where the derivative operator vanishes ---
        {
            "setup": instance + """
detector_ops = local_detector(3, 3, 1.0)
weights = np.linspace(-2.0, 2.0, 16)
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Boundary: empty observable stack, leaving only the derivative operator ---
        {
            "setup": header + shared + """
rng = np.random.default_rng(15)
root = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
rho = root @ root.conj().T + 0.7 * np.eye(4)
rho = rho / np.trace(rho).real
operators = np.zeros((0, 4, 4), dtype=complex)
moments = np.zeros(0)
weights = np.zeros(0)
detector_ops = local_detector(2, 2, 0.07)
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Edge: qubit-qubit instance with one observable and a large weight ---
        {
            "setup": header + shared + """
vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
rho = 0.9 * np.outer(vec, vec.conj()) + 0.1 * np.eye(4, dtype=complex) / 4.0
operators = np.zeros((1, 4, 4), dtype=complex)
operators[0] = np.outer(vec, vec.conj())
moments = np.array([0.92])
weights = np.array([-35.0])
detector_ops = local_detector(2, 2, 0.02)
""",
            "call": "entropy_certificate(rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy())",
            "gold_call": "_oracle_entropy_certificate(rho, operators, moments, detector_ops, weights)",
        },
        # --- Invalid: weight vector length inconsistent with the observable stack ---
        {
            "setup": instance + invalid + """
weights = np.zeros(4)
""",
            "call": "probe(entropy_certificate, (rho.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy()))",
            "gold_call": "probe(_oracle_entropy_certificate, (rho, operators, moments, detector_ops, weights))",
        },
        # --- Invalid: singular linearisation state ---
        {
            "setup": instance + invalid + """
vec = np.zeros(9, dtype=complex)
for k in range(3):
    vec[3 * k + k] = 1.0 / np.sqrt(3.0)
singular = np.outer(vec, vec.conj())
weights = np.zeros(16)
""",
            "call": "probe(entropy_certificate, (singular.copy(), operators.copy(), moments.copy(), detector_ops.copy(), weights.copy()))",
            "gold_call": "probe(_oracle_entropy_certificate, (singular, operators, moments, detector_ops, weights))",
        },
    ]
