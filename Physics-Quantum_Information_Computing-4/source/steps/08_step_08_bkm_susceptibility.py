"""
Evaluate the curvature of the data-matching objective at a given real weight vector, returning

the matrix of its second derivatives with respect to the weights.

The weight vector carries one real entry per constraint observable, in the order of the supplied

observable stack, and the returned matrix uses that same ordering for both of its indices.



The curvature is real and symmetric, and it is positive definite whenever the constraint

observables together with the identity are linearly independent, which is what makes the

data-matching objective strictly convex. It is expressed in the squared units of the observed

moments, and it involves no observed moments itself: it depends only on the reference logarithm,

the observable stack and the weights. Nearly degenerate spectra of the exponential family occur

in the pipeline, so the evaluation must stay accurate when two spectral values coincide or

differ by many orders of magnitude.

Returns
-------
return curvature
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bkm_susceptibility(log_reference: "np.ndarray", constraint_ops: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    '''Return the second-derivative matrix of the data-matching objective at the weights.

    Parameters
    ----------
    log_reference : np.ndarray
        Hermitian array of shape (d, d) holding the logarithm of the positive-definite
        reference operator. Hermiticity is required within 1e-9.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables. Hermiticity is required within 1e-9.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in
        the order of constraint_ops.

    Returns
    -------
    curvature : np.ndarray
        Real symmetric array of shape (n, n) holding the second derivatives of the objective
        with respect to the weights.

    Raises
    ------
    ValueError
        If log_reference is not a Hermitian square two-dimensional array with at least one row
        and finite entries, if constraint_ops is not a three-dimensional array of finite
        Hermitian (d, d) blocks matching log_reference, or if weights is not a real finite
        array of shape (n, ).
    '''
    return curvature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _logarithmic_mean_kernel(exponents: "np.ndarray", populations: "np.ndarray") -> "np.ndarray":
    """Return the matrix of logarithmic means of the populations, keyed by their exponents."""
    gaps = exponents[:, None] - exponents[None, :]
    safe = np.where(gaps == 0.0, 1.0, gaps)
    near = populations[None, :] * np.expm1(np.clip(gaps, -50.0, 50.0)) / safe
    near = np.where(gaps == 0.0, np.broadcast_to(populations[None, :], gaps.shape), near)
    far = (populations[:, None] - populations[None, :]) / safe
    return np.where(np.abs(gaps) <= 1.0, near, far)


def _oracle_bkm_susceptibility(log_reference: "np.ndarray", constraint_ops: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    reference, operators, multipliers = _validate_exponential_family(
        log_reference, constraint_ops, weights)
    count = operators.shape[0]
    if count == 0:
        return np.zeros((0, 0), dtype=float)

    generator = _effective_hamiltonian(reference, operators, multipliers)
    exponents, vectors = np.linalg.eigh(generator)
    populations = np.exp(exponents - float(exponents[-1]))
    populations = populations / populations.sum()

    rotated = np.einsum('pa,ipq,qb->iab', vectors.conj(), operators, vectors)
    expectations = np.einsum('a,iaa->i', populations, rotated).real
    identity = np.eye(reference.shape[0], dtype=complex)
    centered = rotated - expectations[:, None, None] * identity[None, :, :]

    kernel = _logarithmic_mean_kernel(exponents, populations)
    curvature = np.einsum('ab,iab,jba->ij', kernel, centered, centered).real
    return 0.5 * (curvature + curvature.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    header = "import numpy as np\n"

    instance = header + """
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

p_z = [[0.297, 0.021, 0.014],
       [0.018, 0.284, 0.026],
       [0.011, 0.023, 0.306]]
p_x = [[0.281, 0.029, 0.019],
       [0.024, 0.292, 0.031],
       [0.017, 0.026, 0.281]]
""" + """
operators, moments = local_instance(3, p_z, p_x)
log_reference = np.log(1.0 / 9.0) * np.eye(9, dtype=complex)
"""

    curvature_probe = """
def curvature_probe(fn, log_reference, operators, weights):
    curvature = fn(log_reference, operators, weights)
    spectrum = np.linalg.eigvalsh(curvature)
    return np.array([float(np.abs(curvature - curvature.T).max()),
                     float(spectrum[0]), float(spectrum[-1]),
                     float(np.trace(curvature)), float(np.abs(curvature).max())])
"""

    second_difference_probe = """
def second_difference_probe(fn, moment_fn, log_reference, operators, moments, weights, seed):
    curvature = fn(log_reference, operators, weights)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        direction = rng.normal(size=weights.size)
        direction = direction / np.linalg.norm(direction)
        step = 1e-5
        _, grad_plus = moment_fn(log_reference, operators, moments, weights + step * direction)
        _, grad_minus = moment_fn(log_reference, operators, moments, weights - step * direction)
        numeric = (grad_plus - grad_minus) / (2.0 * step)
        out.append(float(np.abs(numeric - curvature @ direction).max()))
    return np.array(out)
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
        # --- Normal: benchmark instance at moderate weights ---
        {
            "setup": instance + """
weights = np.linspace(-1.5, 2.0, 16)
""",
            "call": "bkm_susceptibility(log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "_oracle_bkm_susceptibility(log_reference, operators, weights)",
        },
        # --- Normal: symmetry, positivity and scale of the curvature at random weights ---
        {
            "setup": instance + curvature_probe + """
rng = np.random.default_rng(45)
weights = rng.normal(size=16) * 1.2
""",
            "call": "curvature_probe(bkm_susceptibility, log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "curvature_probe(_oracle_bkm_susceptibility, log_reference, operators, weights)",
        },
        # --- Normal: curvature reproduces directional differences of the objective gradient ---
        {
            "setup": instance + second_difference_probe + """
rng = np.random.default_rng(46)
weights = rng.normal(size=16) * 0.5
""",
            "call": "second_difference_probe(bkm_susceptibility, multiplier_objective, log_reference.copy(), operators.copy(), moments.copy(), weights.copy(), 46)",
            "gold_call": "second_difference_probe(_oracle_bkm_susceptibility, _oracle_multiplier_objective, log_reference, operators, moments, weights, 46)",
            "tol": 1e-6,
        },
        # --- Boundary: all weights zero, where the exponential family is the reference ---
        {
            "setup": instance + """
weights = np.zeros(16)
""",
            "call": "bkm_susceptibility(log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "_oracle_bkm_susceptibility(log_reference, operators, weights)",
        },
        # --- Boundary: fully degenerate spectrum with non-commuting observables ---
        {
            "setup": header + """
log_reference = np.zeros((4, 4), dtype=complex)
operators = np.zeros((3, 4, 4), dtype=complex)
operators[0, 0, 0] = 1.0
operators[0, 1, 1] = 1.0
operators[1, 0, 1] = 0.5
operators[1, 1, 0] = 0.5
operators[2, 2, 3] = 0.5j
operators[2, 3, 2] = -0.5j
weights = np.zeros(3)
""",
            "call": "bkm_susceptibility(log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "_oracle_bkm_susceptibility(log_reference, operators, weights)",
        },
        # --- Edge: spectral values separated by many orders of magnitude ---
        {
            "setup": header + """
log_reference = np.diag([0.0, -0.5, -30.0, -120.0]).astype(complex)
operators = np.zeros((2, 4, 4), dtype=complex)
operators[0, 0, 0] = 1.0
operators[0, 2, 2] = 1.0
operators[1, 1, 3] = 0.5
operators[1, 3, 1] = 0.5
weights = np.array([2.5, -3.5])
""",
            "call": "bkm_susceptibility(log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "_oracle_bkm_susceptibility(log_reference, operators, weights)",
        },
        # --- Boundary: empty observable stack, leaving no weights to differentiate ---
        {
            "setup": header + """
rng = np.random.default_rng(19)
root = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
positive = root @ root.conj().T + 0.5 * np.eye(4)
spectrum, vectors = np.linalg.eigh(positive)
log_reference = (vectors * np.log(spectrum)) @ vectors.conj().T
operators = np.zeros((0, 4, 4), dtype=complex)
weights = np.zeros(0)
""",
            "call": "bkm_susceptibility(log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "_oracle_bkm_susceptibility(log_reference, operators, weights)",
        },
        # --- Edge: observable proportional to the identity, giving a vanishing curvature ---
        {
            "setup": header + """
log_reference = np.diag([0.0, -1.0, -2.0]).astype(complex)
operators = np.zeros((1, 3, 3), dtype=complex)
operators[0] = np.eye(3, dtype=complex)
weights = np.array([0.75])
""",
            "call": "bkm_susceptibility(log_reference.copy(), operators.copy(), weights.copy())",
            "gold_call": "_oracle_bkm_susceptibility(log_reference, operators, weights)",
        },
        # --- Invalid: observable stack whose blocks do not match the reference dimension ---
        {
            "setup": header + invalid + """
log_reference = np.zeros((3, 3), dtype=complex)
operators = np.zeros((1, 2, 2), dtype=complex)
operators[0, 0, 0] = 1.0
weights = np.array([0.1])
""",
            "call": "probe(bkm_susceptibility, (log_reference.copy(), operators.copy(), weights.copy()))",
            "gold_call": "probe(_oracle_bkm_susceptibility, (log_reference, operators, weights))",
        },
    ]
