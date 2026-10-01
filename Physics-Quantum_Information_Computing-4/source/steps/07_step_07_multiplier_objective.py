"""
Evaluate the scalar objective that the protocol's data-matching step minimises over the real

weight vector attached to the constraint observables, together with its gradient.

The weights enter with the sign convention of Step 06.

The weight vector carries one real entry per constraint observable, in the order of the



supplied observable stack, and the observed moment vector uses that same order. The objective is smooth and convex. When the identity together with the constraint observables is linearly independent and the recorded moments admit a positive-definite feasible state, it is strictly convex and coercive and has a unique stationary point.







All logarithms are natural, so the returned objective is in nats. The returned gradient has one



real entry per observable, in the order of the observable stack, and is expressed in the units



of the observed moments. The objective is reported for the supplied weights and is not



minimised here.

Returns
-------
return value, gradient
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def multiplier_objective(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", weights: "np.ndarray") -> "tuple[float, np.ndarray]":
    '''Return the data-matching objective and its gradient at the supplied weights.

    Parameters
    ----------
    log_reference : np.ndarray
        Hermitian array of shape (d, d) holding the logarithm of the positive-definite
        reference operator. Hermiticity is required within 1e-9.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables. Hermiticity is required within 1e-9.
    moments : np.ndarray
        Real array of shape (n, ) with finite entries, holding the observed value of each
        observable in the order of constraint_ops.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in
        the order of constraint_ops.

    Returns
    -------
    value : float
        Objective value at the supplied weights, in nats.
    gradient : np.ndarray
        Real array of shape (n, ) holding the gradient of the objective with respect to the
        weights.

    Raises
    ------
    ValueError
        If log_reference is not a Hermitian square two-dimensional array with at least one row
        and finite entries, if constraint_ops is not a three-dimensional array of finite
        Hermitian (d, d) blocks matching log_reference, or if moments or weights is not a real
        finite array of shape (n, ).
    '''
    return value, gradient

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_moment_vector(moments: "np.ndarray", count: int) -> "np.ndarray":
    """Return the observed-moment vector as a validated float array."""
    values = np.asarray(moments)
    if np.iscomplexobj(values):
        raise ValueError("moments must be real")
    values = values.astype(float)
    if values.shape != (count, ):
        raise ValueError("moments must have shape (n, ) matching constraint_ops")
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("moments must have finite entries")
    return values


def _oracle_multiplier_objective(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", weights: "np.ndarray") -> "tuple[float, np.ndarray]":
    reference, operators, multipliers = _validate_exponential_family(
        log_reference, constraint_ops, weights)
    observed = _validate_moment_vector(moments, operators.shape[0])

    state, log_partition = _oracle_gibbs_state(reference, operators, multipliers)
    if operators.shape[0]:
        expectations = np.einsum('iab,ba->i', operators, state).real
    else:
        expectations = np.zeros(0, dtype=float)

    value = float(log_partition + float(multipliers @ observed))
    return value, observed - expectations

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
            "call": "multiplier_objective(log_reference.copy(), operators.copy(), moments.copy(), weights.copy())",
            "gold_call": "_oracle_multiplier_objective(log_reference, operators, moments, weights)",
        },
        # --- Normal: a full-rank non-identity reference, as met from the second iteration on ---
        {
            "setup": instance + """
rng = np.random.default_rng(17)
root = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
positive = root @ root.conj().T + 0.9 * np.eye(9)
positive = positive / np.trace(positive).real
spectrum, vectors = np.linalg.eigh(positive)
log_reference = (vectors * np.log(spectrum)) @ vectors.conj().T
weights = rng.normal(size=16) * 0.8
""",
            "call": "multiplier_objective(log_reference.copy(), operators.copy(), moments.copy(), weights.copy())",
            "gold_call": "_oracle_multiplier_objective(log_reference, operators, moments, weights)",
        },
        # --- Boundary: all weights zero, where the gradient is the raw moment mismatch ---
        {
            "setup": instance + """
weights = np.zeros(16)
""",
            "call": "multiplier_objective(log_reference.copy(), operators.copy(), moments.copy(), weights.copy())",
            "gold_call": "_oracle_multiplier_objective(log_reference, operators, moments, weights)",
        },
        # --- Boundary: empty observable stack, where the objective reduces to the normalisation ---
        {
            "setup": header + """
rng = np.random.default_rng(9)
root = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
positive = root @ root.conj().T + 0.5 * np.eye(4)
spectrum, vectors = np.linalg.eigh(positive)
log_reference = (vectors * np.log(spectrum)) @ vectors.conj().T
operators = np.zeros((0, 4, 4), dtype=complex)
moments = np.zeros(0)
weights = np.zeros(0)
""",
            "call": "multiplier_objective(log_reference.copy(), operators.copy(), moments.copy(), weights.copy())",
            "gold_call": "_oracle_multiplier_objective(log_reference, operators, moments, weights)",
        },
        # --- Edge: weights large enough to concentrate the state on one eigenvector ---
        {
            "setup": instance + """
weights = np.zeros(16)
weights[4] = -260.0
weights[11] = 190.0
""",
            "call": "multiplier_objective(log_reference.copy(), operators.copy(), moments.copy(), weights.copy())",
            "gold_call": "_oracle_multiplier_objective(log_reference, operators, moments, weights)",
        },
        # --- Edge: a single commuting observable with a moment at the edge of its range ---
        {
            "setup": header + """
log_reference = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)
operators = np.zeros((1, 4, 4), dtype=complex)
operators[0, 0, 0] = 1.0
moments = np.array([0.999])
weights = np.array([-6.5])
""",
            "call": "multiplier_objective(log_reference.copy(), operators.copy(), moments.copy(), weights.copy())",
            "gold_call": "_oracle_multiplier_objective(log_reference, operators, moments, weights)",
        },
        # --- Invalid: moment vector length inconsistent with the observable stack ---
        {
            "setup": header + invalid + """
log_reference = np.zeros((2, 2), dtype=complex)
operators = np.zeros((2, 2, 2), dtype=complex)
operators[0, 0, 0] = 1.0
operators[1, 1, 1] = 1.0
moments = np.array([0.5])
weights = np.zeros(2)
""",
            "call": "probe(multiplier_objective, (log_reference.copy(), operators.copy(), moments.copy(), weights.copy()))",
            "gold_call": "probe(_oracle_multiplier_objective, (log_reference, operators, moments, weights))",
        },
        # --- Invalid: non-finite weight entry ---
        {
            "setup": header + invalid + """
log_reference = np.zeros((2, 2), dtype=complex)
operators = np.zeros((1, 2, 2), dtype=complex)
operators[0, 0, 0] = 1.0
moments = np.array([0.5])
weights = np.array([np.inf])
""",
            "call": "probe(multiplier_objective, (log_reference.copy(), operators.copy(), moments.copy(), weights.copy()))",
            "gold_call": "probe(_oracle_multiplier_objective, (log_reference, operators, moments, weights))",
        },
    ]
