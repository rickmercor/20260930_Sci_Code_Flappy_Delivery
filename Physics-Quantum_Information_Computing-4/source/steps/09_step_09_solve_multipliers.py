"""
Locate the real weight vector at which the data-matching objective attains its minimum, and

report how far the resulting exponential-family state still is from reproducing the observed

moments.

The search is a damped second-order iteration driven by the curvature of the objective. It stops

as soon as every entry of the objective gradient is below tol in magnitude, and it takes at most

max_iter steps; when the budget is exhausted the last weight vector reached is returned, so a

budget of zero returns the supplied starting vector unchanged.



The weight vector carries one real entry per constraint observable, in the order of the supplied

observable stack, and the observed moment vector uses that same order. The returned residual is

the largest magnitude among the gradient entries at the returned weight vector, expressed in the

units of the observed moments.

Returns
-------
return weights, residual
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_multipliers(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", initial_weights: "np.ndarray", tol: float, max_iter: int) -> "tuple[np.ndarray, float]":
    '''Return the minimizing weight vector and the residual reached at it.

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
    initial_weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding the starting weight vector.
    tol : float
        Positive stopping threshold on the largest magnitude among the gradient entries.
    max_iter : int
        Nonnegative maximum number of second-order steps.

    Returns
    -------
    weights : np.ndarray
        Real array of shape (n, ) holding the weight vector reached.
    residual : float
        Largest magnitude among the gradient entries at the returned weight vector.

    Raises
    ------
    ValueError
        If log_reference is not a Hermitian square two-dimensional array with at least one row
        and finite entries, if constraint_ops is not a three-dimensional array of finite
        Hermitian (d, d) blocks matching log_reference, if moments or initial_weights is not a
        real finite array of shape (n, ), if tol is not a positive finite real scalar, or if
        max_iter is not a nonnegative integer.
    '''
    return weights, residual

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_multipliers(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", initial_weights: "np.ndarray", tol: float, max_iter: int) -> "tuple[np.ndarray, float]":
    reference, operators, weights = _validate_exponential_family(
        log_reference, constraint_ops, initial_weights)
    observed = _validate_moment_vector(moments, operators.shape[0])
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.integer, np.floating)):
        raise ValueError("tol must be a real scalar")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be positive and finite")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("max_iter must be an integer")
    if int(max_iter) < 0:
        raise ValueError("max_iter must be nonnegative")

    threshold = float(tol)
    budget = int(max_iter)
    value, gradient = _oracle_multiplier_objective(reference, operators, observed, weights)

    for _ in range(budget):
        residual = float(np.abs(gradient).max()) if gradient.size else 0.0
        if residual <= threshold:
            break
        curvature = _oracle_bkm_susceptibility(reference, operators, weights)
        try:
            displacement = -np.linalg.solve(curvature, gradient)
        except np.linalg.LinAlgError:
            displacement = -np.linalg.lstsq(curvature, gradient, rcond=None)[0]
        predicted = float(gradient @ displacement)
        scale = 1.0
        trial_value, trial_gradient = _oracle_multiplier_objective(
            reference, operators, observed, weights + scale * displacement)
        for _ in range(60):
            if trial_value <= value + 1e-4 * scale * predicted:
                break
            scale *= 0.5
            trial_value, trial_gradient = _oracle_multiplier_objective(
                reference, operators, observed, weights + scale * displacement)
        weights = weights + scale * displacement
        value, gradient = trial_value, trial_gradient

    residual = float(np.abs(gradient).max()) if gradient.size else 0.0
    return weights, residual

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
start = np.zeros(16)
"""

    matching_probe = """
def matching_probe(fn, gibbs_fn, log_reference, operators, moments, start, tol, max_iter):
    weights, residual = fn(log_reference, operators, moments, start, tol, max_iter)
    state, _ = gibbs_fn(log_reference, operators, weights)
    achieved = np.einsum('iab,ba->i', operators, state).real
    spectrum = np.linalg.eigvalsh(state)
    return np.array([float(np.abs(achieved - moments).max()), residual,
                     float(np.trace(state).real), float(spectrum[0] > 0.0)])
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
        # --- Normal: benchmark instance solved from the all-zero start ---
        {
            "setup": instance,
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-12, 100)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, start, 1e-12, 100)",
            "tol": 1e-7,
        },
        # --- Normal: the returned weights reproduce the observed moments ---
        {
            "setup": instance + matching_probe,
            "call": "matching_probe(solve_multipliers, gibbs_state, log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-12, 100)",
            "gold_call": "matching_probe(_oracle_solve_multipliers, _oracle_gibbs_state, log_reference, operators, moments, start, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Normal: a distant start reaches the same minimizer ---
        {
            "setup": instance + """
start = np.linspace(-4.0, 6.0, 16)
""",
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-12, 200)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, start, 1e-12, 200)",
            "tol": 1e-6,
        },
        # --- Boundary: zero step budget, so the start is returned with its own residual ---
        {
            "setup": instance,
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-12, 0)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, start, 1e-12, 0)",
        },
        # --- Boundary: a start that already satisfies the stopping threshold ---
        {
            "setup": instance + """
converged = np.array([4.240305860843e-02, 2.655598463479e+00, 3.230619067046e+00,
                      2.862378688667e+00, 1.063208155680e-01, 2.363053638344e+00,
                      3.620862270583e+00, 2.520159860722e+00, 2.137763865530e-04,
                      2.203484348282e+00, 2.805659028416e+00, 2.442481842883e+00,
                      -5.947979830334e-02, 2.105494943263e+00, 3.049528008674e+00,
                      2.321842802283e+00])
""",
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), converged.copy(), 1e-9, 100)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, converged, 1e-9, 100)",
            "tol": 1e-9,
        },
        # --- Boundary: empty observable stack, where there is nothing to match ---
        {
            "setup": header + """
log_reference = np.diag([0.0, -1.0, -2.0]).astype(complex)
operators = np.zeros((0, 3, 3), dtype=complex)
moments = np.zeros(0)
start = np.zeros(0)
""",
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-12, 50)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, start, 1e-12, 50)",
        },
        # --- Edge: a nearly extreme target moment, where the minimizer is large ---
        {
            "setup": header + """
log_reference = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)
operators = np.zeros((1, 4, 4), dtype=complex)
operators[0, 0, 0] = 1.0
moments = np.array([1.0 - 1e-6])
start = np.zeros(1)
""",
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-14, 300)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, start, 1e-14, 300)",
            "tol": 1e-6,
        },
        # --- Edge: two non-commuting observables on a degenerate reference ---
        {
            "setup": header + """
log_reference = np.zeros((2, 2), dtype=complex)
operators = np.zeros((2, 2, 2), dtype=complex)
operators[0, 0, 0] = 1.0
operators[1, 0, 1] = 0.5
operators[1, 1, 0] = 0.5
moments = np.array([0.8, 0.3])
start = np.zeros(2)
""",
            "call": "solve_multipliers(log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-13, 200)",
            "gold_call": "_oracle_solve_multipliers(log_reference, operators, moments, start, 1e-13, 200)",
            "tol": 1e-7,
        },
        # --- Invalid: non-positive stopping threshold ---
        {
            "setup": instance + invalid,
            "call": "probe(solve_multipliers, (log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 0.0, 100))",
            "gold_call": "probe(_oracle_solve_multipliers, (log_reference, operators, moments, start, 0.0, 100))",
        },
        # --- Invalid: negative step budget ---
        {
            "setup": instance + invalid,
            "call": "probe(solve_multipliers, (log_reference.copy(), operators.copy(), moments.copy(), start.copy(), 1e-12, -1))",
            "gold_call": "probe(_oracle_solve_multipliers, (log_reference, operators, moments, start, 1e-12, -1))",
        },
    ]
