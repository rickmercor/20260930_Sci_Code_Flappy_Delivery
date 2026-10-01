"""
Run the complete pipeline for the two-setting bipartite instance and return the final candidate's entropy-production value and the final iteration's certified lower bound, both in nats.

Alice and Bob each hold a d_local-dimensional system, the joint space is ordered so that the



basis label of the pair (a, b) is a * d_local + b, and Alice's raw key is the symbol reported by



a detector that returns her computational-basis outcome except that, with probability epsilon, it



returns a symbol drawn uniformly at random instead. Parameter estimation records the computational



setting and the Fourier setting described by the two probability tables p_z and p_x, laid out with



rows indexed by Alice's outcome and columns by Bob's outcome.







The run starts from the maximally mixed state on the joint space and performs n_outer outer



iterations. Each inner search starts from the all-zero weight vector, stops once every entry of



its objective gradient is below tol in magnitude, and takes at most max_iter steps. The returned



candidate value is the entropy production of the state reached after the final outer iteration,



and the returned bound is the one certified by the final outer iteration, that is the bound



obtained from the state that entered that iteration together with the weight vector that



iteration produced. Both are in nats.



The first returned value is an upper bound on the constrained minimum only when the final candidate satisfies the recorded moment constraints. An exhausted inner-search budget can leave the candidate infeasible; in that case the first value remains its entropy-production value and is not asserted to be an upper bound. The lower certificate is valid independently of inner convergence.

Returns
-------
return candidate, bound
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certified_entropy_bracket(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float, n_outer: int, tol: float, max_iter: int) -> "tuple[float, float]":
    '''Return the candidate value and the certified lower bound of the instance, in nats.

    Parameters
    ----------
    d_local : int
        Local Hilbert-space dimension of Alice and of Bob, at least 2.
    p_z : np.ndarray
        Real array of shape (d_local, d_local) holding the computational-setting outcome
        probabilities, nonnegative and summing to one within 1e-9.
    p_x : np.ndarray
        Real array of shape (d_local, d_local) holding the Fourier-setting outcome
        probabilities, with the same layout and validity requirements as p_z.
    epsilon : float
        Detector randomisation probability, strictly greater than 0 and at most 1.
    n_outer : int
        Number of outer iterations, at least 1.
    tol : float
        Positive stopping threshold used by each inner search.
    max_iter : int
        Nonnegative step budget for each inner search.

    Returns
    -------
    candidate : float
        Entropy production of the state reached after the final outer iteration, in nats.
    bound : float
        Certified lower bound produced by the final outer iteration, in nats.

    Raises
    ------
    ValueError
        If d_local is not an integer of at least 2, if p_z or p_x fails the requirements above,
        if epsilon is not a real scalar in the half-open interval from 0 exclusive to 1
        inclusive, if n_outer is not an integer of at least 1, if tol is not a positive finite
        real scalar, or if max_iter is not a nonnegative integer.
    '''
    return candidate, bound

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_certified_entropy_bracket(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float, n_outer: int, tol: float, max_iter: int) -> "tuple[float, float]":
    operators, moments, detector = _oracle_build_protocol_data(d_local, p_z, p_x, epsilon)
    if isinstance(n_outer, bool) or not isinstance(n_outer, (int, np.integer)):
        raise ValueError("n_outer must be an integer")
    if int(n_outer) < 1:
        raise ValueError("n_outer must be at least 1")

    dimension = int(d_local) ** 2
    state = np.eye(dimension, dtype=complex) / dimension
    bound = 0.0
    for _ in range(int(n_outer)):
        linearisation = state
        state, weights = _oracle_outer_iteration(
            linearisation, operators, moments, detector, tol, max_iter)
        bound = _oracle_entropy_certificate(
            linearisation, operators, moments, detector, weights)

    candidate = _oracle_entropy_production(state, detector)
    return candidate, bound

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    header = "import numpy as np\n"

    benchmark = header + """
p_z = np.array([[0.297, 0.021, 0.014],
                [0.018, 0.284, 0.026],
                [0.011, 0.023, 0.306]])
p_x = np.array([[0.281, 0.029, 0.019],
                [0.024, 0.292, 0.031],
                [0.017, 0.026, 0.281]])
"""

    bracket_probe = """
def bracket_probe(fn, d_local, p_z, p_x, epsilon, n_outer, tol, max_iter):
    candidate, bound = fn(d_local, p_z, p_x, epsilon, n_outer, tol, max_iter)
    return np.array([candidate, bound, float(bound <= candidate), candidate - bound])
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
        # --- Normal: the benchmark qutrit instance at the prescribed iteration budget ---
        {
            "setup": benchmark,
            "call": "certified_entropy_bracket(3, p_z.copy(), p_x.copy(), 0.04, 5, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(3, p_z, p_x, 0.04, 5, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Normal: the bracket orders correctly at the prescribed budget ---
        {
            "setup": benchmark + bracket_probe,
            "call": "bracket_probe(certified_entropy_bracket, 3, p_z.copy(), p_x.copy(), 0.04, 5, 1e-12, 100)",
            "gold_call": "bracket_probe(_oracle_certified_entropy_bracket, 3, p_z, p_x, 0.04, 5, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Normal: a longer budget on the same benchmark instance ---
        {
            "setup": benchmark,
            "call": "certified_entropy_bracket(3, p_z.copy(), p_x.copy(), 0.04, 7, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(3, p_z, p_x, 0.04, 7, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Normal: a much noisier detector on the same recorded data ---
        {
            "setup": benchmark,
            "call": "certified_entropy_bracket(3, p_z.copy(), p_x.copy(), 0.3, 5, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(3, p_z, p_x, 0.3, 5, 1e-12, 100)",
            "tol": 1e-6,
        },
        # --- Boundary: a single outer iteration, linearised at the maximally mixed state ---
        {
            "setup": benchmark,
            "call": "certified_entropy_bracket(3, p_z.copy(), p_x.copy(), 0.04, 1, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(3, p_z, p_x, 0.04, 1, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Boundary: smallest local dimension with a mildly noisy data set ---
        {
            "setup": header + """
p_z = np.array([[0.46, 0.04],
                [0.03, 0.47]])
p_x = np.array([[0.44, 0.06],
                [0.05, 0.45]])
""",
            "call": "certified_entropy_bracket(2, p_z.copy(), p_x.copy(), 0.05, 5, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(2, p_z, p_x, 0.05, 5, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Boundary: fully randomising detector, where the record carries no key at all ---
        {
            "setup": benchmark,
            "call": "certified_entropy_bracket(3, p_z.copy(), p_x.copy(), 1.0, 3, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(3, p_z, p_x, 1.0, 3, 1e-12, 100)",
            "tol": 1e-10,
        },
        # --- Edge: uniform tables, where the maximally mixed state already matches the data ---
        {
            "setup": header + """
p_z = np.full((3, 3), 1.0 / 9.0)
p_x = np.full((3, 3), 1.0 / 9.0)
""",
            "call": "certified_entropy_bracket(3, p_z.copy(), p_x.copy(), 0.04, 4, 1e-12, 100)",
            "gold_call": "_oracle_certified_entropy_bracket(3, p_z, p_x, 0.04, 4, 1e-12, 100)",
            "tol": 1e-8,
        },
        # --- Edge: a fourth local dimension, which enlarges both the space and the data set ---
        {
            "setup": header + """
p_z = np.full((4, 4), 0.01)
np.fill_diagonal(p_z, 0.22)
p_z[3, 3] = 0.22 + (1.0 - p_z.sum())
p_x = np.full((4, 4), 0.012)
np.fill_diagonal(p_x, 0.214)
p_x[3, 3] = 0.214 + (1.0 - p_x.sum())
""",
            "call": "certified_entropy_bracket(4, p_z.copy(), p_x.copy(), 0.05, 3, 1e-12, 150)",
            "gold_call": "_oracle_certified_entropy_bracket(4, p_z, p_x, 0.05, 3, 1e-12, 150)",
            "tol": 1e-7,
        },
        # --- Invalid: non-positive number of outer iterations ---
        {
            "setup": benchmark + invalid,
            "call": "probe(certified_entropy_bracket, (3, p_z.copy(), p_x.copy(), 0.04, 0, 1e-12, 100))",
            "gold_call": "probe(_oracle_certified_entropy_bracket, (3, p_z, p_x, 0.04, 0, 1e-12, 100))",
        },
        # --- Invalid: detector randomisation probability at zero ---
        {
            "setup": benchmark + invalid,
            "call": "probe(certified_entropy_bracket, (3, p_z.copy(), p_x.copy(), 0.0, 5, 1e-12, 100))",
            "gold_call": "probe(_oracle_certified_entropy_bracket, (3, p_z, p_x, 0.0, 5, 1e-12, 100))",
        },
    ]
