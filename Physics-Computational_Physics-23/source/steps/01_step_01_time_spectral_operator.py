"""
Build the Gauss node times of one time step together with the matrix that turns nodal values of a field into nodal values of its first time derivative.

Boundary element formulations of transient problems avoid finite-difference time stepping by expanding the time derivative of the field over one step in Legendre polynomials sampled at the Gauss-Legendre nodes of that step. The nodal values of the field and of its time derivative are then related by a fixed matrix of the step, assembled once and reused at every source point.

Returns
-------
tuple of two np.ndarray, float: the (P,) Gauss node times inside the step and the (P, P) nodal first time derivative operator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def time_spectral_operator(num_nodes: int, time_step: float) -> tuple:
    """Return the Gauss node times of one time step and the nodal derivative operator.

    Parameters
    ----------
    num_nodes : int
        Number of Gauss-Legendre nodes ``P`` placed inside the time step
        (``P >= 2``). The Legendre expansion is truncated at ``P`` terms, that
        is at polynomial degrees 0 to ``P - 1``.
    time_step : float
        Length of the time step (``time_step > 0``). The step is taken to run
        from time zero to ``time_step``.

    Returns
    -------
    result : tuple
        ``(times, operator)`` where ``times`` is a float array of shape ``(P,)``
        holding the Gauss node times inside the step in increasing order, and
        ``operator`` is a float array of shape ``(P, P)`` such that the nodal
        time derivatives equal ``operator`` applied to the nodal values minus
        the value carried at the start of the step, the time derivative being
        expanded over the step in Legendre polynomials of degrees 0 to
        ``P - 1``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is not an integer >= 2, or if ``time_step`` is not a
        finite real number > 0.

    """
    return (times, operator)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_time_spectral_operator(num_nodes: int, time_step: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)):
        raise ValueError("num_nodes must be an integer")
    if int(num_nodes) < 2:
        raise ValueError("num_nodes must be an integer >= 2")
    if isinstance(time_step, bool) or not isinstance(time_step, (int, float, np.integer, np.floating)):
        raise ValueError("time_step must be a real number")
    if not np.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite number > 0")

    order = int(num_nodes)
    step = float(time_step)

    # Gauss-Legendre nodes and weights of the reference interval [-1, 1].
    nodes, weights = np.polynomial.legendre.leggauss(order)

    # Spectral integration matrix. The k-th Legendre coefficient of a function
    # sampled at the nodes is (1 + 2k)/2 times the quadrature sum of the
    # function against P_k, and the running integral of the expansion from -1
    # to each node collects those coefficients against the antiderivatives of
    # the Legendre polynomials.
    spectral = np.zeros((order, order), dtype=float)
    for degree in range(order):
        basis = np.zeros(degree + 1, dtype=float)
        basis[degree] = 1.0
        values = np.polynomial.legendre.legval(nodes, basis)
        primitive = np.polynomial.legendre.legint(basis)
        running = (np.polynomial.legendre.legval(nodes, primitive)
                   - float(np.polynomial.legendre.legval(-1.0, primitive)))
        spectral += (0.5 * (1.0 + 2.0 * degree)) * np.outer(running, weights * values)

    # The reference interval carries a factor of half the step length, so that
    # the running integral in physical time is step * spectral applied to the
    # nodal values.
    spectral *= 0.5

    times = (1.0 + nodes) * step / 2.0
    operator = np.linalg.inv(spectral) / step
    return (np.asarray(times, dtype=float), np.asarray(operator, dtype=float))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- The operator must differentiate a linear ramp exactly, and the
        # node times must be the Gauss-Legendre nodes mapped onto the step. ---
        {
            "setup": """import numpy as np
step = 2.5
order = 4
slope = 3.25
def summarize(result):
    times, operator = result
    return float(np.sum(np.abs(times)) + 1000.0 * np.sum(operator @ (slope * times)))
""",
            "call": "summarize(time_spectral_operator(order, step))",
            "gold_call": "summarize(_oracle_time_spectral_operator(order, step))",
        },
        # --- A quadratic in time is inside the span of the expansion for three
        # or more nodes, so the recovered derivative is exact there too. ---
        {
            "setup": """import numpy as np
step = 1.5
order = 5
def summarize(result):
    times, operator = result
    weights = np.sqrt(np.arange(1.0, order + 1.0))
    return float(np.sum(weights * (operator @ (times ** 2 + 0.5 * times))))
""",
            "call": "summarize(time_spectral_operator(order, step))",
            "gold_call": "summarize(_oracle_time_spectral_operator(order, step))",
        },
        # --- Valid: a five node step of unit length, the benchmark setting ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
def summarize(result):
    times, operator = result
    return float(pin(operator) + 7.0 * pin(times))
""",
            "call": "summarize(time_spectral_operator(5, 1.0))",
            "gold_call": "summarize(_oracle_time_spectral_operator(5, 1.0))",
        },
        # --- Valid: a long step with more nodes, where the operator entries grow ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
""",
            "call": "float(pin(time_spectral_operator(8, 5.0)[1]))",
            "gold_call": "float(pin(_oracle_time_spectral_operator(8, 5.0)[1]))",
        },
        # --- Boundary: the smallest admissible node count ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
def summarize(result):
    times, operator = result
    return float(pin(operator) + pin(times))
""",
            "call": "summarize(time_spectral_operator(2, 0.25))",
            "gold_call": "summarize(_oracle_time_spectral_operator(2, 0.25))",
        },
        # --- Edge: the truncation error on an exponential shrinks with the node
        # count, so the recovered derivative of exp(t) is checked as a scalar.
        # The residual is a cancellation of operator entries spanning seven
        # decades, so it is only knowable to about 1e-14 and is reported at a
        # modest amplification: that keeps the comparison insensitive to the
        # order in which an implementation accumulates the Legendre degrees
        # while still separating one node count from the next by 1e-4 or more.
        {
            "setup": """import numpy as np
def summarize(result):
    times, operator = result
    return float(1.0 + 1.0e2 * float(np.sum(
        operator @ (np.exp(times) - 1.0) - np.exp(times))))
""",
            "call": "summarize(time_spectral_operator(6, 1.0))",
            "gold_call": "summarize(_oracle_time_spectral_operator(6, 1.0))",
        },
        # --- Invalid: fewer than two nodes ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        time_spectral_operator(1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_time_spectral_operator(1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive time step ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        time_spectral_operator(4, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_time_spectral_operator(4, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
