"""
Return the non-negative emission-rate spectral density obtained from one weighted inversion system or a temperature-indexed batch by Tikhonov-regularized least squares.

Inverting a sum of exponentials is the textbook ill-posed problem: the kernel columns belonging to neighbouring rates are nearly parallel, so the system has a singular-value spectrum that decays geometrically and an unregularised solution amplifies measurement noise without bound. Adding a penalty proportional to the squared norm of the spectral density trades a small, controlled increase in the residual for an enormous reduction in that amplification. The penalised problem is equivalent to an ordinary least-squares problem on a system in which the kernel is stacked on top of the regularisation parameter times the identity and the right-hand side is padded with zeros, which is how it is solved in practice.




The physical constraint that a spectral density cannot be negative is the second, and in this setting the more powerful, stabiliser. For a positive Tikhonov parameter the ridge term makes the quadratic objective strictly convex and hence gives a unique minimiser; restricting that minimiser to the non-negative orthant enforces physical admissibility and suppresses the oscillating positive and negative lobes that an unconstrained inversion places around genuine features. Non-negativity and an L2 penalty do not mathematically enforce sparsity. In this particular discrete-level data set the non-negative least-squares solution nevertheless has clustered support, with each level represented by a few adjacent non-zero grid weights rather than a broad continuous band.




The regularisation parameter controls the width of those clusters directly. Too small a value lets noise fragment a single level into several spurious features; too large a value merges neighbouring levels and, more insidiously, narrows and symmetrises a weak feature that sits beside a much stronger one, biasing any amplitude read off the spectrum. That bias is the reason amplitudes are better determined elsewhere, and it is why the spectrum is used here for what it does well, which is locating the rates. For a temperature sweep, each record has its own right-hand side and must be solved as an independent constrained inverse problem; batching preserves that independence rather than coupling the densities across temperature.

Returns
-------
np.ndarray of shape (n_rates,) or (n_records, n_rates), float: non-negative spectral density in pF per grid rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_regularized_spectrum(system: np.ndarray, regularization: float) -> np.ndarray:
    """Return the non-negative spectral density of the weighted inversion system.

    Parameters
    ----------
    system : np.ndarray
        One array of shape (n_kept, n_rates + 1), or a temperature-indexed
        batch of shape (n_records, n_kept, n_rates + 1). The final axis holds
        the weighted exponential kernel followed by the weighted right-hand
        side.
    regularization : float
        Tikhonov regularization parameter (regularization >= 0), in the same
        units as the right-hand side divided by the spectral density.

    Returns
    -------
    density : np.ndarray
        For one system, shape (n_rates,); for a batch, shape
        (n_records, n_rates). Each row is the independently recovered
        non-negative density for the corresponding input record.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return density  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_regularized_spectrum(system: np.ndarray, regularization: float) -> np.ndarray:
    import numpy as np
    from scipy.optimize import nnls

    system = np.asarray(system, dtype=float)

    if (system.ndim not in (2, 3) or system.shape[-1] < 2
            or system.shape[-2] < 1 or (system.ndim == 3 and system.shape[0] < 1)):
        raise ValueError(
            "system must have shape (n_kept, n_rates + 1) or "
            "(n_records, n_kept, n_rates + 1)")
    if not np.all(np.isfinite(system)):
        raise ValueError("system must be finite")
    if (isinstance(regularization, bool)
            or not isinstance(regularization, (int, float, np.floating, np.integer))
            or not np.isfinite(regularization) or float(regularization) < 0.0):
        raise ValueError("regularization must be a finite number >= 0")

    def solve_one(one_system):
        kernel = one_system[:, :-1]
        rhs = one_system[:, -1]
        n_rates = kernel.shape[1]

        # The penalised problem is an ordinary least-squares problem on the
        # kernel stacked on lambda times the identity, with zeros appended to b.
        stacked = np.vstack([kernel, float(regularization) * np.eye(n_rates)])
        padded = np.concatenate([rhs, np.zeros(n_rates)])

        # For lambda > 0 the ridge term, not non-negativity, makes the
        # quadratic objective strictly convex.  At zero, uniqueness is not
        # guaranteed for a rank-deficient kernel.
        density, _ = nnls(stacked, padded)
        return density

    if system.ndim == 2:
        return solve_one(system)
    return np.stack([solve_one(one_system) for one_system in system])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark regularization on a two-level transient ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 89)
counts = np.maximum(np.round(np.geomspace(1.0, 8000.0, 89)), 1.0)
values = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
grid = np.geomspace(0.1, 1.0e5, 150)
w = np.sqrt(counts)
system = np.column_stack([np.exp(-np.outer(times, grid)) * w[:, None], -(values - values[-1]) * w])
""",
            "call": "float(np.sum(solve_regularized_spectrum(system, 0.01)))",
            "gold_call": "float(np.sum(_oracle_solve_regularized_spectrum(system, 0.01)))",
        },
        # --- Valid: a batch is solved record by record without cross-coupling ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 0.8, 55)
counts = np.maximum(np.round(np.geomspace(1.0, 4000.0, 55)), 1.0)
grid = np.geomspace(0.5, 2.0e4, 70)
w = np.sqrt(counts)
v1 = 204.5 - 0.72 * np.exp(-65.0 * times) - 0.06 * np.exp(-1300.0 * times)
v2 = 204.7 - 0.58 * np.exp(-105.0 * times) - 0.11 * np.exp(-760.0 * times)
kernel = np.exp(-np.outer(times, grid)) * w[:, None]
s1 = np.column_stack([kernel, -(v1 - v1[-1]) * w])
s2 = np.column_stack([kernel, -(v2 - v2[-1]) * w])
system = np.stack([s1, s2])
""",
            "call": "solve_regularized_spectrum(system, 0.015) + 1000.0",
            "gold_call": "_oracle_solve_regularized_spectrum(system, 0.015) + 1000.0",
        },
        # --- Valid: non-negativity and objective value are solver-independent ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 89)
counts = np.maximum(np.round(np.geomspace(1.0, 8000.0, 89)), 1.0)
values = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
grid = np.geomspace(0.1, 1.0e5, 150)
w = np.sqrt(counts)
system = np.column_stack([np.exp(-np.outer(times, grid)) * w[:, None], -(values - values[-1]) * w])
def summarize(density):
    density = np.asarray(density, dtype=float)
    kernel = system[:, :-1]
    rhs = system[:, -1]
    objective = (np.linalg.norm(kernel @ density - rhs) ** 2
                 + 0.01 ** 2 * np.dot(density, density))
    return np.array([float(np.all(density >= -1.0e-12)),
                     float(np.round(objective, 10))])
""",
            "call": "summarize(solve_regularized_spectrum(system, 0.01))",
            "gold_call": "summarize(_oracle_solve_regularized_spectrum(system, 0.01))",
        },
        # --- Valid: a stronger penalty broadens the recovered clusters ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-4, 1.0, 50)
counts = np.geomspace(1.0, 2000.0, 50)
values = 150.0 - 0.4 * np.exp(-200.0 * times)
grid = np.geomspace(1.0, 1.0e4, 60)
w = np.sqrt(counts)
system = np.column_stack([np.exp(-np.outer(times, grid)) * w[:, None], -(values - values[-1]) * w])
""",
            "call": "float(np.sum(solve_regularized_spectrum(system, 0.2)))",
            "gold_call": "float(np.sum(_oracle_solve_regularized_spectrum(system, 0.2)))",
        },
        # --- Boundary: zero penalty leaves the pure non-negative least squares ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-4, 1.0, 40)
counts = np.full(40, 5.0)
values = 100.0 - 0.5 * np.exp(-120.0 * times)
grid = np.geomspace(1.0, 1.0e4, 30)
w = np.sqrt(counts)
system = np.column_stack([np.exp(-np.outer(times, grid)) * w[:, None], -(values - values[-1]) * w])
""",
            "call": "float(np.sum(solve_regularized_spectrum(system, 0.0)))",
            "gold_call": "float(np.sum(_oracle_solve_regularized_spectrum(system, 0.0)))",
        },
        # --- Edge: a rising transient has no non-negative decomposition ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-4, 1.0, 30)
counts = np.full(30, 8.0)
values = 150.0 + 0.2 * np.exp(-70.0 * times)
grid = np.geomspace(1.0, 1.0e4, 25)
w = np.sqrt(counts)
system = np.column_stack([np.exp(-np.outer(times, grid)) * w[:, None], -(values - values[-1]) * w])
""",
            "call": "float(np.sum(solve_regularized_spectrum(system, 0.01)))",
            "gold_call": "float(np.sum(_oracle_solve_regularized_spectrum(system, 0.01)))",
        },
        # --- Invalid: negative regularization parameter ---
        {
            "setup": """import numpy as np
system = np.ones((6, 5))
def run_model():
    try:
        solve_regularized_spectrum(system, -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_regularized_spectrum(system, -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a one-column system carries no kernel ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_regularized_spectrum(np.ones((6, 1)), 0.01)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_regularized_spectrum(np.ones((6, 1)), 0.01)
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
