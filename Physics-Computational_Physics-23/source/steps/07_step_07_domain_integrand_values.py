"""
Combine the spectral time derivative of the temperature with the heat source density into the density that the domain integral of the representation formula must carry at every Gauss node of the time step.

Writing the transient conduction balance as a Poisson problem at each instant moves the transient term into the domain integral of the representation formula, so the density of that integral is whatever that balance leaves on the right-hand side. The time derivative it needs at the Gauss nodes of a step comes from the spectral operator of the step rather than from a finite difference.

Returns
-------
np.ndarray, float, shape (p, n): the density of the domain integral at each sample point and each Gauss node time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def domain_integrand_values(operator: "np.ndarray", nodal_values: "np.ndarray",
                            initial_values: "np.ndarray", source_values: "np.ndarray",
                            conductivity: float) -> "np.ndarray":
    """Assemble the density of the domain integral at every Gauss node time.

    Parameters
    ----------
    operator : np.ndarray
        Float array of shape ``(p, p)`` mapping nodal temperatures, measured
        from the value carried into the step, onto nodal time derivatives.
    nodal_values : np.ndarray
        Float array of shape ``(p, n)`` holding the temperature at the ``n``
        sample points at each of the ``p`` Gauss node times.
    initial_values : np.ndarray
        Float array of shape ``(n,)`` holding the temperature at the ``n``
        sample points at the start of the step.
    source_values : np.ndarray
        Float array of shape ``(p, n)`` holding the heat source density at the
        ``n`` sample points at each of the ``p`` Gauss node times.
    conductivity : float
        Thermal conductivity of the layer (``conductivity > 0``).

    Returns
    -------
    density : np.ndarray
        Float array of shape ``(p, n)`` holding the density of the domain
        integral of the representation formula at each sample point and each
        Gauss node time.

    Raises
    ------
    ValueError
        If ``operator`` is not a finite real array of shape ``(p, p)`` with
        ``p >= 1``, if ``nodal_values`` or ``source_values`` is not a finite
        real array of shape ``(p, n)`` with ``n >= 1``, if ``initial_values``
        is not a finite real array of shape ``(n,)``, or if ``conductivity`` is
        not a finite real number > 0.

    """
    return density  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_domain_integrand_values(operator: "np.ndarray", nodal_values: "np.ndarray",
                                    initial_values: "np.ndarray", source_values: "np.ndarray",
                                    conductivity: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    matrix = np.asarray(operator, dtype=float)
    values = np.asarray(nodal_values, dtype=float)
    initial = np.asarray(initial_values, dtype=float)
    sources = np.asarray(source_values, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square array with at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator must be finite")
    order = matrix.shape[0]
    if values.ndim != 2 or values.shape[0] != order or values.shape[1] < 1:
        raise ValueError("nodal_values must have shape (p, n) with n >= 1")
    if sources.shape != values.shape:
        raise ValueError("source_values must have the same shape as nodal_values")
    if initial.shape != (values.shape[1],):
        raise ValueError("initial_values must have shape (n,)")
    for name, block in (("nodal_values", values), ("initial_values", initial),
                        ("source_values", sources)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
    if isinstance(conductivity, bool) or not isinstance(conductivity, (int, float, np.integer, np.floating)):
        raise ValueError("conductivity must be a real number")
    if not np.isfinite(float(conductivity)) or float(conductivity) <= 0.0:
        raise ValueError("conductivity must be a finite number > 0")

    # The spectral operator sees the increment of the temperature over the step,
    # not the temperature itself.
    rate = matrix @ (values - initial[None, :])

    return np.asarray(rate / float(conductivity) + sources, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: with the identity operator the density reduces to the
        # temperature increment over the conductivity plus the source, which is
        # a closed form obtained without the oracle.
        {
            "setup": """import numpy as np
op = np.eye(3)
vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
init = np.array([0.5, 1.5])
src = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
k = 4.0
""",
            "call": "float(np.sum(domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), k)))",
            "gold_call": "float(np.sum(_oracle_domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), k)))",
        },
        # --- Pinned: a constant temperature over the whole step has no time
        # derivative, so the density must collapse onto the source alone
        # whatever the operator.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
op = rng.normal(size=(4, 4))
init = np.array([2.0, -1.0, 0.25])
vals = np.tile(init, (4, 1))
src = rng.normal(size=(4, 3))
""",
            "call": ("float(np.sum(np.arange(1.0, 13.0)"
                     " * domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 3.0).ravel()))"),
            "gold_call": ("float(np.sum(np.arange(1.0, 13.0)"
                           " * _oracle_domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 3.0).ravel()))"),
        },
        # --- Valid: five Gauss nodes over a set of coating sample points ---
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
rng = np.random.default_rng(2026)
op = rng.normal(size=(5, 5))
vals = rng.normal(size=(5, 9)) + 3.0
init = rng.normal(size=9)
src = rng.normal(size=(5, 9))
""",
            "call": "float(pin(domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 2.0)))",
            "gold_call": "float(pin(_oracle_domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 2.0)))",
        },
        # --- Valid: the same data read with the substrate conductivity, which
        # scales the derivative contribution down by an order of magnitude ---
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
rng = np.random.default_rng(2026)
op = rng.normal(size=(5, 5))
vals = rng.normal(size=(5, 9)) + 3.0
init = rng.normal(size=9)
src = rng.normal(size=(5, 9))
""",
            "call": "float(pin(domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 15.0)))",
            "gold_call": "float(pin(_oracle_domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 15.0)))",
        },
        # --- Boundary: a single Gauss node and a single sample point ---
        {
            "setup": """import numpy as np
""",
            "call": ("float(1.0 + 1.0e6 * float(domain_integrand_values(np.array([[2.5]]),"
                     " np.array([[4.0]]), np.array([1.0]), np.array([[-0.5]]), 0.75)[0, 0]))"),
            "gold_call": ("float(1.0 + 1.0e6 * float(_oracle_domain_integrand_values(np.array([[2.5]]),"
                          " np.array([[4.0]]), np.array([1.0]), np.array([[-0.5]]), 0.75)[0, 0]))"),
        },
        # --- Edge: a very small conductivity, where the derivative term
        # dominates the source term ---
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
rng = np.random.default_rng(7)
op = rng.normal(size=(3, 3))
vals = rng.normal(size=(3, 4))
init = rng.normal(size=4)
src = rng.normal(size=(3, 4))
""",
            "call": "float(pin(domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 1.0e-3)))",
            "gold_call": "float(pin(_oracle_domain_integrand_values(op.copy(), vals.copy(), init.copy(), src.copy(), 1.0e-3)))",
        },
        # --- Invalid: the initial values do not match the sample count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        domain_integrand_values(np.eye(2), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3)), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_domain_integrand_values(np.eye(2), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3)), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative conductivity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        domain_integrand_values(np.eye(2), np.zeros((2, 3)), np.zeros(3), np.zeros((2, 3)), -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_domain_integrand_values(np.eye(2), np.zeros((2, 3)), np.zeros(3), np.zeros((2, 3)), -1.0)
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
