"""
Evaluate, elementwise, the flow-parameter-dependent energy-denominator weight that multiplies each two-electron-integral product in the static self-energy of the flow-regularised quasiparticle self-consistent second-order Green's-function method (the source method).

In quasiparticle self-consistent second-order Green's-function theory, each orbital pair (p, q)
is coupled to two-hole-one-particle and two-particle-one-hole configurations through energy
denominators, one associated with p and one with q. The unregularised weight built from these
denominators diverges as a denominator approaches zero, which destabilises the self-consistent
cycle. The source method replaces it by a weight that depends on a flow parameter s
(hartree^-2) and remains finite.

Returns
-------
np.ndarray with the broadcast shape of x and y: the flow-regularized weight in hartree^-1, equal to 0 where both denominators are 0 and everywhere when s is 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_flow_regularized_weight(x: "np.ndarray", y: "np.ndarray", s: float) -> "np.ndarray":
    '''Flow-regularised energy-denominator weight of the source method's static self-energy.

    The weight is the complete factor that depends on the energy denominators and on s
    and multiplies a product of two-electron integrals in the source method's
    spin-integrated (spatial-orbital) static self-energy. It is taken exactly as it
    appears there, with no numerical prefactor added or removed.

    Parameters
    ----------
    x : np.ndarray
        Energy denominators (hartree) evaluated at the quasiparticle energy of the row
        orbital p. Any shape broadcastable with ``y``.
    y : np.ndarray
        Energy denominators (hartree) evaluated at the quasiparticle energy of the column
        orbital q. Any shape broadcastable with ``x``.
    s : float
        Flow parameter in hartree^-2, finite and non-negative.

    Returns
    -------
    w : np.ndarray
        Float array of the broadcast shape of ``x`` and ``y`` containing the weight
        elementwise, in hartree^-1. Elements with x = y = 0 take the continuous limit,
        0, for every s; the whole array is 0 when s = 0. No element may be NaN or
        infinite for finite inputs.

    Raises
    ------
    ValueError
        If ``s`` is negative or not finite.
    '''
    return w

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_flow_regularized_weight(x: "np.ndarray", y: "np.ndarray", s: float) -> "np.ndarray":
    s = float(s)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("the flow parameter s must be finite and non-negative")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    z = s * (x * x + y * y)
    small = z < 1e-10
    z_safe = np.where(small, 1.0, z)
    # (1 - exp(-z)) / z evaluated without cancellation; equals 1 in the limit z -> 0
    phi = np.where(small, 1.0 - 0.5 * z, -np.expm1(-z_safe) / z_safe)
    return (x + y) * s * phi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    raise_setup = """import numpy as np
def raises_value_error(fn, x, y, s):
    try:
        fn(x, y, s)
    except ValueError:
        return 1
    return 0
"""
    return [
        # --- Normal: broadcast grid of mixed-sign denominators at an intermediate flow parameter ---
        {
            "setup": """import numpy as np
x = np.array([-2.3, -0.9, -0.15, 0.4, 1.7]).reshape(5, 1)
y = np.array([-1.8, -0.6, 0.05, 0.8, 2.5, 21.0])
s = 0.8
""",
            "call": "compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "gold_call": "_oracle_compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "tol": 1e-10,
        },
        # --- Edge: removable singularity x = y = 0, and x = -y pairs, inside a larger array ---
        {
            "setup": """import numpy as np
x = np.array([[0.0, 0.0, 0.3, -0.7], [1.2, 0.0, -0.45, 0.0]])
y = np.array([[0.0, 0.5, -0.3, 0.7], [-1.2, -2.0, 0.45, 0.0]])
s = 1.3
""",
            "call": "compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "gold_call": "_oracle_compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "tol": 1e-10,
        },
        # --- Boundary: s = 0 (no flow) ---
        {
            "setup": """import numpy as np
x = np.array([-1.5, 0.0, 0.25, 3.0])
y = np.array([0.5, 0.0, -2.0, 1.0])
s = 0.0
""",
            "call": "compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "gold_call": "_oracle_compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "tol": 1e-10,
        },
        # --- Boundary: very large flow parameter, including tiny denominators ---
        {
            "setup": """import numpy as np
x = np.array([-3.0, -0.02, 1e-4, 0.9, 5.0, 0.0])
y = np.array([-2.5, 0.03, 2e-4, -0.4, 6.0, 1e-3])
s = 1.0e5
""",
            "call": "compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "gold_call": "_oracle_compute_flow_regularized_weight(x.copy(), y.copy(), s)",
            "tol": 1e-10,
        },
        # --- Invalid: negative flow parameter ---
        {
            "setup": raise_setup + """
x = np.array([0.5, -0.5])
y = np.array([1.0, 2.0])
""",
            "call": "raises_value_error(compute_flow_regularized_weight, x.copy(), y.copy(), -0.1)",
            "gold_call": "raises_value_error(_oracle_compute_flow_regularized_weight, x.copy(), y.copy(), -0.1)",
        },
    ]
