"""
Step 02: Local exchange potential of one spin channel.

Local exchange potential of one spin channel in the soft-Coulomb local spin-density approximation.

In a Kohn-Sham calculation with a local spin-density exchange functional E_x[n] = integral of n(x) eps_x(n(x)) dx for
a spin channel of density n, the exchange contribution to that channel's effective potential is the functional
derivative v_x(x) = d[n eps_x(n)]/dn evaluated at the local density n(x). Here eps_x is the exchange energy per electron
of the fully spin-polarized uniform one-dimensional gas whose electrons interact through w(u) = 1 / sqrt(u^2 + b^2),
as in the previous step.

The potential is negative, tends to zero as the density vanishes, and approaches a finite limit set by the softening at
high density because the softened interaction is bounded at contact. It is expressed in hartree for densities in
electrons per bohr.

Returns
-------
numpy.ndarray with the shape of density, exchange potential d[n eps_x(n)]/dn in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polarized_exchange_potential(density: "np.ndarray", softening: float) -> "np.ndarray":
    '''Exchange potential d[n eps_x(n)]/dn of one spin channel for the soft-Coulomb uniform-gas exchange.

    Parameters
    ----------
    density : np.ndarray
        Array of any shape of non-negative spin densities n, in electrons per bohr.
    softening : float
        Softening length b > 0 of the pair interaction 1/sqrt(u^2 + b^2), in bohr.

    Returns
    -------
    result : np.ndarray
        Array of the same shape as density holding v_x(n) in hartree, with v_x(0) = 0.

    Raises
    ------
    ValueError
        If any density is negative or not finite, or if the softening is not strictly positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import k0, k1, modstruve


def _integral_k0(y: "np.ndarray") -> "np.ndarray":
    """Integral of the modified Bessel function K_0 from 0 to y, elementwise for y >= 0."""
    import numpy as np
    from scipy.special import k0, k1, modstruve
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 1e-2
    large = y > 40.0
    mid = ~(small | large)
    ys = np.where(y[small] > 0.0, y[small], 1.0)
    lg = np.log(ys / 2.0)
    series = ys * (1.0 - np.euler_gamma - lg) + ys ** 3 / 12.0 * (-lg - np.euler_gamma + 4.0 / 3.0)
    out[small] = np.where(y[small] > 0.0, series, 0.0)
    ym = y[mid]
    out[mid] = 0.5 * np.pi * ym * (k0(ym) * modstruve(-1, ym) + k1(ym) * modstruve(0, ym))
    out[large] = 0.5 * np.pi
    return out


def _oracle_polarized_exchange_potential(density: "np.ndarray", softening: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n < 0.0):
        raise ValueError("densities must be finite and non-negative")
    if not softening > 0.0:
        raise ValueError("softening must be strictly positive")
    b = float(softening)
    # d[n eps_x]/dn = -F(2 pi n b) / (pi b) with F(y) = integral_0^y K_0
    return -_integral_k0(2.0 * np.pi * n * b) / (np.pi * b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: densities of a molecular one-electron density, unit softening ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([0.01, 0.07, 0.2, 0.45, 0.9])\n",
            "call": "polarized_exchange_potential(n.copy(), 1.0)",
            "gold_call": "_oracle_polarized_exchange_potential(n.copy(), 1.0)",
            "tol": 1e-10,
        },
        # --- Normal: potential on a grid profile with a short softening ---
        {
            "setup": "import numpy as np\n"
                     "x = np.linspace(-5.0, 5.0, 21)\n"
                     "n = 0.35 * np.exp(-np.abs(x - 1.0)) + 0.15 * np.exp(-0.5 * (x + 1.5) ** 2)\n",
            "call": "polarized_exchange_potential(n.copy(), 0.6)",
            "gold_call": "_oracle_polarized_exchange_potential(n.copy(), 0.6)",
            "tol": 1e-10,
        },
        # --- Boundary: zero and very small densities in the logarithmic regime ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([0.0, 1e-10, 1e-7, 4e-4, 1.5e-3])\n",
            "call": "polarized_exchange_potential(n.copy(), 1.0)",
            "gold_call": "_oracle_polarized_exchange_potential(n.copy(), 1.0)",
            "tol": 1e-10,
        },
        # --- Edge: high densities where the potential saturates, long softening ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([3.0, 8.0, 25.0, 120.0])\n",
            "call": "polarized_exchange_potential(n.copy(), 1.7)",
            "gold_call": "_oracle_polarized_exchange_potential(n.copy(), 1.7)",
            "tol": 1e-10,
        },
        # --- Error: a non-positive softening must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.1, 0.2]), 0.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(polarized_exchange_potential)",
            "gold_call": "_probe(_oracle_polarized_exchange_potential)",
        },
    ]
