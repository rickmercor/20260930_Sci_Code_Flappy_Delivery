"""
Step 01: Exchange energy per electron of the spin-polarized soft-Coulomb uniform gas.

Exchange energy per electron of the fully spin-polarized uniform one-dimensional electron gas with a soft-Coulomb pair interaction.

In one dimension the bare Coulomb interaction diverges at contact, so model electrons interact through the softened
pair potential w(u) = 1 / sqrt(u^2 + b^2), where u is the separation in bohr and b > 0 is the softening length. A
local spin-density approximation for such electrons is built from the homogeneous gas of the same interaction: the
exchange energy of an inhomogeneous spin density is the integral of the density times the exchange energy per electron
of the uniform gas evaluated at the local spin density.

This step supplies that ingredient for a single spin channel. For a homogeneous one-dimensional gas in which every
electron carries the same spin and the linear density is n electrons per bohr, the exchange energy per electron
eps_x(n) is the Hartree-Fock exchange energy of the ideal Fermi gas, divided by the number of electrons, with the pair
interaction w. It is negative for every n > 0, vanishes as n goes to zero, and is expressed in hartree. All quantities
are in atomic units.

Returns
-------
numpy.ndarray with the shape of density, exchange energy per electron eps_x(n) in hartree of the spin-polarized uniform 1D soft-Coulomb gas
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polarized_exchange_per_electron(density: "np.ndarray", softening: float) -> "np.ndarray":
    '''Exchange energy per electron of the spin-polarized uniform 1D gas with pair interaction 1/sqrt(u^2 + b^2).

    Parameters
    ----------
    density : np.ndarray
        Array of any shape of non-negative linear densities n of one spin channel, in electrons per bohr.
    softening : float
        Softening length b > 0 of the pair interaction, in bohr.

    Returns
    -------
    result : np.ndarray
        Array of the same shape as density holding eps_x(n) in hartree, with eps_x(0) = 0, accurate to within 1e-10
        hartree for every density.

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


def _integral_t_k0(y: "np.ndarray") -> "np.ndarray":
    """Integral of t K_0(t) from 0 to y, elementwise for y >= 0, equal to 1 - y K_1(y)."""
    import numpy as np
    from scipy.special import k1
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 1e-2
    ys = np.where(y[small] > 0.0, y[small], 1.0)
    lg = np.log(ys / 2.0)
    series = -(ys ** 2 / 2.0) * (lg + np.euler_gamma - 0.5) - (ys ** 4 / 16.0) * (lg + np.euler_gamma - 1.25)
    out[small] = np.where(y[small] > 0.0, series, 0.0)
    out[~small] = 1.0 - y[~small] * k1(y[~small])
    return out


def _oracle_polarized_exchange_per_electron(density: "np.ndarray", softening: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n < 0.0):
        raise ValueError("densities must be finite and non-negative")
    if not softening > 0.0:
        raise ValueError("softening must be strictly positive")
    b = float(softening)
    # With k_F = pi n for one spin channel, eps_x = -I(pi n b) / (2 pi^2 b^2 n), where
    # I(q) = integral over the real line of sin^2(q t) / (t^2 sqrt(t^2 + 1)) dt = 2 q F(2q) - G(2q),
    # F(y) = integral_0^y K_0 and G(y) = integral_0^y t K_0(t) dt, because I''(q) = 4 K_0(2q) and I(0) = I'(0) = 0.
    q = np.pi * n * b
    integral = 2.0 * q * _integral_k0(2.0 * q) - _integral_t_k0(2.0 * q)
    out = np.zeros_like(n)
    pos = n > 0.0
    out[pos] = -integral[pos] / (2.0 * np.pi ** 2 * b ** 2 * n[pos])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: densities typical of a one-electron molecular density, unit softening ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([0.02, 0.1, 0.25, 0.5, 0.8])\n",
            "call": "polarized_exchange_per_electron(n.copy(), 1.0)",
            "gold_call": "_oracle_polarized_exchange_per_electron(n.copy(), 1.0)",
            "tol": 1e-9,
        },
        # --- Normal: a two-dimensional array with a softening shorter than one bohr ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([[0.05, 0.3, 1.1], [2.0, 0.7, 0.012]])\n",
            "call": "polarized_exchange_per_electron(n.copy(), 0.45)",
            "gold_call": "_oracle_polarized_exchange_per_electron(n.copy(), 0.45)",
            "tol": 1e-9,
        },
        # --- Boundary: exact zero density and very dilute densities where the logarithmic regime dominates ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([0.0, 1e-9, 1e-6, 3e-4, 2e-3])\n",
            "call": "polarized_exchange_per_electron(n.copy(), 1.0)",
            "gold_call": "_oracle_polarized_exchange_per_electron(n.copy(), 1.0)",
            "tol": 1e-10,
        },
        # --- Edge: dense gas with a long softening ---
        {
            "setup": "import numpy as np\n"
                     "n = np.array([5.0, 12.0, 40.0])\n",
            "call": "polarized_exchange_per_electron(n.copy(), 2.3)",
            "gold_call": "_oracle_polarized_exchange_per_electron(n.copy(), 2.3)",
            "tol": 1e-9,
        },
        # --- Normal: a smooth grid density profile evaluated pointwise ---
        {
            "setup": "import numpy as np\n"
                     "x = np.linspace(-6.0, 6.0, 25)\n"
                     "n = 0.6 * np.exp(-0.7 * (x - 0.4) ** 2) + 0.25 * np.exp(-1.3 * (x + 2.1) ** 2)\n",
            "call": "polarized_exchange_per_electron(n.copy(), 1.0)",
            "gold_call": "_oracle_polarized_exchange_per_electron(n.copy(), 1.0)",
            "tol": 1e-9,
        },
        # --- Error: a negative density must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.2, -0.01]), 1.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(polarized_exchange_per_electron)",
            "gold_call": "_probe(_oracle_polarized_exchange_per_electron)",
        },
    ]
