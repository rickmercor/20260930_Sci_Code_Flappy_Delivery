"""
Step 04: Hartree plus local spin-density exchange energy on a grid.

Hartree plus local spin-density exchange energy of spin densities sampled on a uniform grid.

For electrons that interact through the soft-Coulomb pair potential w(u) = 1 / sqrt(u^2 + b^2), the classical Coulomb
(Hartree) energy of the total density n = n_up + n_down is J[n] = (1/2) double integral of n(x) n(x') w(x - x') and the
exchange-only local spin-density approximation is E_x[n_up, n_down] = sum over spins of the integral of n_s(x)
eps_x(n_s(x)), where eps_x is the exchange energy per electron of the fully spin-polarized uniform gas from step 01.
For an exact one-electron density J + E_x would vanish; with the local approximation it does not, and this residual is
the one-electron self-interaction error that drives the delocalization error of approximate functionals.

On the uniform grid x_j = x_0 + j*h_x both integrals are evaluated with the rectangle rule: J = (1/2) h_x^2
sum_i sum_j n_i n_j w((i - j) h_x), including the i = j terms with w(0) = 1/b, and E_x = h_x sum_j sum_s n_(s,j)
eps_x(n_(s,j)). The result is in hartree.

Returns
-------
float, rectangle-rule Hartree energy of the total density plus local spin-density exchange energy in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hartree_exchange_energy(density_up: "np.ndarray", density_down: "np.ndarray", spacing: float, softening: float) -> float:
    '''Rectangle-rule Hartree energy of the total density plus soft-Coulomb local spin-density exchange energy.

    Parameters
    ----------
    density_up : np.ndarray
        Shape (G,), non-negative spin-up density on a uniform grid, in electrons per bohr.
    density_down : np.ndarray
        Shape (G,), non-negative spin-down density on the same grid.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    softening : float
        Softening length b > 0 of the pair interaction, in bohr.

    Returns
    -------
    result : float
        J[n_up + n_down] + E_x[n_up, n_down] in hartree.

    Raises
    ------
    ValueError
        If the two densities are not one-dimensional arrays of equal length, if any density is negative or not finite,
        or if the spacing or softening is not strictly positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hartree_exchange_energy(density_up: "np.ndarray", density_down: "np.ndarray", spacing: float, softening: float) -> float:
    """Reference implementation."""
    import numpy as np
    nu = np.asarray(density_up, dtype=float)
    nd = np.asarray(density_down, dtype=float)
    if nu.ndim != 1 or nd.ndim != 1 or nu.size != nd.size:
        raise ValueError("densities must be one-dimensional arrays of equal length")
    if not (spacing > 0.0 and softening > 0.0):
        raise ValueError("spacing and softening must be positive")
    if not (np.all(np.isfinite(nu)) and np.all(np.isfinite(nd))) or np.any(nu < 0.0) or np.any(nd < 0.0):
        raise ValueError("densities must be finite and non-negative")
    idx = np.arange(nu.size)
    kernel = 1.0 / np.sqrt(((idx[:, None] - idx[None, :]) * spacing) ** 2 + softening ** 2)
    n = nu + nd
    hartree = 0.5 * spacing ** 2 * float(n @ (kernel @ n))
    exchange = spacing * float(np.sum(nu * _oracle_polarized_exchange_per_electron(nu, softening))
                               + np.sum(nd * _oracle_polarized_exchange_per_electron(nd, softening)))
    return float(hartree + exchange)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: a normalized one-electron spin-up density (self-interaction residual of a single electron) ---
        {
            "setup": "import numpy as np\n"
                     "h = 0.1\n"
                     "x = -15.0 + h * np.arange(301)\n"
                     "n = np.exp(-np.sqrt(x ** 2 + 1.0))\n"
                     "n = n / (h * n.sum())\n"
                     "z = np.zeros_like(n)\n",
            "call": "hartree_exchange_energy(n.copy(), z.copy(), h, 1.0)",
            "gold_call": "_oracle_hartree_exchange_energy(n.copy(), z.copy(), h, 1.0)",
            "tol": 1e-10,
        },
        # --- Normal: the same electron split evenly over both spin channels ---
        {
            "setup": "import numpy as np\n"
                     "h = 0.1\n"
                     "x = -15.0 + h * np.arange(301)\n"
                     "n = np.exp(-np.sqrt(x ** 2 + 1.0))\n"
                     "n = n / (h * n.sum())\n",
            "call": "hartree_exchange_energy(0.5 * n, 0.5 * n, h, 1.0)",
            "gold_call": "_oracle_hartree_exchange_energy(0.5 * n, 0.5 * n, h, 1.0)",
            "tol": 1e-10,
        },
        # --- Normal: a delocalized two-center density with unequal spin populations and a short softening ---
        {
            "setup": "import numpy as np\n"
                     "h = 0.05\n"
                     "x = -12.0 + h * np.arange(481)\n"
                     "nu = 0.8 * np.exp(-(x - 2.0) ** 2) + 0.3 * np.exp(-(x + 2.0) ** 2)\n"
                     "nd = 0.2 * np.exp(-0.5 * (x + 1.0) ** 2)\n",
            "call": "hartree_exchange_energy(nu.copy(), nd.copy(), h, 0.6)",
            "gold_call": "_oracle_hartree_exchange_energy(nu.copy(), nd.copy(), h, 0.6)",
            "tol": 1e-10,
        },
        # --- Boundary: an empty spin-down channel and a density that vanishes on most of the grid ---
        {
            "setup": "import numpy as np\n"
                     "h = 0.2\n"
                     "x = -6.0 + h * np.arange(61)\n"
                     "nu = np.where(np.abs(x) < 1.0, 0.5 * (1.0 - x ** 2), 0.0)\n"
                     "nd = np.zeros_like(nu)\n",
            "call": "hartree_exchange_energy(nu.copy(), nd.copy(), h, 1.4)",
            "gold_call": "_oracle_hartree_exchange_energy(nu.copy(), nd.copy(), h, 1.4)",
            "tol": 1e-10,
        },
        # --- Edge: a single occupied grid point, where only the contact term w(0) = 1/b contributes to J ---
        {
            "setup": "import numpy as np\n"
                     "nu = np.array([0.0, 0.0, 2.5, 0.0])\n"
                     "nd = np.array([0.0, 0.0, 1.5, 0.0])\n",
            "call": "hartree_exchange_energy(nu.copy(), nd.copy(), 0.4, 0.9)",
            "gold_call": "_oracle_hartree_exchange_energy(nu.copy(), nd.copy(), 0.4, 0.9)",
            "tol": 1e-10,
        },
        # --- Error: spin densities of different lengths must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.ones(5), np.ones(4), 0.1, 1.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(hartree_exchange_energy)",
            "gold_call": "_probe(_oracle_hartree_exchange_energy)",
        },
    ]
