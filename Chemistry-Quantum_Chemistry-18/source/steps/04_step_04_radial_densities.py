"""
Step 04: Exact densities at a given radius. On-top two-electron density, one-electron density and reduced on-top density at distance r from the trap centre for the pair function Psi.

For a singlet with normalized spatial part Psi, the spin-summed one-electron density is
rho1(r_vec) = 2 * integral of |Psi(r_vec, r2_vec)|^2 over r2_vec, and the on-top two-electron density is the pair density at
coincident positions, Phi(r_vec) = rho2(r_vec, r_vec) = |Psi(r_vec, r_vec)|^2. Both depend only on r = |r_vec| here. Their
combination phi(r) = 4 Phi(r) / rho1(r)^2 is the reduced on-top density; it equals one for a singlet built from a single
doubly occupied orbital and falls below one where correlation keeps the electrons apart.

At coincidence r12 = 0, so the on-top density needs no integration. For rho1, measuring the second position from the first,
r2_vec = r_vec + u_vec, makes the correlation factor depend on u alone, and the angular integral of the Gaussian factor over
the direction of u_vec can be done in closed form, which leaves a single radial quadrature in u.

Returns
-------
numpy.ndarray [Phi(r), rho1(r), phi(r)]: on-top, one-electron and reduced on-top densities at radius r
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_densities(omega: float, c: float, r: float) -> "np.ndarray":
    '''Phi(r), rho1(r) and phi(r) = 4 Phi(r) / rho1(r)^2 for the normalized pair function Psi.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.
    r : float
        Distance from the trap centre in bohr, 0 <= r <= 5 omega^(-1/2).

    Returns
    -------
    result : np.ndarray
        Float array [Phi(r) in bohr^-6, rho1(r) in bohr^-3, phi(r)], each accurate to 1e-10 relative.

    Raises
    ------
    ValueError
        If r is negative.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad


def _rho1_integrand(u, omega, c, r):
    """Radial integrand of rho1(r) after the closed-form angular integration, without the prefactor 2 C^2."""
    x = 2.0 * omega * r * u
    if x < 1e-6:
        gauss = np.exp(-omega * (u * u + 2.0 * r * r)) * (1.0 + x * x / 6.0)
    else:
        gauss = np.exp(-omega * r * r) * (np.exp(-omega * (u - r) ** 2) - np.exp(-omega * (u + r) ** 2)) / (2.0 * x)
    return (1.0 + 0.5 * u + c * u * u) ** 2 * gauss * 4.0 * np.pi * u * u


def _oracle_radial_densities(omega: float, c: float, r: float) -> "np.ndarray":
    """Reference implementation."""
    r = float(r)
    if r < 0.0:
        raise ValueError("r must be non-negative")
    norm = _oracle_pair_normalization(omega, c)
    omega = float(omega)
    c = float(c)
    ontop = norm ** 2 * np.exp(-2.0 * omega * r * r)

    radial = quad(_rho1_integrand, 0.0, np.inf, args=(omega, c, r), epsabs=0.0, epsrel=1e-12, limit=400)[0]
    density = 2.0 * norm ** 2 * radial
    return np.array([ontop, density, 4.0 * ontop / density ** 2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: exact omega = 1/10 ground state near its radial density maximum ---
        {
            "setup": "import numpy as np\n",
            "call": "radial_densities(0.1, 0.05, 3.96)",
            "gold_call": "_oracle_radial_densities(0.1, 0.05, 3.96)",
            "tol": 1e-10,
        },
        # --- Normal: exact omega = 1/2 ground state ---
        {
            "setup": "import numpy as np\n",
            "call": "radial_densities(0.5, 0.0, 1.3)",
            "gold_call": "_oracle_radial_densities(0.5, 0.0, 1.3)",
            "tol": 1e-10,
        },
        # --- Boundary: the trap centre ---
        {
            "setup": "import numpy as np\n",
            "call": "radial_densities(0.1, 0.05, 0.0)",
            "gold_call": "_oracle_radial_densities(0.1, 0.05, 0.0)",
            "tol": 1e-10,
        },
        # --- Edge: far tail of a tight trap, graded on logarithms so that zero cannot pass for a small value ---
        {
            "setup": "import numpy as np\ndef logged(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.log(v)\n",
            "call": "logged(radial_densities(2.0, 0.3, 2.5))",
            "gold_call": "logged(_oracle_radial_densities(2.0, 0.3, 2.5))",
            "tol": 1e-9,
        },
        # --- Edge: the same tail one step further out, where the on-top density is smaller again ---
        {
            "setup": "import numpy as np\ndef logged(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.log(v)\n",
            "call": "logged(radial_densities(2.0, 0.3, 3.2))",
            "gold_call": "logged(_oracle_radial_densities(2.0, 0.3, 3.2))",
            "tol": 1e-9,
        },
        # --- Error: a negative radius must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.1, 0.05, -1.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(radial_densities)",
            "gold_call": "_probe(_oracle_radial_densities)",
        },
    ]
