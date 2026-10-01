"""
Step 03 - Long-wavelength fermionic exchange constant of two 1s excitons.

Two excitons interact even though each is neutral, because their electrons, and likewise their holes, are identical fermions. This exchange interaction is the dominant exciton-exciton interaction in transition-metal dichalcogenide monolayers. At Hartree-Fock level, and in the long-wavelength limit where every centre-of-mass and transferred momentum is set to zero, it reduces to a single constant W, defined so that a 1s exciton immersed in a gas of identical 1s excitons of areal density n is shifted by W n.

With the momentum-space 1s amplitude phi(k), the Fourier transform of psi(r) from step 02 and so normalised that the integral of phi(k)^2 d^2k/(2 pi)^2 is one, and the repulsive screened interaction V(q) of step 01, the constant is W = 2 times the double integral of d^2k d^2k'/(2 pi)^4 V(|k - k'|) [phi(k') - phi(k)] phi(k) phi(k')^2. For an unscreened two-dimensional Coulomb interaction this reproduces the classic value close to 6.06 E_b lambda^2, where E_b is the binding energy and lambda the two-dimensional Bohr radius; for a screened sheet W has to be evaluated numerically. The logarithmic short-range behaviour of V inside the double integral is the main numerical hazard, and W is needed to about one part in 1e8.

Returns
-------
float, the long-wavelength exchange constant W in eV nm^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def exchange_constant(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, kappa: float, r0: float) -> float:
    '''Long-wavelength exchange constant W of two identical 1s excitons.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of finite positive Gaussian exponents a_i, nm^-2.
    coefficients : array_like
        Coefficients c_i of the real amplitude psi(r) = sum_i c_i exp(-a_i r^2),
        same length as exponents, taken as given (not renormalised).
    kappa : float
        Mean dielectric constant of the surrounding media, finite and > 0.
    r0 : float
        Screening length of the sheet in nm, finite and >= 0.

    Returns
    -------
    W : float
        The exchange constant in eV nm^2, defined so that a gas of density n of
        identical 1s excitons shifts each of them by W n.

    Raises
    ------
    ValueError
        If exponents and coefficients are not non-empty one-dimensional arrays of
        equal length with finite entries and positive exponents, or if kappa or
        r0 is invalid as in step 01.
    '''
    return W

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_exchange_constant(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, kappa: float, r0: float) -> float:
    """Long-wavelength fermionic exchange constant W of two identical 1s excitons, eV nm^2."""
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if a.ndim != 1 or a.size == 0 or c.shape != a.shape:
        raise ValueError("exponents and coefficients must be non-empty 1D arrays of equal length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))) or np.any(a <= 0.0):
        raise ValueError("exponents must be finite and positive and coefficients finite")
    # phi(k) = sum_i w_i exp(-t_i k^2) is the Fourier amplitude of psi(r) = sum_i c_i exp(-a_i r^2)
    w = c * np.pi / a
    t = 1.0 / (4.0 * a)
    b2 = (t[:, None] + t[None, :]).ravel()
    f2 = (w[:, None] * w[None, :]).ravel() / (4.0 * np.pi * b2)
    e2 = 1.0 / (4.0 * b2)
    b3 = (t[:, None, None] + t[None, :, None] + t[None, None, :]).ravel()
    f3 = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel() / (4.0 * np.pi * b3)
    e3 = 1.0 / (4.0 * b3)
    # real-space form: W = 2 [int V psi F3 - int V F2^2], F_n the inverse transform of phi^n
    g1 = _oracle_screened_gaussian_moments((a[:, None] + e3[None, :]).ravel(), kappa, r0)
    t1 = float(np.sum(c[:, None] * f3[None, :] * g1.reshape(a.size, e3.size)))
    g2 = _oracle_screened_gaussian_moments((e2[:, None] + e2[None, :]).ravel(), kappa, r0)
    t2 = float(np.sum(f2[:, None] * f2[None, :] * g2.reshape(e2.size, e2.size)))
    return 2.0 * (t1 - t2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: MoS2 ground state in a five-function basis ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([-0.00484781934334, 0.131037470592, 0.300175291225, 0.0798788920878, 0.0234948262371])\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'),
            "call": 'exchange_constant(exponents, coefficients, kappa, r0)',
            "gold_call": '_oracle_exchange_constant(exponents, coefficients, kappa, r0)',
            "tol": 1e-07,
        },
        # --- Boundary: two-dimensional hydrogen state with r0 = 0 ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([0.00192612859442, -0.0124482428401, 0.277816293025, 0.594038400312, 0.525864373425])\n'
                      'kappa = 4.5\n'
                      'r0 = 0.0\n'),
            "call": 'exchange_constant(exponents, coefficients, kappa, r0)',
            "gold_call": '_oracle_exchange_constant(exponents, coefficients, kappa, r0)',
            "tol": 1e-07,
        },
        # --- Edge: a single normalised Gaussian trial state ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.3])\n'
                      'coefficients = np.array([np.sqrt(0.6 / np.pi)])\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'),
            "call": 'exchange_constant(exponents, coefficients, kappa, r0)',
            "gold_call": '_oracle_exchange_constant(exponents, coefficients, kappa, r0)',
            "tol": 1e-07,
        },
        # --- Normal: compact exciton of heavy carriers under weak screening ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.05, 0.4, 3.2, 25.6])\n'
                      'coefficients = np.array([-0.00951763743232, 0.169820649119, 1.06859210545, 0.17307684644])\n'
                      'kappa = 1.0\n'
                      'r0 = 2.0\n'),
            "call": 'exchange_constant(exponents, coefficients, kappa, r0)',
            "gold_call": '_oracle_exchange_constant(exponents, coefficients, kappa, r0)',
            "tol": 1e-07,
        },
    ]
