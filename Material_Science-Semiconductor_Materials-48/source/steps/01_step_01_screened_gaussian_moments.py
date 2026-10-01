"""
Step 01 - Gaussian moments of the screened electron-hole interaction in a monolayer.

In an atomically thin semiconductor the Coulomb interaction between an electron and a hole is screened by the polarisable sheet itself as well as by the dielectric on either side of it. A widely used description is the Rytova-Keldysh form, whose two-dimensional Fourier transform is V(q) = e^2/(2 eps0 q (kappa + r0 q)). The environment enters through kappa, the mean dielectric constant of the media above and below the sheet, and the sheet enters through its screening length r0. At separations large compared with r0/kappa the interaction is the Coulomb law screened by kappa, while at short range it grows only logarithmically.

When the exciton is expanded in Gaussians exp(-a r^2), every potential matrix element is an integral of V against a single Gaussian exp(-s r^2), where s is the sum of two exponents. This step returns those Gaussian moments, U(s) = integral over the plane of V(r) exp(-s r^2), for the repulsive sign of V, in eV nm^2, with e^2/(4 pi eps0) = 1.439964 eV nm. Setting r0 = 0 must reproduce the unscreened sheet, the Coulomb interaction divided by kappa.

In a converged basis the exponents span many decades, so the moments have to stay accurate to about one part in 1e9 both for very diffuse Gaussians, where the Coulomb tail dominates, and for very compact ones, where the logarithmic core does.

Returns
-------
numpy.ndarray, the Gaussian moments U(s) of the screened interaction in eV nm^2, same shape as s
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def screened_gaussian_moments(s: npt.ArrayLike, kappa: float, r0: float) -> np.ndarray:
    '''Gaussian moments of the repulsive Rytova-Keldysh interaction.

    Parameters
    ----------
    s : array_like
        One-dimensional array of Gaussian exponents, each finite and > 0, in nm^-2.
    kappa : float
        Mean dielectric constant of the surrounding media, finite and > 0.
    r0 : float
        Screening length of the sheet in nm, finite and >= 0; r0 = 0 is the
        Coulomb limit.

    Returns
    -------
    moments : numpy.ndarray
        U(s) = integral over the plane of V(r) exp(-s r^2) for each s, in eV nm^2,
        where V(q) = e^2/(2 eps0 q (kappa + r0 q)) and e^2/(4 pi eps0) = 1.439964
        eV nm. Same shape as s.

    Raises
    ------
    ValueError
        If s is not a non-empty one-dimensional array of finite positive values,
        if kappa is not finite and positive, or if r0 is negative or not finite.
    '''
    return moments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_screened_gaussian_moments(s: npt.ArrayLike, kappa: float, r0: float) -> np.ndarray:
    """Gaussian moments of the repulsive screened interaction, int d^2r V(r) exp(-s r^2)."""
    import numpy as np
    from scipy.special import dawsn, expi
    e2 = 1.439964                      # e^2/(4 pi eps0), eV nm
    sv = np.asarray(s, dtype=float)
    if sv.ndim != 1 or sv.size == 0:
        raise ValueError("s must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sv)) or np.any(sv <= 0.0):
        raise ValueError("every s must be finite and positive")
    kappa = float(kappa)
    r0 = float(r0)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(r0) or r0 < 0.0:
        raise ValueError("r0 must be finite and non-negative")
    pref = 2.0 * np.pi * e2            # e^2/(2 eps0), eV nm
    if r0 == 0.0:
        return pref * np.sqrt(np.pi) / (2.0 * kappa * np.sqrt(sv))
    # momentum-space form: (pref / 2s) int_0^inf exp(-q^2/4s) / (kappa + r0 q) dq,
    # reduced to J(c) = int_0^inf exp(-u^2)/(u + c) du with c = kappa / (2 r0 sqrt(s))
    c = kappa / (2.0 * r0 * np.sqrt(sv))
    x = c * c
    jc = np.empty_like(c)
    low = x < 600.0
    jc[low] = np.sqrt(np.pi) * dawsn(c[low]) - 0.5 * np.exp(-x[low]) * expi(x[low])
    if np.any(~low):
        xh = x[~low]
        tail = np.zeros_like(xh)
        term = 1.0 / xh
        for n in range(1, 40):         # exp(-x) Ei(x) = sum_n n!/x^(n+1) for large x
            tail += term
            term = term * n / xh
        jc[~low] = np.sqrt(np.pi) * dawsn(c[~low]) - 0.5 * tail
    return pref / (2.0 * sv * r0) * jc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: exponents typical of a converged exciton basis, hBN-like environment ---
        {
            "setup": ('import numpy as np\n'
                      's = np.array([0.004, 0.03, 0.25, 2.0, 16.0, 128.0])\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'),
            "call": 'screened_gaussian_moments(s, kappa, r0)',
            "gold_call": '_oracle_screened_gaussian_moments(s, kappa, r0)',
            "tol": 3e-07,
        },
        # --- Boundary: r0 = 0, the unscreened-sheet Coulomb limit ---
        {
            "setup": ('import numpy as np\n'
                      's = np.array([0.01, 0.5, 20.0])\n'
                      'kappa = 2.45\n'
                      'r0 = 0.0\n'),
            "call": 'screened_gaussian_moments(s, kappa, r0)',
            "gold_call": '_oracle_screened_gaussian_moments(s, kappa, r0)',
            "tol": 4e-07,
        },
        # --- Edge: ten decades of exponent in a freestanding-like environment ---
        {
            "setup": ('import numpy as np\n'
                      's = np.array([1.0e-5, 1.0e-3, 1.0e3, 1.0e5])\n'
                      'kappa = 1.0\n'
                      'r0 = 6.0\n'),
            "call": 'screened_gaussian_moments(s, kappa, r0)',
            "gold_call": '_oracle_screened_gaussian_moments(s, kappa, r0)',
            "tol": 3e-05,
        },
        # --- Edge: a very short screening length, close to but not at the Coulomb limit ---
        {
            "setup": ('import numpy as np\n'
                      's = np.array([0.1, 1.0, 10.0])\n'
                      'kappa = 4.5\n'
                      'r0 = 0.05\n'),
            "call": 'screened_gaussian_moments(s, kappa, r0)',
            "gold_call": '_oracle_screened_gaussian_moments(s, kappa, r0)',
            "tol": 6e-08,
        },
    ]
