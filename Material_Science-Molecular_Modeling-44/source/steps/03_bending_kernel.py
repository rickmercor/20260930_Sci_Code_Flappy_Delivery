"""
Build the angular coupling kernel of two successive bonds on a polar-angle grid: for each pair of polar angles (theta, theta_prime) measured from the force axis, the Boltzmann factor of the harmonic bending energy in the bond angle between the two bond vectors, integrated over their relative azimuthal angle on [0, 2 pi) (source, Sec. S2.1; the same kernel underlies the transfer-matrix elasticity of the deformable-bond chain). Use the midpoint rule with n_omega equally spaced azimuths. The bond angle is the angle between the bond vectors, and the bending energy is k_phi (phi - phi_e)^2 / 2 in units of k_B T, i.e. beta_kphi = k_phi / (k_B T).

In a chain with bending stiffness the orientations of neighbouring bonds are correlated through the bond angle, which depends on both polar angles and on the relative azimuth of the two bonds. Integrating the bending Boltzmann factor over the azimuth gives a kernel in the polar angles alone, the transfer operator of the chain's orientational statistics under a force that singles out one axis.

Returns
-------
ndarray of float64, shape (n_theta, n_theta): kernel[j, k] for polar angles theta[j] (later bond) and theta[k] (earlier bond).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bending_kernel(theta: "np.ndarray", beta_kphi: float, phi_e: float, n_omega: int = 512) -> "np.ndarray":
    """Build the angular coupling kernel of two successive bonds on a polar-angle grid: for each pair of polar angles (theta, theta_prime) measured from the force axis, the Boltzmann factor of the harmonic bending energy in the bond angle between the two bond vectors, integrated over their relative azimuthal angle on [0, 2 pi) (source, Sec. S2.1; the same kernel underlies the transfer-matrix elasticity of the deformable-bond chain). Use the midpoint rule with n_omega equally spaced azimuths. The bond angle is the angle between the bond vectors, and the bending energy is k_phi (phi - phi_e)^2 / 2 in units of k_B T, i.e. beta_kphi = k_phi / (k_B T).

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid in [0, pi] (radians), shape (n_theta,).
    beta_kphi : float
        Bending stiffness in units of k_B T per rad^2 (non-negative).
    phi_e : float
        Equilibrium bond angle between successive bond vectors in radians, in [0, pi].
    n_omega : int
        Number of midpoint azimuthal nodes (default 512).

    Returns
    -------
    kernel : np.ndarray
        Symmetric kernel of shape (n_theta, n_theta).

    Raises
    ------
    ValueError
        If theta is not a finite one-dimensional array inside [0, pi], beta_kphi is negative or not finite, phi_e is outside [0, pi], or n_omega is not a positive integer.
    """
    return kernel

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _oracle_bending_kernel(theta: "np.ndarray", beta_kphi: float, phi_e: float, n_omega: int = 512) -> "np.ndarray":
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 1 or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a one-dimensional finite array")
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        raise ValueError("theta must lie in [0, pi]")
    if not np.isfinite(beta_kphi) or beta_kphi < 0.0:
        raise ValueError("beta_kphi must be finite and non-negative")
    if not np.isfinite(phi_e) or phi_e < 0.0 or phi_e > np.pi:
        raise ValueError("phi_e must lie in [0, pi]")
    if not isinstance(n_omega, (int, np.integer)) or n_omega < 1:
        raise ValueError("n_omega must be a positive integer")
    omega = (np.arange(int(n_omega)) + 0.5) * (2.0 * np.pi / int(n_omega))   # midpoint rule, periodic
    ct, st = np.cos(theta), np.sin(theta)
    cos_phi = ct[:, None, None] * ct[None, :, None] + st[:, None, None] * st[None, :, None] * np.cos(omega)[None, None, :]
    phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
    return np.exp(-0.5 * beta_kphi * (phi - phi_e) ** 2).sum(axis=2) * (2.0 * np.pi / int(n_omega))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\n",
            "call": "bending_kernel(theta, beta_kphi, phi_e)",
            "gold_call": "_oracle_bending_kernel(theta, beta_kphi, phi_e)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_kphi, phi_e = 0.0, np.deg2rad(70.5)\n",
            "call": "bending_kernel(theta, beta_kphi, phi_e)",
            "gold_call": "_oracle_bending_kernel(theta, beta_kphi, phi_e)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le = 279.0, 2.15\nbeta_kphi, phi_e = 1.82e3 / np.pi ** 2, np.deg2rad(69.0)\npref1, pref2 = 1.18e12, 1.67e12\n",
            "call": "bending_kernel(theta, beta_kphi, phi_e)",
            "gold_call": "_oracle_bending_kernel(theta, beta_kphi, phi_e)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\ndef run_model():\n    try:\n        bending_kernel(np.array([0.5, 4.0]), beta_kphi, phi_e)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_bending_kernel(np.array([0.5, 4.0]), beta_kphi, phi_e)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
