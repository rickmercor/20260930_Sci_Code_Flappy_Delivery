"""
Step 01: Excited-state surface energy, gradient and Hessian.

Potential energy, gradient and Hessian of a two-dimensional anharmonic excited-state surface with a stretch-dependent bend.

Vibronic spectra of polyatomic molecules are shaped by the excited-state potential energy surface in the region the
nuclear wavepacket explores after a vertical transition. A minimal anharmonic model keeps one Morse stretch and one bend
whose harmonic force constant depends on the stretch: compressing the bond stiffens the bend and stretching it softens
the bend, so the curvature felt by the wavepacket changes along its path even though the minimum is harmonic.

Coordinates are mass-weighted with unit masses and hbar = 1. With stretch coordinate q1, bend coordinate q2, excited-state
minimum at (d, delta), x = q1 - d and y = q2 - delta, the surface is
    V(q1, q2) = D [1 - exp(-a x)]^2 + (1/2) omega_b^2 exp(-gamma x) y^2,
with well depth D = omega_e / (4 chi) and range parameter a = sqrt(2 omega_e chi), where omega_e is the harmonic stretch
frequency, chi the dimensionless anharmonicity, omega_b the bend frequency at the minimum and gamma the stretch-bend
coupling. The minimum energy is zero and the Hessian at the minimum is diag(omega_e^2, omega_b^2).

Returns
-------
numpy.ndarray of shape (N, 6), rows [V, dV/dq1, dV/dq2, d2V/dq1^2, d2V/dq1dq2, d2V/dq2^2] of the excited-state surface at the points
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def excited_surface_derivatives(points: "np.ndarray", displacement: float, surface_params: "np.ndarray") -> "np.ndarray":
    '''Potential energy, gradient and Hessian of the excited-state surface at a set of points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2), N >= 1, of mass-weighted coordinates (q1, q2).
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] with omega_e > 0, chi > 0, omega_b > 0, any real gamma and the bend
        coordinate delta of the minimum.

    Returns
    -------
    result : np.ndarray
        Array of shape (N, 6) whose rows are [V, dV/dq1, dV/dq2, d2V/dq1^2, d2V/dq1dq2, d2V/dq2^2] at the points.

    Raises
    ------
    ValueError
        If points is not a finite array of shape (N, 2) with N >= 1, if surface_params does not hold five finite numbers,
        or if omega_e, chi or omega_b is not strictly positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_excited_surface_derivatives(points: "np.ndarray", displacement: float, surface_params: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    pts = np.asarray(points, dtype=float)
    par = np.asarray(surface_params, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1 or not np.all(np.isfinite(pts)):
        raise ValueError("points must be a finite (N, 2) array with N >= 1")
    if par.size != 5 or not np.all(np.isfinite(par)):
        raise ValueError("surface_params must hold five finite numbers")
    omega_e, chi, omega_b, gamma, delta = par
    if omega_e <= 0.0 or chi <= 0.0 or omega_b <= 0.0:
        raise ValueError("omega_e, chi and omega_b must be strictly positive")
    depth = omega_e / (4.0 * chi)
    a = np.sqrt(2.0 * omega_e * chi)
    wb2 = omega_b * omega_b
    x = pts[:, 0] - float(displacement)
    y = pts[:, 1] - delta
    e = np.exp(-a * x)
    f = np.exp(-gamma * x)
    out = np.empty((pts.shape[0], 6))
    out[:, 0] = depth * (1.0 - e) ** 2 + 0.5 * wb2 * f * y * y
    out[:, 1] = 2.0 * depth * a * e * (1.0 - e) - 0.5 * gamma * wb2 * f * y * y
    out[:, 2] = wb2 * f * y
    out[:, 3] = 2.0 * depth * a * a * e * (2.0 * e - 1.0) + 0.5 * gamma * gamma * wb2 * f * y * y
    out[:, 4] = -gamma * wb2 * f * y
    out[:, 5] = wb2 * f
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: points around the Franck-Condon region and the minimum for the benchmark-like surface ---
        {
            "setup": "import numpy as np\n"
                     "pts = np.array([[0.0, 0.0], [2.2, 0.8], [1.1, -0.3], [4.0, 2.5]])\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n",
            "call": "excited_surface_derivatives(pts.copy(), 2.2, par.copy())",
            "gold_call": "_oracle_excited_surface_derivatives(pts.copy(), 2.2, par.copy())",
            "tol": 1e-10,
        },
        # --- Edge: far on the stretched side, beyond the Morse inflection, where the stretch curvature is negative ---
        {
            "setup": "import numpy as np\n"
                     "pts = np.array([[9.0, 0.2], [12.5, -1.4], [6.3, 3.0]])\n"
                     "par = np.array([1.1, 0.035, 0.6, -0.25, 0.3])\n",
            "call": "excited_surface_derivatives(pts.copy(), 1.5, par.copy())",
            "gold_call": "_oracle_excited_surface_derivatives(pts.copy(), 1.5, par.copy())",
            "tol": 1e-10,
        },
        # --- Boundary: a single point at the minimum, where the gradient vanishes and the Hessian is diagonal ---
        {
            "setup": "import numpy as np\n"
                     "pts = np.array([[-0.7, 1.9]])\n"
                     "par = np.array([0.6, 0.01, 0.35, 0.9, 1.9])\n",
            "call": "excited_surface_derivatives(pts.copy(), -0.7, par.copy())",
            "gold_call": "_oracle_excited_surface_derivatives(pts.copy(), -0.7, par.copy())",
            "tol": 1e-12,
        },
        # --- Normal: strongly compressed geometries with a strong stretch-bend coupling ---
        {
            "setup": "import numpy as np\n"
                     "q1 = np.linspace(-2.0, 1.0, 7)\n"
                     "pts = np.column_stack([q1, 0.5 * np.sin(q1)])\n"
                     "par = np.array([0.95, 0.025, 0.4, 0.55, -0.6])\n",
            "call": "excited_surface_derivatives(pts.copy(), 3.1, par.copy())",
            "gold_call": "_oracle_excited_surface_derivatives(pts.copy(), 3.1, par.copy())",
            "tol": 1e-9,
        },
        # --- Error: a non-positive anharmonicity must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([[0.0, 0.0]]), 1.0, np.array([0.85, 0.0, 0.45, 0.3, 0.8]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(excited_surface_derivatives)",
            "gold_call": "_probe(_oracle_excited_surface_derivatives)",
        },
    ]
