"""
Tabulate the logarithm of the orientational coupling at every link of a possibly heterogeneous chain, on a Gauss-Legendre grid of polar angles measured from the force direction, after integrating out relative azimuth.

A harmonic bond-angle penalty couples neighbouring bond orientations, so the configurational integral can be propagated link by link. Evaluating the azimuthal integral and retaining each link kernel in the log domain avoids loss of the narrow angular channels produced by stiff bending potentials.

Returns
-------
np.ndarray: finite float array of shape (n_links, n_theta, n_theta) with the natural log of each azimuth-integrated bending kernel.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_bending_kernel(
    n_theta: int,
    n_omega: int,
    stiffness_reduced: "np.ndarray",
    equilibrium_angle_deg: "np.ndarray",
) -> "np.ndarray":
    """Return log azimuth-integrated bending kernels for successive links.

    Polar angles are the ``n_theta`` Gauss-Legendre nodes of ``[0, pi]``
    (``numpy.polynomial.legendre.leggauss`` mapped affinely, in its increasing
    order). The one-dimensional inputs contain one stiffness and equilibrium
    angle for each link between successive bonds. Entry ``[q, a, b]`` is the
    natural logarithm of the coupling at link ``q`` between the successor at
    polar angle ``theta_a`` and its predecessor at ``theta_b``. The coupling is
    the integral over relative azimuth ``omega`` in ``[0, 2 pi)`` of
    ``exp(-beta v_ben(phi))``, with ``beta v_ben = 0.5 * beta_k_phi[q] *
    (phi - phi_e[q])**2``, ``beta_k_phi = stiffness_reduced / pi**2``, and
    angles converted to radians. Use the ``n_omega``-point periodic trapezoidal
    rule at nodes ``2 pi k / n_omega`` and accumulate it with log-sum-exp so
    every returned entry stays finite even when the ordinary kernel underflows.

    Parameters
    ----------
    n_theta : int
        Number of polar-angle nodes, at least 2.
    n_omega : int
        Number of azimuthal nodes, at least 4.
    stiffness_reduced : np.ndarray
        Non-empty one-dimensional array of linkwise reduced bending stiffnesses
        ``beta k_phi pi**2``, all finite and nonnegative.
    equilibrium_angle_deg : np.ndarray
        One-dimensional array of the same shape containing linkwise equilibrium
        angles in [0, 180] degrees.

    Returns
    -------
    np.ndarray
        Finite float array of shape ``(n_links, n_theta, n_theta)`` containing
        natural logarithms of the azimuth-integrated kernels.

    Raises
    ------
    ValueError
        If ``n_theta`` is not an integer of at least 2 or ``n_omega`` not an
        integer of at least 4 (booleans are rejected), if
        either link-parameter input is not a non-empty one-dimensional array,
        their shapes differ, a stiffness is negative or non-finite, or an
        equilibrium angle is non-finite or lies outside [0, 180].
    """
    return kernel

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_bending_kernel(
    n_theta: int,
    n_omega: int,
    stiffness_reduced: "np.ndarray",
    equilibrium_angle_deg: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation using log-add-exp azimuthal accumulation."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_int(n_theta) and n_theta >= 2):
        raise ValueError("n_theta must be an integer of at least 2")
    if not (_is_int(n_omega) and n_omega >= 4):
        raise ValueError("n_omega must be an integer of at least 4")
    stiffness = np.asarray(stiffness_reduced, dtype=float)
    angle_deg = np.asarray(equilibrium_angle_deg, dtype=float)
    if stiffness.ndim != 1 or stiffness.size == 0 or not np.all(np.isfinite(stiffness)) or np.any(stiffness < 0):
        raise ValueError("stiffness_reduced must be a non-empty finite nonnegative 1D array")
    if angle_deg.shape != stiffness.shape or not np.all(np.isfinite(angle_deg)) or np.any((angle_deg < 0) | (angle_deg > 180)):
        raise ValueError("equilibrium_angle_deg must match stiffness_reduced and lie in [0, 180]")
    nodes, _ = np.polynomial.legendre.leggauss(int(n_theta))
    theta = 0.5 * np.pi * (nodes + 1.0)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    log_kernel = np.full((stiffness.size, int(n_theta), int(n_theta)), -np.inf)
    for k in range(int(n_omega)):
        omega = 2.0 * np.pi * k / int(n_omega)
        cos_phi = np.outer(sin_t, sin_t) * np.cos(omega) + np.outer(cos_t, cos_t)
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        terms = -0.5 * (stiffness / np.pi**2)[:, None, None] * (
            phi[None, :, :] - np.deg2rad(angle_deg)[:, None, None]
        ) ** 2
        log_kernel = np.logaddexp(log_kernel, terms)
    return log_kernel + np.log(2.0 * np.pi / int(n_omega))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(values):\n"
        "    arr = np.asarray(values, dtype=float)\n"
        "    flat = arr.ravel()\n"
        "    weight = 1.0 + 0.5 * np.cos(0.7 * np.arange(flat.size))\n"
        "    return float(arr.ndim + arr.shape[0] + flat.size + np.sum(weight * flat) + 0.001 * np.sum(flat ** 2))\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        build_bending_kernel({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_build_bending_kernel({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    invalid = [
        "1, 32, np.array([1820.0]), np.array([69.0])",
        "12, 3, np.array([1820.0]), np.array([69.0])",
        "12, 32, np.array([-5.0]), np.array([69.0])",
        "12, 32, np.array([1820.0]), np.array([181.0])",
        "12, 32, np.array([1820.0, 900.0]), np.array([69.0])",
        "12, 32, 1820.0, np.array([69.0])",
    ]
    return [
        {
            "setup": digest,
            "call": "_sig(build_bending_kernel(12, 48, np.array([1820.0, 2400.0, 900.0]), np.array([69.0, 82.0, 111.0])))",
            "gold_call": "_sig(_oracle_build_bending_kernel(12, 48, np.array([1820.0, 2400.0, 900.0]), np.array([69.0, 82.0, 111.0])))",
        },
        {
            "setup": digest,
            "call": "_sig(build_bending_kernel(20, 64, np.array([100.0]), np.array([111.0])))",
            "gold_call": "_sig(_oracle_build_bending_kernel(20, 64, np.array([100.0]), np.array([111.0])))",
        },
        {
            "setup": digest,
            "call": "_sig(build_bending_kernel(6, 16, np.array([0.0, 500.0]), np.array([0.0, 180.0])))",
            "gold_call": "_sig(_oracle_build_bending_kernel(6, 16, np.array([0.0, 500.0]), np.array([0.0, 180.0])))",
        },
        {
            "setup": digest,
            "call": "_sig(build_bending_kernel(9, 40, np.array([1.0e6, 1.0e4]), np.array([0.0, 137.0])))",
            "gold_call": "_sig(_oracle_build_bending_kernel(9, 40, np.array([1.0e6, 1.0e4]), np.array([0.0, 137.0])))",
        },
        {
            "setup": digest,
            "call": "_sig(build_bending_kernel(2, 4, np.array([5000.0, 1.0, 2500.0]), np.array([90.0, 45.0, 120.0])))",
            "gold_call": "_sig(_oracle_build_bending_kernel(2, 4, np.array([5000.0, 1.0, 2500.0]), np.array([90.0, 45.0, 120.0])))",
        },
    ] + [
        {"setup": raises.replace("{args}", args), "call": "_candidate()", "gold_call": "_reference()"}
        for args in invalid
    ]
