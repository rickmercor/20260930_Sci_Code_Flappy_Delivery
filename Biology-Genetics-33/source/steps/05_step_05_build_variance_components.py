"""
Assemble the two trait-space covariance components of the mixed model from a step eigenvalue profile for each, with the family-level component expressed in a rotated eigenbasis.

A variance component whose eigenvalues take one positive value on a fraction of the trait directions and vanish on the rest is the simplest spectrum that carries a genuine null space. Placing the two components in different eigenbases is what makes their product depend on more than their spectra.

Returns
-------
np.ndarray of shape (2, n_traits, n_traits), float: the family-level and individual-level covariance matrices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_variance_components(n_traits: int,
                              tau_genetic: float,
                              rho_genetic: float,
                              tau_residual: float,
                              rho_residual: float,
                              rotation_angle: float) -> np.ndarray:
    """Assemble the family-level and individual-level trait covariance matrices.

    Each component has a single positive eigenvalue carried by the leading
    ``ceil(n_traits * rho)`` directions of its own eigenbasis and zero on the
    rest. The individual-level eigenbasis is the coordinate basis. The
    family-level eigenbasis is obtained from it by rotating each coordinate pair
    ``(i, n_traits - 1 - i)`` through ``rotation_angle``, the first coordinate of
    the pair being mapped to a combination whose own-coordinate weight is the
    cosine of the angle.

    Parameters
    ----------
    n_traits : int
        Number of measured traits; an integer of at least two.
    tau_genetic : float
        Positive nonzero eigenvalue of the family-level component.
    rho_genetic : float
        Fraction of trait directions carrying that eigenvalue, in ``(0, 1]``.
    tau_residual : float
        Positive nonzero eigenvalue of the individual-level component.
    rho_residual : float
        Fraction of trait directions carrying that eigenvalue, in ``(0, 1]``.
    rotation_angle : float
        Angle in radians between the two eigenbases.

    Returns
    -------
    components : np.ndarray
        Array of shape ``(2, n_traits, n_traits)``. Entry ``0`` is the
        family-level covariance matrix and entry ``1`` the individual-level
        covariance matrix.

    Raises
    ------
    ValueError
        If ``n_traits`` is not an integer value of at least two, if either
        ``tau`` is not a finite number greater than zero, if either ``rho`` is
        not a finite number in ``(0, 1]``, or if ``rotation_angle`` is not
        finite.
    """
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_variance_components(n_traits: int,
                                      tau_genetic: float,
                                      rho_genetic: float,
                                      tau_residual: float,
                                      rho_residual: float,
                                      rotation_angle: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    if not _is_number(n_traits) or float(n_traits) != float(int(n_traits)) or int(n_traits) < 2:
        raise ValueError("n_traits must be an integer value of at least two")
    for name, value in (("tau_genetic", tau_genetic), ("tau_residual", tau_residual)):
        if not _is_number(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number greater than zero")
    for name, value in (("rho_genetic", rho_genetic), ("rho_residual", rho_residual)):
        if not _is_number(value) or not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not _is_number(rotation_angle):
        raise ValueError("rotation_angle must be a finite number")

    n_traits = int(n_traits)
    angle = float(rotation_angle)

    # Step eigenvalue profiles: one positive level on the leading directions.
    genetic_spectrum = np.zeros(n_traits, dtype=float)
    genetic_spectrum[:int(np.ceil(n_traits * float(rho_genetic)))] = float(tau_genetic)
    residual_spectrum = np.zeros(n_traits, dtype=float)
    residual_spectrum[:int(np.ceil(n_traits * float(rho_residual)))] = float(tau_residual)

    # Eigenbasis of the family-level component: a plane rotation inside every
    # pair of mirrored coordinates, leaving a central coordinate fixed when the
    # number of traits is odd.
    basis = np.eye(n_traits, dtype=float)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    for low in range(n_traits // 2):
        high = n_traits - 1 - low
        basis[low, low] = cosine
        basis[low, high] = -sine
        basis[high, low] = sine
        basis[high, high] = cosine

    genetic = basis @ (genetic_spectrum[:, None] * basis.T)
    residual = np.diag(residual_spectrum)

    components = np.stack([genetic, residual])
    # Symmetrise to remove the asymmetry that floating point accumulation leaves.
    return 0.5 * (components + np.transpose(components, (0, 2, 1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the testbed spectra on a reduced trait count (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_variance_components(12, 1.4, 0.35, 0.6, 0.8, np.pi / 5.0), 1.0)",
            "gold_call": "sig(_oracle_build_variance_components(12, 1.4, 0.35, 0.6, 0.8, np.pi / 5.0), 1.0)",
        },
        # --- Valid: a quarter turn, the most adversarial alignment of the two
        #     eigenbases ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_variance_components(10, 2.0, 0.5, 0.75, 0.9, np.pi / 2.0), 1.0)",
            "gold_call": "sig(_oracle_build_variance_components(10, 2.0, 0.5, 0.75, 0.9, np.pi / 2.0), 1.0)",
        },
        # --- Boundary: a vanishing angle, where the two eigenbases coincide and
        #     both components are diagonal ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_variance_components(9, 1.4, 0.35, 0.6, 0.8, 0.0), 1.0)",
            "gold_call": "sig(_oracle_build_variance_components(9, 1.4, 0.35, 0.6, 0.8, 0.0), 1.0)",
        },
        # --- Edge: an odd trait count, where the central coordinate is not rotated,
        #     together with a fraction whose product with the trait count is not
        #     an integer ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_variance_components(11, 0.9, 0.37, 1.3, 0.62, -0.85), 1.0)",
            "gold_call": "sig(_oracle_build_variance_components(11, 0.9, 0.37, 1.3, 0.62, -0.85), 1.0)",
        },
        # --- Edge: a fully occupied genetic spectrum, for which the rotation leaves
        #     the component proportional to the identity ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_variance_components(8, 1.25, 1.0, 0.5, 0.5, 1.1), 1.0)",
            "gold_call": "sig(_oracle_build_variance_components(8, 1.25, 1.0, 0.5, 0.5, 1.1), 1.0)",
        },
        # --- Invalid: a genetic fraction outside the unit interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_variance_components(10, 1.4, 1.5, 0.6, 0.8, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_variance_components(10, 1.4, 1.5, 0.6, 0.8, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive eigenvalue ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_variance_components(10, 0.0, 0.35, 0.6, 0.8, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_variance_components(10, 0.0, 0.35, 0.6, 0.8, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
