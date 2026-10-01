"""
Return umbrella-integration gradient estimates for one window or a broadcast-compatible batch of harmonically restrained windows on the model nucleation surface.

A restrained window adds a harmonic bias centred on a target value of the collective variable to the underlying free energy, samples the biased ensemble, and reports the average of the biasing force. In the strong-restraint (stiff-spring) approximation, the negative mean biasing force is assigned to the window centre as an estimate of the free-energy gradient there. The deterministic surrogate used here replaces the ensemble average by the stationary point of the biased surface, so the reported value equals the model derivative at that displaced point and approximates the derivative at the centre. A stiffer restraint reduces the displacement and therefore the finite-restraint error in this centre-assigned estimate.




This step replaces the biased molecular dynamics run by its deterministic stiff-spring limit on the coarse-grained nucleation free energy of classical-nucleation-theory form defined in the problem statement, with the three coefficients supplied as arguments, so the whole pipeline downstream can be reproduced exactly. The restrained mean is the stationary point of the biased surface, that is of the model free energy plus the harmonic restraint, and the window assigns the resulting negative mean biasing force to its centre. A window is mechanically stable, and the stationary point well defined, only while the curvature of the biased surface stays positive; a restraint too weak to hold the window against the curvature of the underlying surface has no stable solution and must be rejected. Locating the stationary point to the precision the rest of the pipeline needs requires an iterative solve rather than a closed form.

Returns
-------
float or np.ndarray: one gradient estimate for scalar inputs, otherwise the broadcast batch of estimates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_umbrella_mean_force(center: np.ndarray, kappa: np.ndarray, gamma: float = 2.5,
                                dmu: float = 0.4, offset: float = 8.0) -> np.ndarray:
    """Estimate gradients for one restrained window or a broadcast batch.

    Parameters
    ----------
    center : float or np.ndarray
        Scalar target or one-dimensional array of targets. Every target must
        lie above -offset.
    kappa : float or np.ndarray
        Positive scalar restraint constant or one-dimensional array
        broadcast-compatible with center.
    gamma : float
        Surface-term coefficient of the model free energy (gamma > 0).
    dmu : float
        Bulk-term coefficient of the model free energy.
    offset : float
        Regularising cluster-size offset of the surface term (offset > 0).

    Raises
    ------
    ValueError
        If center or kappa is non-numeric, non-finite, more than one
        dimensional, or not broadcast-compatible; if gamma, dmu, or offset is
        not a finite real scalar; if any kappa, gamma, or offset is
        non-positive; if any center is not above -offset; or if any restraint
        is too weak to yield a stable, converged stationary point.

    Returns
    -------
    gradient : float or np.ndarray
        A native Python float for scalar center and kappa, otherwise an array
        with the broadcast shape of center and kappa. Each entry is the
        umbrella-integration gradient estimate for that window.
    """
    return gradient  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_boolean(value) -> bool:
    """Return True for a boolean scalar or a numpy boolean."""
    import numpy as np

    return isinstance(value, (bool, np.bool_))


def _oracle_compute_umbrella_mean_force(center: np.ndarray, kappa: np.ndarray,
                                        gamma: float = 2.5, dmu: float = 0.4,
                                        offset: float = 8.0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("gamma", gamma), ("dmu", dmu), ("offset", offset)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    gamma = float(gamma)
    dmu = float(dmu)
    offset = float(offset)

    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    if offset <= 0.0:
        raise ValueError("offset must be > 0")

    def _vector_argument(name, value):
        if _is_boolean(value):
            raise ValueError(f"{name} must contain real numbers")
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain real numbers")
        if array.ndim > 1:
            raise ValueError(f"{name} must be scalar or one-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        return array

    center_array = _vector_argument("center", center)
    kappa_array = _vector_argument("kappa", kappa)
    try:
        center_array, kappa_array = np.broadcast_arrays(center_array, kappa_array)
    except ValueError:
        raise ValueError("center and kappa must be broadcast-compatible")
    if np.any(kappa_array <= 0.0):
        raise ValueError("every kappa must be > 0")
    if np.any(center_array <= -offset):
        raise ValueError("every center must lie above -offset")

    def _gradient_at(s):
        return (2.0 / 3.0) * gamma * (s + offset) ** (-1.0 / 3.0) - dmu

    def _curvature_at(s):
        return -(2.0 / 9.0) * gamma * (s + offset) ** (-4.0 / 3.0)

    result = np.empty(center_array.shape, dtype=float)
    for index in np.ndindex(center_array.shape):
        one_center = float(center_array[index])
        one_kappa = float(kappa_array[index])
        mean_cv = one_center
        for _ in range(100):
            denominator = _curvature_at(mean_cv) + one_kappa
            if denominator <= 0.0:
                raise ValueError("the restraint is too weak to stabilise a window")
            step = (_gradient_at(mean_cv) + one_kappa * (mean_cv - one_center)) / denominator
            mean_cv -= step
            if mean_cv <= -offset:
                raise ValueError("a restrained mean left the domain of the model surface")
            if abs(step) <= 1.0e-15 * max(1.0, abs(mean_cv)):
                break
        else:
            raise ValueError("a restrained mean did not converge")
        result[index] = -one_kappa * (mean_cv - one_center)

    if result.ndim == 0:
        return float(result)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: an initialisation window inside the liquid basin (normal scenario) ---
        {
            "setup": """import numpy as np
center = 1.6
kappa = 1.0
""",
            "call": "compute_umbrella_mean_force(center, kappa)",
            "gold_call": "_oracle_compute_umbrella_mean_force(center, kappa)",
        },
        # --- Valid: a window on the far side of the barrier, where the gradient is negative ---
        {
            "setup": """import numpy as np
center = 284.16
kappa = 1.0
""",
            "call": "compute_umbrella_mean_force(center, kappa)",
            "gold_call": "_oracle_compute_umbrella_mean_force(center, kappa)",
        },
        # --- Boundary: a window at the lower edge of the collective-variable range ---
        {
            "setup": """import numpy as np
center = 0.0
kappa = 1.0
""",
            "call": "compute_umbrella_mean_force(center, kappa)",
            "gold_call": "_oracle_compute_umbrella_mean_force(center, kappa)",
        },
        # --- Edge: a very stiff restraint, where the displacement collapses ---
        {
            "setup": """import numpy as np
center = 64.0
kappa = 1.0e3
""",
            "call": "compute_umbrella_mean_force(center, kappa)",
            "gold_call": "_oracle_compute_umbrella_mean_force(center, kappa)",
        },
        # --- Edge: a different surface parametrisation with a sharper surface term ---
        {
            "setup": """import numpy as np
center = 45.0
kappa = 2.5
gamma = 4.0
dmu = 0.55
offset = 3.0
""",
            "call": "compute_umbrella_mean_force(center, kappa, gamma, dmu, offset)",
            "gold_call": "_oracle_compute_umbrella_mean_force(center, kappa, gamma, dmu, offset)",
        },
        # --- Valid batch: one restraint constant broadcast over several windows ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 64.0, 142.5, 284.16])
kappa = 1.0
""",
            "call": "compute_umbrella_mean_force(centers, kappa)",
            "gold_call": "_oracle_compute_umbrella_mean_force(centers, kappa)",
        },
        # --- Valid batch: each window has its own mechanically stable restraint ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 45.0, 200.0, 287.0])
kappas = np.array([0.2, 2.5, 0.08, 5.0])
""",
            "call": "compute_umbrella_mean_force(centers, kappas)",
            "gold_call": "_oracle_compute_umbrella_mean_force(centers, kappas)",
        },
        # --- Boundary batch: a scalar centre broadcast against several stiffnesses ---
        {
            "setup": """import numpy as np
center = 64.0
kappas = np.array([0.05, 1.0, 1.0e3])
""",
            "call": "compute_umbrella_mean_force(center, kappas)",
            "gold_call": "_oracle_compute_umbrella_mean_force(center, kappas)",
        },
        # --- Invalid: a restraint too weak to hold the window against the surface curvature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_umbrella_mean_force(0.0, 0.01)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_umbrella_mean_force(0.0, 0.01)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive force constant ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_umbrella_mean_force(50.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_umbrella_mean_force(50.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a window centre outside the domain of the model surface ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_umbrella_mean_force(-8.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_umbrella_mean_force(-8.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: two one-dimensional batches cannot be broadcast together ---
        {
            "setup": """import numpy as np
centers = np.array([1.0, 2.0, 3.0])
kappas = np.array([1.0, 2.0])
def run_model():
    try:
        compute_umbrella_mean_force(centers, kappas)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_umbrella_mean_force(centers, kappas)
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
