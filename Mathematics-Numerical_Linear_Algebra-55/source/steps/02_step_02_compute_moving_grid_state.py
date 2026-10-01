"""
Evaluate the translating-grid velocity and the three stretched physical node vectors in the fixed order used by all later steps.

Each component grid has uniform computational nodes $\\xi_j=j/(N_\\kappa-1)$ and stretching $\\psi_\\sigma(\\xi)=\\xi-(\\sigma/\\pi)\\sin(\\pi\\xi)$. The physical maps are $x_L=-1+0.9\\psi_{\\sigma_L}(\\xi_L)$, $x_M=-0.35+A\\sin(2\\pi t/T_m)+0.7\\psi_{\\sigma_M}(\\xi_M)$, and $x_R=0.1+0.9\\psi_{\\sigma_R}(\\xi_R)$. Only the middle grid moves, with velocity $x_\\tau=(2\\pi A/T_m)\\cos(2\\pi t/T_m)$.



The result stores the scalar velocity first, followed by the left, middle, and right node vectors. Every grid count must be at least eighteen, every $|\\sigma_\\kappa|<1$, and the period must be positive.

Returns
-------
np.ndarray of length 1 + counts.sum(), ordered as velocity, x_L, x_M, x_R
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_moving_grid_state(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    amplitude: float = 0.1,
    period: float = 1.0,
) -> np.ndarray:
    r"""Return $x_\tau(t)$ and the stretched physical nodes of all three grids.

    Parameters
    ----------
    t : float
        Evaluation time.
    counts : np.ndarray
        Integer array ``[N_L, N_M, N_R]`` with every count at least eighteen.
    sigmas : np.ndarray
        Finite stretching parameters ``[sigma_L, sigma_M, sigma_R]``, each of
        magnitude below one.
    amplitude : float
        Translation amplitude $A$.
    period : float
        Positive translation period $T_m$.

    Returns
    -------
    np.ndarray
        Vector of length ``1 + counts.sum()`` ordered as $x_\tau$, left-grid
        nodes, middle-grid nodes, and right-grid nodes.

    Raises
    ------
    ValueError
        If ``counts`` is not an integer array of shape ``(3,)``; any count is
        below eighteen; ``sigmas`` is not a finite array of shape ``(3,)`` or has
        an entry of magnitude at least one; ``t``, ``amplitude``, or ``period``
        is not finite; or ``period`` is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_ORIGINS = (-1.0, -0.35, 0.1)
_LENGTHS = (0.9, 0.7, 0.9)


def _validated_grid_inputs(t, counts, sigmas, amplitude, period):
    counts = np.asarray(counts)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,) or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas must be a finite array of shape (3,)")
    if np.any(np.abs(sigmas) >= 1.0):
        raise ValueError("every stretching parameter must satisfy |sigma| < 1")
    values = np.asarray([t, amplitude, period], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("t, amplitude, and period must be finite")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    return float(t), counts.astype(int), sigmas, float(amplitude), float(period)




def _oracle_compute_moving_grid_state(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    amplitude: float = 0.1,
    period: float = 1.0,
) -> np.ndarray:
    """Reference evaluation of the stretched translating overset geometry."""
    counts = np.asarray(counts)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,) or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas must be a finite array of shape (3,)")
    if np.any(np.abs(sigmas) >= 1.0):
        raise ValueError("every stretching parameter must satisfy |sigma| < 1")
    if not np.all(np.isfinite(np.asarray([t, amplitude, period], dtype=float))):
        raise ValueError("t, amplitude, and period must be finite")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    t = float(t)
    counts = counts.astype(int)
    amplitude = float(amplitude)
    period = float(period)
    angle = 2.0 * np.pi * t / period
    shift = amplitude * np.sin(angle)
    velocity = 2.0 * np.pi * amplitude * np.cos(angle) / period

    pieces = [np.array([velocity])]
    for grid in range(3):
        xi = np.linspace(0.0, 1.0, counts[grid])
        stretched = xi - (sigmas[grid] / np.pi) * np.sin(np.pi * xi)
        origin = _ORIGINS[grid] + (shift if grid == 1 else 0.0)
        pieces.append(origin + _LENGTHS[grid] * stretched)
    return np.concatenate(pieces)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return maximum-speed, mixed-stretching, and invalid-period cases."""
    return [
        {
            "setup": """import numpy as np
t = 0.0
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -0.15, 0.20])
amplitude = 0.1
period = 1.0
""",
            "call": "compute_moving_grid_state(t, counts, sigmas, amplitude, period)",
            "gold_call": "_oracle_compute_moving_grid_state(t, counts, sigmas, amplitude, period)",
        },
        {
            "setup": """import numpy as np
t = 0.25
counts = np.array([18, 18, 18])
sigmas = np.array([0.0, 0.5, -0.5])
amplitude = 0.08
period = 0.8
""",
            "call": "compute_moving_grid_state(t, counts, sigmas, amplitude, period)",
            "gold_call": "_oracle_compute_moving_grid_state(t, counts, sigmas, amplitude, period)",
        },
        {
            "setup": """import numpy as np
t = 0.0
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -1.0, 0.20])
amplitude = 0.1
period = 1.0
def run_model():
    try:
        compute_moving_grid_state(t, counts, sigmas, amplitude, period)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_moving_grid_state(t, counts, sigmas, amplitude, period)
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
