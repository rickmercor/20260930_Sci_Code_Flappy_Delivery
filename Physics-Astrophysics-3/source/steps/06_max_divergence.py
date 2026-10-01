"""
Implement max_divergence, the spectral divergence diagnostic of a periodic

vector field.

Contract

--------

The input B is a (3, N, N, N) float array on the grid x_i = i/N (likewise y_j,

z_k), component-first, with N >= 2 and a cubic grid. The return is a Python

float: the maximum over the grid of the absolute value of div B.



The divergence is evaluated spectrally, from the coefficients of

``np.fft.fftn`` taken over the three spatial axes together with the wavevector

grid of step 02, with the real part of the inverse transform taken before the

maximum. It is not a finite-difference estimate. Differencing the field on the

grid measures a different operator and returns a different number, so the two

are not interchangeable here.



Inputs

------

B: (3, N, N, N) float array, component-first vector field on the periodic unit

   cube with grid x_i = i/N (likewise y_j, z_k), N >= 2 and a cubic grid.



Returns

-------

float

    max over the grid of |div B|, computed spectrally.

Returns
-------
float : the maximum over all grid points of the absolute value of div B.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def max_divergence(B: np.ndarray) -> float:
    '''Return the maximum absolute divergence of a periodic vector field.

    The divergence is evaluated spectrally, from the coefficients of
    np.fft.fftn taken over the three spatial axes together with the wavevector
    grid of step 02, with the real part of the inverse transform taken before
    the maximum. A grid finite-difference estimate is a different operator and
    is not interchangeable with it.

    Parameters
    ----------
    B : np.ndarray
        (3, N, N, N) float array, component-first vector field on the periodic
        unit cube with grid x_i = i/N (likewise y_j, z_k), with N >= 2 and a
        cubic grid.

    Returns
    -------
    float
        The maximum over all grid points of the absolute value of div B.

    Raises
    ------
    ValueError
        If B is not a (3, N, N, N) float array on a cubic grid with
        N >= 2.
    '''
    return max_div  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_max_divergence(B: np.ndarray) -> float:
    arr = np.asarray(B, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("B must be an array of shape (3, N, N, N)")
    n = int(arr.shape[1])
    if arr.shape[2] != n or arr.shape[3] != n:
        raise ValueError("B must be defined on a cubic (N, N, N) grid")
    if n < 2:
        raise ValueError("N must be >= 2")

    kgrid = _oracle_wavevector_grid(n)
    KX, KY, KZ = kgrid[0], kgrid[1], kgrid[2]

    Bh = np.fft.fftn(arr, axes=(1, 2, 3))
    div_hat = 1j * (KX * Bh[0] + KY * Bh[1] + KZ * Bh[2])
    div = np.real(np.fft.ifftn(div_hat))
    return float(np.max(np.abs(div)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the raw N = 8 seed field, whose divergence is large ---
        {
            "setup": """import numpy as np
n = 8
amp = 20.0
sig = 1.0 / 30.0
kx = 4
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (XX - 0.5) ** 2 + (YY - 0.5) ** 2 + (ZZ - 0.5) ** 2
env = amp * np.exp(-dr2 / (2 * sig ** 2))
phi = 2 * np.pi * kx * XX
B_in = np.empty((3, n, n, n))
B_in[0] = 1.0
B_in[1] = np.cos(phi) * env
B_in[2] = np.sin(phi) * env
""",
            "call": "max_divergence(B_in)",
            "gold_call": "_oracle_max_divergence(B_in)",
        },
        # --- Normal: a smoother N = 8 seed, well resolved on the grid ---
        {
            "setup": """import numpy as np
n = 8
amp = 2.0
sig = 0.15
kx = 2
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (XX - 0.5) ** 2 + (YY - 0.5) ** 2 + (ZZ - 0.5) ** 2
env = amp * np.exp(-dr2 / (2 * sig ** 2))
phi = 2 * np.pi * kx * XX
B_in = np.empty((3, n, n, n))
B_in[0] = 1.0
B_in[1] = np.cos(phi) * env
B_in[2] = np.sin(phi) * env
""",
            "call": "max_divergence(B_in)",
            "gold_call": "_oracle_max_divergence(B_in)",
        },
        # --- Boundary: the projected N = 8 seed, which must be at round-off ---
        {
            "setup": """import numpy as np
n = 8
amp = 20.0
sig = 1.0 / 30.0
kx = 4
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (XX - 0.5) ** 2 + (YY - 0.5) ** 2 + (ZZ - 0.5) ** 2
env = amp * np.exp(-dr2 / (2 * sig ** 2))
phi = 2 * np.pi * kx * XX
F_seed = np.empty((3, n, n, n))
F_seed[0] = 1.0
F_seed[1] = np.cos(phi) * env
F_seed[2] = np.sin(phi) * env
B_in = _oracle_solenoidal_projection(F_seed)
""",
            "call": "max_divergence(B_in)",
            "gold_call": "_oracle_max_divergence(B_in)",
        },
        # --- Edge: a uniform constant field, exactly zero divergence ---
        {
            "setup": """import numpy as np
n = 4
B_in = np.empty((3, n, n, n))
B_in[0] = 1.0
B_in[1] = -2.0
B_in[2] = 0.5
""",
            "call": "max_divergence(B_in)",
            "gold_call": "_oracle_max_divergence(B_in)",
        },
        # --- Invalid: only two components, shape (2, 8, 8, 8) ---
        {
            "setup": """import numpy as np
B_bad = np.zeros((2, 8, 8, 8))
def run_model():
    try:
        max_divergence(B_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_max_divergence(B_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-cubic grid, shape (3, 8, 8, 4) ---
        {
            "setup": """import numpy as np
B_bad = np.zeros((3, 8, 8, 4))
def run_model():
    try:
        max_divergence(B_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_max_divergence(B_bad)
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
