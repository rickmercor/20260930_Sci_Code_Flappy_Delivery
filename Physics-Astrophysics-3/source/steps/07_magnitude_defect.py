"""
Implement magnitude_defect, the uniformity diagnostic of the field magnitude.

Contract

--------

The input B is a (3, N, N, N) float array on the periodic unit cube in the

component-first layout used throughout this task, with N >= 2 and a cubic grid.

The return is a Python float: the population standard deviation of |B| over all

N**3 grid sites, in the units of the field itself. The magnitude is formed

pointwise as the square root of the sum of squares over the component axis, and

the standard deviation is the numpy default, ddof = 0.



Because the diagnostic is a standard deviation and not a deviation from unity,

it measures magnitude UNIFORMITY rather than magnitude scale: any field whose

length is the same at every site returns exactly 0.0, whether that common length

is 1.0 or not. The scale is fixed elsewhere; this quantity isolates the spread

about whatever mean the field happens to have.



Inputs

------

B: (3, N, N, N) float array, a vector field on the periodic unit cube in the

    component-first layout used throughout this task



Returns

-------

defect: float, the standard deviation of |B| over all N**3 grid sites

Returns
-------
defect : float, population standard deviation (ddof = 0) of |B| over all N**3 grid sites.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def magnitude_defect(B: np.ndarray) -> float:
    '''Return the spread of the pointwise field magnitude over the grid.

    Parameters
    ----------
    B : np.ndarray
        Vector field of shape (3, N, N, N) on the periodic unit cube, stored
        component-first with the spatial axes ordered (x, y, z).

    Returns
    -------
    defect : float
        Population standard deviation (ddof = 0) of |B| over all N**3 grid
        sites, in the units of B. It is 0.0 for any field of uniform
        magnitude, whether or not that common magnitude equals one.

    Raises
    ------
    ValueError
        If B is not a (3, N, N, N) float array with equal trailing
        dimensions and N >= 2.
    '''
    return defect  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_magnitude_defect(B: np.ndarray) -> float:
    arr = np.asarray(B, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("B must be an array of shape (3, N, N, N)")
    n = int(arr.shape[1])
    if arr.shape[2] != n or arr.shape[3] != n:
        raise ValueError("B must be defined on a cubic (N, N, N) grid")
    if n < 2:
        raise ValueError("N must be >= 2")

    mag = np.sqrt(arr[0] ** 2 + arr[1] ** 2 + arr[2] ** 2)
    return float(np.std(mag))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the unconverged N = 8 seed field, whose magnitude is far
        #     from uniform because the Gaussian envelope is localised ---
        {
            "setup": """import numpy as np
N = 8
A = 20.0
sigma = 1.0 / 6.0
kx = 2
coord = np.arange(N) / N
X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
env = A * np.exp(-dr2 / (2.0 * sigma ** 2))
phi = 2.0 * np.pi * kx * X
F0 = np.empty((3, N, N, N))
F0[0] = 1.0
F0[1] = np.cos(phi) * env
F0[2] = np.sin(phi) * env
""",
            "call": "magnitude_defect(F0)",
            "gold_call": "_oracle_magnitude_defect(F0)",
        },
        # --- Normal: a coarser seed with a different envelope width and
        #     wavenumber, so the defect is a different non-zero value ---
        {
            "setup": """import numpy as np
N = 6
A = 3.0
sigma = 1.0 / 4.0
kx = 1
coord = np.arange(N) / N
X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
env = A * np.exp(-dr2 / (2.0 * sigma ** 2))
phi = 2.0 * np.pi * kx * X
F0 = np.empty((3, N, N, N))
F0[0] = 1.0
F0[1] = np.cos(phi) * env
F0[2] = np.sin(phi) * env
""",
            "call": "magnitude_defect(F0)",
            "gold_call": "_oracle_magnitude_defect(F0)",
        },
        # --- Boundary: a field of exactly unit magnitude everywhere, the fully
        #     converged limit, giving a defect of exactly 0.0 ---
        {
            "setup": """import numpy as np
N = 8
idx = np.arange(N)
I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
sel = (I + J + K) % 3
sign = np.where((I * J + K) % 2 == 0, 1.0, -1.0)
B = np.zeros((3, N, N, N))
B[0] = np.where(sel == 0, sign, 0.0)
B[1] = np.where(sel == 1, sign, 0.0)
B[2] = np.where(sel == 2, sign, 0.0)
""",
            "call": "magnitude_defect(B)",
            "gold_call": "_oracle_magnitude_defect(B)",
        },
        # --- Edge: uniform but non-unit magnitude, which also scores 0.0
        #     because the standard deviation is referenced to the mean and so
        #     measures uniformity rather than absolute scale ---
        {
            "setup": """import numpy as np
N = 10
scale = 7.5
idx = np.arange(N)
I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
sel = (I + 2 * J + K) % 3
sign = np.where((I + K) % 2 == 0, scale, -scale)
B = np.zeros((3, N, N, N))
B[0] = np.where(sel == 0, sign, 0.0)
B[1] = np.where(sel == 1, sign, 0.0)
B[2] = np.where(sel == 2, sign, 0.0)
""",
            "call": "magnitude_defect(B)",
            "gold_call": "_oracle_magnitude_defect(B)",
        },
        # --- Invalid: wrong number of components on the leading axis ---
        {
            "setup": """import numpy as np
B = np.ones((2, 4, 4, 4))
def run_model():
    try:
        magnitude_defect(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_magnitude_defect(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a scalar-shaped array with no component axis at all ---
        {
            "setup": """import numpy as np
B = np.ones((4, 4, 4))
def run_model():
    try:
        magnitude_defect(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_magnitude_defect(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-cubic spatial grid ---
        {
            "setup": """import numpy as np
B = np.ones((3, 4, 5, 4))
def run_model():
    try:
        magnitude_defect(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_magnitude_defect(B)
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
