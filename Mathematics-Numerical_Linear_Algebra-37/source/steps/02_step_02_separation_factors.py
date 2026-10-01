"""
Compute the separable expansion of a sampled central difference potential and return its factors packed as a single array. Determine the number of retained terms as the count of singular values strictly exceeding rtol times the largest one, distribute each retained singular value symmetrically as its square root between the two factors, and fix the residual sign freedom by requiring that, in each spatial factor column, the first entry whose magnitude exceeds 1e-12 times that column's largest magnitude be positive, applying the same flip to the matching dual factor column. Return the spatial factors stacked above the dual factors. Raise ValueError if the input is not a non-empty two-dimensional array, if it contains non-finite entries, or if rtol does not satisfy 0 < rtol < 1.

A low-rank representation of the quasi-distribution buys nothing unless the operator acting on it also respects the factorisation. The advection term does so automatically, because differentiation in position and multiplication by the wave vector each touch only one variable. The pseudo-differential term does not: its nonlocality couples the two variables, and evaluating its projection would in general require rebuilding the full phase-space array at every step, which is precisely the cost the low-rank ansatz was introduced to avoid. The resolution is to demand that the central difference potential itself splits into a finite sum of products, each a function of position multiplied by a function of the dual variable. The number of terms in that sum is the separation rank, and it absorbs the entire position-momentum coupling of the operator, so the whole method's efficiency is governed by how small it is. Real-analytic potentials admit such an expansion, and computing it numerically is a one-off cost paid during initialisation. Two conventions must be fixed for the result to be reproducible: how the singular value is apportioned between the two factors, and the sign of each factor pair, since flipping both leaves their product unchanged. Symmetric grids make the second convention delicate, because the largest-magnitude entry of a factor column can be attained more than once and any tie-break decided by rounding will disagree between linear-algebra backends.

Returns
-------
np.ndarray of shape (Nx + Ny, R), the spatial separation factors stacked above the dual separation factors, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def separation_factors(D_V: "np.ndarray", rtol: float = 1e-12) -> "np.ndarray":
    '''Separate a sampled central difference potential into paired factors.

    Parameters
    ----------
    D_V : np.ndarray
        (Nx, Ny) array of the central difference potential on a tensor grid.
    rtol : float
        Relative threshold on the singular values, 0 < rtol < 1.

    Returns
    -------
    factors : np.ndarray
        (Nx + Ny, R) array whose first Nx rows hold the spatial factors and
        whose remaining Ny rows hold the dual factors, float64.
    '''
    return factors  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_separation_factors(D_V: "np.ndarray", rtol: float = 1e-12) -> "np.ndarray":
    A = np.asarray(D_V, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        raise ValueError("D_V must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("D_V must contain only finite values")
    if not isinstance(rtol, (int, float)) or not (0.0 < float(rtol) < 1.0):
        raise ValueError("rtol must satisfy 0 < rtol < 1")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    R = 0 if s.size == 0 or s[0] <= 0.0 else int(np.count_nonzero(s > float(rtol) * s[0]))
    root = np.sqrt(s[:R])
    DX = U[:, :R] * root
    DY = Vt[:R, :].T * root
    for j in range(R):
        col = DX[:, j]
        big = np.flatnonzero(np.abs(col) > 1e-12 * np.abs(col).max())
        if big.size and col[big[0]] < 0.0:
            DX[:, j] = -DX[:, j]
            DY[:, j] = -DY[:, j]
    return np.vstack([DX, DY]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: harmonic difference potential on the production grid ---
        {
            "setup": """import numpy as np
x = np.linspace(-10.0, 10.0, 128, endpoint=False)
V = lambda t: 0.5*t**2
D_V = V(x[:, None] + 0.5*x[None, :]) - V(x[:, None] - 0.5*x[None, :])
""",
            "call": "separation_factors(D_V, 1e-12)",
            "gold_call": "_oracle_separation_factors(D_V, 1e-12)",
        },
        # --- edge: symmetric grid with a globally negated potential, so the
        #     largest-magnitude entry of the spatial factor is attained twice ---
        {
            "setup": """import numpy as np
x = np.linspace(-4.0, 4.0, 9)
y = np.linspace(-6.0, 6.0, 13)
V = lambda t: 0.5*t**2
D_V = -(V(x[:, None] + 0.5*y[None, :]) - V(x[:, None] - 0.5*y[None, :]))
""",
            "call": "separation_factors(D_V, 1e-12)",
            "gold_call": "_oracle_separation_factors(D_V, 1e-12)",
        },
        # --- edge: anharmonic potential on rectangular grids, R > 1 ---
        {
            "setup": """import numpy as np
x = np.linspace(-2.0, 5.0, 11)
y = np.linspace(-1.5, 4.5, 8)
V = lambda t: t**3/3.0 - 2.0*t
D_V = V(x[:, None] + 0.5*y[None, :]) - V(x[:, None] - 0.5*y[None, :])
""",
            "call": "separation_factors(D_V, 1e-12)",
            "gold_call": "_oracle_separation_factors(D_V, 1e-12)",
        },
        # --- edge: quartic on a symmetric grid, loose threshold ---
        {
            "setup": """import numpy as np
x = np.linspace(-2.0, 2.0, 15)
y = np.linspace(-2.0, 2.0, 9)
V = lambda t: t**4/4.0
D_V = V(x[:, None] + 0.5*y[None, :]) - V(x[:, None] - 0.5*y[None, :])
""",
            "call": "separation_factors(D_V, 1e-8)",
            "gold_call": "_oracle_separation_factors(D_V, 1e-8)",
        },
        # --- edge: smooth barrier with a decaying spectrum truncated mid-tail ---
        {
            "setup": """import numpy as np
x = np.linspace(0.0, 60.0, 33)
y = np.linspace(-8.0, 8.0, 17)
V = lambda t: 0.3*np.exp(-(t-30.0)**2/2.0)
D_V = V(x[:, None] + 0.5*y[None, :]) - V(x[:, None] - 0.5*y[None, :])
""",
            "call": "separation_factors(D_V, 1e-4)",
            "gold_call": "_oracle_separation_factors(D_V, 1e-4)",
        },
        # --- boundary: retained-term count across potentials, including a
        #     constant potential for which the expansion is empty ---
        {
            "setup": """import numpy as np
def build(V, xa, xb, nx, ya, yb, ny):
    x = np.linspace(xa, xb, nx); y = np.linspace(ya, yb, ny)
    return V(x[:, None] + 0.5*y[None, :]) - V(x[:, None] - 0.5*y[None, :])
mats = [build(lambda t: np.full_like(t, 7.0), -3.0, 3.0, 6, -2.0, 2.0, 4),
        build(lambda t: 0.5*t**2, -4.0, 4.0, 9, -6.0, 6.0, 13),
        build(lambda t: t**3/3.0 - 2.0*t, -2.0, 5.0, 11, -1.5, 4.5, 8),
        build(lambda t: t**5, -2.0, 2.0, 21, -2.0, 2.0, 17)]
def probe(fn):
    out = []
    for M in mats:
        try:
            out.append(int(fn(M, 1e-12).shape[1]))
        except ValueError:
            out.append(-1)
        except Exception:
            out.append(-2)
    return out
def run_model():
    return probe(separation_factors)
def run_gold():
    return probe(_oracle_separation_factors)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: one-dimensional input ---
        {
            "setup": """import numpy as np
D_V = np.zeros(5)
def run_model():
    try:
        separation_factors(D_V, 1e-12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_separation_factors(D_V, 1e-12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: threshold outside the open unit interval ---
        {
            "setup": """import numpy as np
D_V = np.eye(3)
def run_model():
    try:
        separation_factors(D_V, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_separation_factors(D_V, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite entry ---
        {
            "setup": """import numpy as np
D_V = np.array([[np.nan, 1.0], [1.0, 1.0]])
def run_model():
    try:
        separation_factors(D_V, 1e-12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_separation_factors(D_V, 1e-12)
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
