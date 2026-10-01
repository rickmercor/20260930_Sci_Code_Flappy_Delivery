"""
Build the Gaussian-damped Cartesian multipole interaction tensors of ranks 0 through 4 for a given displacement between two MM centres of mass and a Gaussian width g.  These are used for every near-field MM--MM pair by the multipole potential calculator

When two point-multipole sites approach one another the bare interaction tensors

T^(k) ~ r^-(k+1) diverge and can drive the induced-moment iteration to a polarization catastrophe.  A workable scheme must damp the interaction at short range while leaving it essentially unchanged at long range.  When both partners are point multipoles with no independent spatial extent, the natural fix is to damp each tensor rank directly -- a family of screening functions, one per rank, that regularise the monopole, dipole, quadrupole and higher-order terms so none of them diverges as the separation goes to zero.  Normalised-Gaussian screening is used here; its kernels satisfy a simple recursion in the tensor rank, which is why the higher-rank forms follow from the zeroth by differentiation.

Returns
-------
return np.zeros(_TOTAL, dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import erf

_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def gaussian_damped_tensors(r_vec, g=None) -> np.ndarray:
    # Flat layout of a "tensor set": [T0 | T1(3) | T2(3,3) | T3(3,3,3) | T4(3,3,3,3)]
    '''Gaussian-damped Cartesian interaction tensors T^{ij,d} of ranks 0..4.

    Parameters
    ----------
    r_vec : array_like, shape (3,)
        Displacement between the two MM centres of mass (field point minus source).
        Must be a finite, non-zero 3-vector.
    g : float or None
        Gaussian damping width (strictly positive, same length unit as ``r_vec``).
        ``g = None`` returns the bare (undamped) tensor set -- the g -> 0 limit in which
        every kernel lambda_k -> 1.

    Returns
    -------
    tensors : np.ndarray, shape (121,)
        The five (damped) tensors flattened row-major and concatenated in the order
        [T0 (1), T1 (3), T2 (9), T3 (27), T4 (81)].

    Raises
    ------
    ValueError
        If ``r_vec`` is not a finite 3-vector or is the zero vector, or if ``g`` is
        given (not ``None``) but is not finite or is not strictly positive.
    '''
    return np.zeros(_TOTAL, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf

_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def _lambda_series(s: float, kmax: int = 4) -> np.ndarray:
    """Gaussian damping kernels [lambda_0, ..., lambda_kmax] at screened distance s = r/g."""

    s = float(s)
    erf_s = erf(s)
    gauss = np.exp(-s * s)
    two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)
    lams = np.empty(kmax + 1, dtype=float)
    lams[0] = erf_s
    poly = 0.0        # running sum_{i=1..k} c_i s^{2i-1}
    c = 1.0           # c_1
    for k in range(1, kmax + 1):
        poly += c * s ** (2 * k - 1)
        lams[k] = erf_s - two_over_sqrt_pi * poly * gauss
        c *= 2.0 / (2.0 * k + 1.0)
    return lams


def _bare_tensor_set(r_vec) -> np.ndarray:
    """Undamped Cartesian interaction tensors T^(0..4) (gradients of 1/r), flattened (121,)."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(3)
    r = float(np.sqrt(r_vec @ r_vec))
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("r_vec must be a non-zero finite 3-vector")
    d = np.eye(3)
    x = r_vec

    t0 = np.array([1.0 / r])
    t1 = -x / r ** 3
    t2 = (3.0 * np.outer(x, x) - r ** 2 * d) / r ** 5

    t3 = np.zeros((3, 3, 3))
    xxx = x[:, None, None] * x[None, :, None] * x[None, None, :]
    dd3 = (x[:, None, None] * d[None, :, :]
           + x[None, :, None] * d[:, None, :]
           + x[None, None, :] * d[:, :, None])
    t3 = -(15.0 * xxx) / r ** 7 + 3.0 * dd3 / r ** 5

    xxxx = xxx[..., None] * x[None, None, None, :]
    # six r r delta permutations
    rr_d = (
        np.einsum("a,b,cd->abcd", x, x, d)
        + np.einsum("a,c,bd->abcd", x, x, d)
        + np.einsum("a,d,bc->abcd", x, x, d)
        + np.einsum("b,c,ad->abcd", x, x, d)
        + np.einsum("b,d,ac->abcd", x, x, d)
        + np.einsum("c,d,ab->abcd", x, x, d)
    )
    dddd = (
        np.einsum("ab,cd->abcd", d, d)
        + np.einsum("ac,bd->abcd", d, d)
        + np.einsum("ad,bc->abcd", d, d)
    )
    t4 = 105.0 * xxxx / r ** 9 - 15.0 * rr_d / r ** 7 + 3.0 * dddd / r ** 5

    return np.concatenate([t0, t1.ravel(), t2.ravel(), t3.ravel(), t4.ravel()])


def _apply_lambda(bare: np.ndarray, lams: np.ndarray, r_vec) -> np.ndarray:
    """Rebuild the damped tensor set from the bare one by inserting lambda_k rank by rank."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(3)
    r = float(np.sqrt(r_vec @ r_vec))
    d = np.eye(3)
    x = r_vec
    l0, l1, l2, l3, l4 = lams

    out = np.empty(_TOTAL, dtype=float)
    out[_SLICES[0]] = l0 / r
    out[_SLICES[1]] = (-x / r ** 3 * l1)
    t2 = 3.0 * np.outer(x, x) / r ** 5 * l2 - d / r ** 3 * l1
    out[_SLICES[2]] = t2.ravel()

    xxx = x[:, None, None] * x[None, :, None] * x[None, None, :]
    dd3 = (x[:, None, None] * d[None, :, :]
           + x[None, :, None] * d[:, None, :]
           + x[None, None, :] * d[:, :, None])
    t3 = -15.0 * xxx / r ** 7 * l3 + 3.0 * dd3 / r ** 5 * l2
    out[_SLICES[3]] = t3.ravel()

    xxxx = xxx[..., None] * x[None, None, None, :]
    rr_d = (
        np.einsum("a,b,cd->abcd", x, x, d)
        + np.einsum("a,c,bd->abcd", x, x, d)
        + np.einsum("a,d,bc->abcd", x, x, d)
        + np.einsum("b,c,ad->abcd", x, x, d)
        + np.einsum("b,d,ac->abcd", x, x, d)
        + np.einsum("c,d,ab->abcd", x, x, d)
    )
    dddd = (
        np.einsum("ab,cd->abcd", d, d)
        + np.einsum("ac,bd->abcd", d, d)
        + np.einsum("ad,bc->abcd", d, d)
    )
    t4 = 105.0 * xxxx / r ** 9 * l4 - 15.0 * rr_d / r ** 7 * l3 + 3.0 * dddd / r ** 5 * l2
    out[_SLICES[4]] = t4.ravel()
    return out

def _oracle_gaussian_damped_tensors(r_vec, g=None) -> np.ndarray:
    """Reference damped-tensor builder (Gaussian screening recursion, ranks 0-4)."""
    _SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
    _TOTAL = 121
    r_vec = np.asarray(r_vec, dtype=float).reshape(-1)
    if r_vec.size != 3 or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    r = float(np.sqrt(r_vec @ r_vec))
    if r <= 0.0:
        raise ValueError("r_vec must be non-zero (COM separation)")

    bare = _bare_tensor_set(r_vec)
    if g is None:
        return bare
    if not np.isfinite(g) or float(g) <= 0.0:
        raise ValueError("g must be a finite strictly-positive width (or None for undamped)")
    lams = _lambda_series(r / float(g), kmax=4)
    return _apply_lambda(bare, lams, r_vec)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test-case specifications."""
    return [
        {
            "setup": """import numpy as np
r_vec = np.array([2.4, -1.1, 0.7], dtype=float) * 1.8897261246  # Angstrom -> Bohr
g = 0.32 * 1.8897261246
""",
            "call": "gaussian_damped_tensors(r_vec, g)",
            "gold_call": "_oracle_gaussian_damped_tensors(r_vec, g)",
        },
        {
            "setup": """import numpy as np
r_vec = np.array([1.0, 0.5, -0.25], dtype=float)
bare = _bare_tensor_set(r_vec)
""",
            "call": "float(np.max(np.abs(gaussian_damped_tensors(r_vec, None) - bare)))",
            "gold_call": "float(np.max(np.abs(_oracle_gaussian_damped_tensors(r_vec, g=None) - bare)))",
        },
        {
            "setup": """import numpy as np
r_vec = np.array([0.15, -0.05, 0.02], dtype=float)
g = 0.6
""",
            "call": "gaussian_damped_tensors(r_vec, g)",
            "gold_call": "_oracle_gaussian_damped_tensors(r_vec, g)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        gaussian_damped_tensors(np.zeros(3), 0.5); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gaussian_damped_tensors(np.zeros(3), 0.5); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
r_vec = np.array([1.0, 1.0, 1.0])
def run_model():
    try:
        gaussian_damped_tensors(r_vec, 0.0); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gaussian_damped_tensors(r_vec, 0.0); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
