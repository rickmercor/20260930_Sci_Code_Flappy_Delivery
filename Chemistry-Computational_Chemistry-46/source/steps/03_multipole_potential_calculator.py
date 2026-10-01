"""
Compute the electrostatic potential (rank 0), field (rank 1) and field gradient (rank 2)produced at a target point by a source carrying a permanent charge q, dipole mu and traceless quadrupole Theta (Buckingham normalisation, Theta_ab = 1/2 sum q (3 r_a r_b - r^2 delta_ab))

Each site is assigned permanent multipole moments about its centre of mass.  The potential, field and field gradient it radiates at another point follow from contracting those moments with the Coulomb interaction tensors: the charge couples through rank k, the dipole through rank k+1 and the quadrupole through rank k+2, with the quadrupole in its traceless form.  Both the near-field QM--MM and MM--MM terms and the far-field cluster terms are built from these same rank-0..2 quantities.

Returns
-------
return np.zeros(13, dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# Flat layout of a rank-0..4 "tensor set": [T0 | T1(3) | T2(3,3) | T3(3,3,3) | T4(3,3,3,3)]
_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def multipole_potential_field(r_vec, q: float, mu, theta, tensors=None, iso: float = 1.0) -> np.ndarray:
    '''Potential, field and field gradient at a target point from a permanent multipole.

    Parameters
    ----------
    r_vec : array_like, shape (3,)
        Displacement r_target - r_source (finite, non-zero).
    q : float
        Source permanent charge.
    mu : array_like, shape (3,)
        Source permanent dipole.
    theta : array_like
        Source permanent quadrupole, either a (3,3) traceless matrix or the 5-vector
        [Theta_xx, Theta_yy, Theta_xy, Theta_xz, Theta_yz].
    tensors : array_like, shape (121,), optional
        Pre-built damped interaction-tensor set (flat layout [T0(1) | T1(3) | T2(9) |
        T3(27) | T4(81)], as produced by step 2).  If ``None`` the bare (undamped)
        tensors are built internally from ``r_vec``.
    iso : float, optional
        Scalar isotropic prefactor applied to the whole result (default 1.0); this is
        where the QM--MM envelope S^G multiplies in.

    Returns
    -------
    out : np.ndarray, shape (13,)
        [phi, V_x, V_y, V_z, V_xx, V_xy, V_xz, V_yx, V_yy, V_yz, V_zx, V_zy, V_zz],
        with V_a = -d phi/d r_a and V_ab = -d^2 phi/d r_a d r_b.

    Raises
    ------
    ValueError
        If ``r_vec`` is not a finite 3-vector or is the zero vector; if ``mu`` is not a
        finite 3-vector or ``q`` / ``iso`` is not finite; if ``theta`` is not a (3, 3)
        matrix or a 5-vector, or is not traceless; or if ``tensors`` is supplied with a
        length other than 121.
    '''
    return np.zeros(13, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def _bare_tensor_set(r_vec) -> np.ndarray:
    """Undamped Cartesian interaction tensors T^(0..4) (gradients of 1/r), flattened (121,)."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(-1)
    if r_vec.size != 3 or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    r = float(np.sqrt(r_vec @ r_vec))
    if r <= 0.0:
        raise ValueError("r_vec must be non-zero")
    d = np.eye(3)
    x = r_vec

    t0 = np.array([1.0 / r])
    t1 = -x / r ** 3
    t2 = (3.0 * np.outer(x, x) - r ** 2 * d) / r ** 5

    xxx = x[:, None, None] * x[None, :, None] * x[None, None, :]
    dd3 = (x[:, None, None] * d[None, :, :]
           + x[None, :, None] * d[:, None, :]
           + x[None, None, :] * d[:, :, None])
    t3 = -(15.0 * xxx) / r ** 7 + 3.0 * dd3 / r ** 5

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
    t4 = 105.0 * xxxx / r ** 9 - 15.0 * rr_d / r ** 7 + 3.0 * dddd / r ** 5

    return np.concatenate([t0, t1.ravel(), t2.ravel(), t3.ravel(), t4.ravel()])


def _unpack_tensor_set(flat: np.ndarray):
    """Split a (121,) flat tensor set into (T0 scalar, T1 (3,), T2 (3,3), T3 (3,3,3), T4 (3,3,3,3))."""
    flat = np.asarray(flat, dtype=float).reshape(-1)
    if flat.size != _TOTAL:
        raise ValueError("tensor set must have length 121")
    t0 = float(flat[_SLICES[0]][0])
    t1 = flat[_SLICES[1]].reshape(3)
    t2 = flat[_SLICES[2]].reshape(3, 3)
    t3 = flat[_SLICES[3]].reshape(3, 3, 3)
    t4 = flat[_SLICES[4]].reshape(3, 3, 3, 3)
    return t0, t1, t2, t3, t4


def _as_quadrupole_matrix(theta) -> np.ndarray:
    """Coerce a quadrupole given as (3,3) or as [xx, yy, xy, xz, yz] into a (3,3) traceless matrix."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (3, 3):
        q = 0.5 * (theta + theta.T)
    elif theta.shape == (5,):
        xx, yy, xy, xz, yz = theta
        q = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -(xx + yy)]], dtype=float)
    else:
        raise ValueError("theta must be (3,3) or a 5-vector [xx, yy, xy, xz, yz]")
    if abs(np.trace(q)) > 1e-7 * (1.0 + np.max(np.abs(q))):
        raise ValueError("quadrupole must be traceless")
    return q

def _oracle_multipole_potential_field(r_vec, q: float, mu, theta, tensors=None, iso: float = 1.0) -> np.ndarray:
    """Reference multipole potential/field/field-gradient (ranks 0-2)."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(-1)
    if r_vec.size != 3 or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    if float(r_vec @ r_vec) <= 0.0:
        raise ValueError("r_vec must be non-zero")
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if mu.size != 3 or not np.all(np.isfinite(mu)):
        raise ValueError("mu must be a finite 3-vector")
    if not np.isfinite(q):
        raise ValueError("q must be finite")
    if not np.isfinite(iso):
        raise ValueError("iso must be finite")
    th = _as_quadrupole_matrix(theta)

    if tensors is None:
        flat = _bare_tensor_set(r_vec)
    else:
        flat = np.asarray(tensors, dtype=float).reshape(-1)
    t0, t1, t2, t3, t4 = _unpack_tensor_set(flat)

    phi = q * t0 - mu @ t1 + (1.0 / 3.0) * np.einsum("ab,ab->", th, t2)
    field = -q * t1 + mu @ t2 - (1.0 / 3.0) * np.einsum("bc,abc->a", th, t3)
    grad = -q * t2 + np.einsum("c,abc->ab", mu, t3) - (1.0 / 3.0) * np.einsum("cd,abcd->ab", th, t4)

    out = np.empty(13, dtype=float)
    out[0] = phi
    out[1:4] = field
    out[4:13] = grad.ravel()
    return float(iso) * out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test-case specifications."""
    return [
        {
            "setup": """import numpy as np
r_vec = np.array([1.9, -1.3, 0.8], dtype=float)
q = 0.0
mu = np.array([0.10, 0.05, -0.60], dtype=float)
theta = np.array([0.20, 0.15, -0.05, 0.10, -0.08], dtype=float)
""",
            "call": "multipole_potential_field(r_vec, q, mu, theta)",
            "gold_call": "_oracle_multipole_potential_field(r_vec, q, mu, theta)",
        },
        {
            "setup": """import numpy as np
r_vec = np.array([0.8, 0.0, 0.0], dtype=float)
bare = _bare_tensor_set(r_vec)
scale = np.ones(121)
scale[_SLICES[2]] = 0.5
scale[_SLICES[3]] = 0.25
scale[_SLICES[4]] = 0.10
tensors = bare * scale
q = 0.0
mu = np.array([0.3, -0.2, 0.1])
theta = np.zeros((3, 3))
iso = 0.27
""",
            "call": "multipole_potential_field(r_vec, q, mu, theta, tensors=tensors, iso=iso)",
            "gold_call": "_oracle_multipole_potential_field(r_vec, q, mu, theta, tensors=tensors, iso=iso)",
        },
        {
            # Charged source (q != 0), non-zero dipole and quadrupole, generic r_vec:
            # the only case that exercises the monopole potential term; compares the full
            # returned [phi, field, field-gradient] vector, not a projection of it.
            "setup": """import numpy as np
r_vec = np.array([1.7, -0.9, 1.3], dtype=float)
q = 0.6
mu = np.array([0.30, -0.20, 0.10], dtype=float)
theta = np.array([0.22, 0.18, -0.07, 0.11, -0.09], dtype=float)
""",
            "call": "multipole_potential_field(r_vec, q, mu, theta)",
            "gold_call": "_oracle_multipole_potential_field(r_vec, q, mu, theta)",
        },
        {
            "setup": """import numpy as np
r_vec = np.array([1.0, 1.0, 1.0])
bad = np.eye(3)
def run_model():
    try:
        multipole_potential_field(r_vec, 0.0, np.zeros(3), bad); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_multipole_potential_field(r_vec, 0.0, np.zeros(3), bad); return 0
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
def run_model():
    try:
        multipole_potential_field(np.zeros(3), 0.0, np.zeros(3), np.zeros(3)); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_multipole_potential_field(np.zeros(3), 0.0, np.zeros(3), np.zeros(3)); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
