"""
Compute the reaction-coordinate mass and crossing diagnostics.

Use the peaked-crossing branch when ``peaked`` is true and the sloped-crossing
branch otherwise.  Return the reduced mass, gradient-gap magnitude, gradient
product scale, and projected curvature in the order and units declared by the
function signature.  Inputs must be finite, real, shape-compatible, and
physically nonsingular; otherwise raise ``ValueError``.

Returns
-------
np.ndarray of shape (4,), ordered as [mu, gradient_gap, gradient_product_root, curvature]; mu is in m_e, both gradient quantities are in E_h bohr^-1, and the mass-weighted curvature is in E_h m_e^-1 bohr^-2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_crossing_descriptor(
    hessian_a: 'np.ndarray',
    hessian_b: 'np.ndarray',
    gradient_a: 'np.ndarray',
    gradient_b: 'np.ndarray',
    masses: 'np.ndarray',
    peaked: bool = True,
) -> 'np.ndarray':
    """Return reduced mass, two gradient invariants, and projected curvature.

    Parameters
    ----------
    hessian_a, hessian_b
        Finite real symmetric Hessians in ``E_h bohr^-2`` with common shape
        ``(n, n)``.
    gradient_a, gradient_b
        Finite real gradient vectors in ``E_h bohr^-1`` with shape ``(n,)``.
    masses
        Strictly positive finite masses in ``m_e`` with shape ``(n,)``.
    peaked
        Select the plus-sign effective Hessian when true and the minus-sign
        branch when false.

    Returns
    -------
    np.ndarray
        ``[mu, gradient_gap, gradient_product_root, curvature]``, with ``mu``
        in ``m_e``, the two gradient quantities in ``E_h bohr^-1``, and the
        mass-weighted curvature in ``E_h m_e^-1 bohr^-2``.

    Raises
    ------
    ValueError
        If shapes or values are invalid, the gradient difference is zero, or
        the projected mode is physically singular.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_crossing_descriptor(
    hessian_a: 'np.ndarray',
    hessian_b: 'np.ndarray',
    gradient_a: 'np.ndarray',
    gradient_b: 'np.ndarray',
    masses: 'np.ndarray',
    peaked: bool = True,
) -> 'np.ndarray':
    raw = [
        np.asarray(hessian_a),
        np.asarray(hessian_b),
        np.asarray(gradient_a),
        np.asarray(gradient_b),
        np.asarray(masses),
    ]
    if any(
        not np.issubdtype(value.dtype, np.number)
        or not np.isrealobj(value)
        or np.any(~np.isfinite(value))
        for value in raw
    ):
        raise ValueError("all arrays must be finite, real, and numeric")
    ha, hb, ga, gb, mass = [np.asarray(value, dtype=float) for value in raw]
    if (
        ha.ndim != 2
        or hb.shape != ha.shape
        or ha.shape[0] != ha.shape[1]
        or ha.shape[0] < 1
        or ga.shape != (ha.shape[0],)
        or gb.shape != ga.shape
        or mass.shape != ga.shape
    ):
        raise ValueError("Hessians, gradients, and masses have incompatible shapes")
    if (
        not np.allclose(ha, ha.T, rtol=0.0, atol=1e-12)
        or not np.allclose(hb, hb.T, rtol=0.0, atol=1e-12)
        or np.any(mass <= 0.0)
        or not isinstance(peaked, (bool, np.bool_))
    ):
        raise ValueError("Hessians must be symmetric, masses positive, and peaked boolean")
    norm_a = float(np.linalg.norm(ga))
    norm_b = float(np.linalg.norm(gb))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("both diabatic gradients must be nonzero")
    branch = 1.0 if bool(peaked) else -1.0
    denominator = norm_b + branch * norm_a
    if abs(denominator) <= 1e-12:
        raise ValueError("the selected effective-Hessian branch is singular")
    effective = (norm_b * ha + branch * norm_a * hb) / denominator
    root_mass = np.sqrt(mass)
    mass_weighted_hessian = effective / np.outer(root_mass, root_mass)
    difference = ga / root_mass - gb / root_mass
    difference_norm = float(np.linalg.norm(difference))
    if difference_norm == 0.0:
        raise ValueError("the mass-weighted gradient difference must be nonzero")
    direction = difference / difference_norm
    curvature = float(direction @ mass_weighted_hessian @ direction)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("the projected effective curvature must be positive")
    unweighted_direction = direction / root_mass
    reduced_mass = 1.0 / float(unweighted_direction @ unweighted_direction)
    gradient_gap = float(np.linalg.norm(ga - gb))
    gradient_product_root = float(np.sqrt(norm_a * norm_b))
    return np.array(
        [reduced_mass, gradient_gap, gradient_product_root, curvature],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nha=np.array([[1.4,0.1],[0.1,1.0]])\nhb=np.array([[1.1,-0.05],[-0.05,1.6]])\nga=np.array([0.3,-0.2])\ngb=np.array([-0.15,0.25])\nm=np.array([12.0,16.0])",
            "call": "compute_crossing_descriptor(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),True)",
            "gold_call": "_oracle_compute_crossing_descriptor(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),True)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nha=np.array([[2.0]])\nhb=np.array([[1.0]])\nga=np.array([0.4])\ngb=np.array([-0.2])\nm=np.array([7.0])",
            "call": "compute_crossing_descriptor(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),True)",
            "gold_call": "_oracle_compute_crossing_descriptor(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),True)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nha=np.diag([2.2,1.8])\nhb=np.diag([0.7,0.9])\nga=np.array([0.5,0.1])\ngb=np.array([0.12,-0.04])\nm=np.array([10.0,14.0])",
            "call": "compute_crossing_descriptor(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),False)",
            "gold_call": "_oracle_compute_crossing_descriptor(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),False)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nha=np.eye(2); hb=np.eye(2); ga=np.array([0.2,-0.1]); gb=-ga; m=np.array([12.0,-1.0])\ndef check(fn):\n try: fn(ha.copy(),hb.copy(),ga.copy(),gb.copy(),m.copy(),True)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(compute_crossing_descriptor)",
            "gold_call": "check(_oracle_compute_crossing_descriptor)",
            "tol": 0.0,
        },
    ]
