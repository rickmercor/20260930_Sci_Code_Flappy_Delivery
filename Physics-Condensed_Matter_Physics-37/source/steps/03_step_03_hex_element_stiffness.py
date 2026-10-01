"""
Use trilinear shape functions on the eight corners and integrate with the two-point Gauss rule in each direction, eight points in all at the local coordinates plus and minus one over the square root of three. That rule is exact for this integrand on an undistorted cube, so the matrix returned is the exact element energy and not an approximation of it; a one-point rule would be rank deficient and admit hourglass modes.

Order the twenty-four degrees of freedom so that the three components of a corner are adjacent, corner index times three plus component. Order the corners themselves by the sign pattern with the first axis varying slowest, so that corner index is four times the first bit plus two times the second bit plus the third, each bit being zero at the low face and one at the high face. The ordering is a convention, but it has to be the same convention the assembly stage uses, and returning it explicitly is what lets the next stage be written without guessing.

Build the constitutive matrix in Voigt form with the normal block carrying lam off the diagonal and lam plus two mu on it, and the three shear entries carrying mu, then form the strain-displacement matrix at each Gauss point and accumulate B transpose C B times the Jacobian determinant. The Jacobian of a cube of edge h is h over two times the identity, so the determinant is h cubed over eight and the derivative conversion is a factor two over h.

Returns
-------
dict, holding the 24 by 24 element_stiffness, the 8 by 3 corner_signs and the element volume.

Every element outside the resonator is a cube of edge h filled with the same isotropic background, so one element stiffness matrix serves the whole assembly and is worth forming once. The elastic energy the matrix has to represent is the one the variational formulation of the problem uses,

$$a(u, v) = lam * integral of (div u)(div v) + (mu / 2) * integral of (grad u + grad u transpose) : (grad v + grad v transpose),$$

which is the ordinary isotropic form: writing the strain as the symmetric part of the displacement gradient, it is the integral of lam times trace of strain squared plus two mu times strain contracted with strain. The traction that goes with it, and that the next stages will read off the resonator surface, is

$$sigma n = lam (div u) n + mu (grad u + grad u transpose) n.$$

Returns
-------
dict, holding the 24 by 24 element_stiffness, the 8 by 3 corner_signs and the element volume.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hex_element_stiffness(
    lam: float,
    mu: float,
    h: float,
) -> dict:
    """Form the twenty-four by twenty-four stiffness of one trilinear cubic element of isotropic background.

    Parameters
    ----------
    lam : float
        First Lame parameter of the background in pascal.
    mu : float
        Shear modulus of the background in pascal, above zero.
    h : float
        Element edge in metre, above zero.

    Returns
    -------
    dict
        Under the keys element_stiffness, corner_signs and volume.
        element_stiffness has shape (24, 24). Despite its name, corner_signs
        is the integer array of binary corner offsets in {0, 1}, with shape
        (8, 3), ordered as (0,0,0), (0,0,1), (0,1,0), (0,1,1),
        (1,0,0), (1,0,1), (1,1,0), (1,1,1). Reference-element signs
        used in the shape functions are 2 * corner_signs - 1.
        volume is the element volume h**3.

    Raises
    ------
    ValueError
        When lam is not finite, when mu is not finite and above zero, or when h is not finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _corner_offsets():
    """The eight corner offsets, first axis varying slowest."""
    return np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int64)


def _voigt_isotropic(lam, mu):
    """Six by six isotropic constitutive matrix in the Voigt ordering used here."""
    C = np.zeros((6, 6), dtype=np.float64)
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _strain_displacement(signs, xi, eta, zeta, h):
    """Six by twenty-four strain-displacement matrix at one local point."""
    s = signs.astype(np.float64) * 2.0 - 1.0          # zero or one becomes minus or plus one
    dN = np.empty((8, 3), dtype=np.float64)
    dN[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
    dN[:, 1] = 0.125 * (1.0 + s[:, 0] * xi) * s[:, 1] * (1.0 + s[:, 2] * zeta)
    dN[:, 2] = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta) * s[:, 2]
    g = dN * (2.0 / h)
    B = np.zeros((6, 24), dtype=np.float64)
    B[0, 0::3] = g[:, 0]
    B[1, 1::3] = g[:, 1]
    B[2, 2::3] = g[:, 2]
    B[3, 1::3] = g[:, 2]
    B[3, 2::3] = g[:, 1]
    B[4, 0::3] = g[:, 2]
    B[4, 2::3] = g[:, 0]
    B[5, 0::3] = g[:, 1]
    B[5, 1::3] = g[:, 0]
    return B


def _oracle_hex_element_stiffness(
    lam: float,
    mu: float,
    h: float,
) -> dict:
    """Reference implementation."""
    lam = float(lam)
    mu = float(mu)
    h = float(h)
    if not np.isfinite(lam):
        raise ValueError("lam must be finite")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and above zero")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and above zero")

    signs = _corner_offsets()
    C = _voigt_isotropic(lam, mu)
    g = 1.0 / np.sqrt(3.0)
    det = (h / 2.0) ** 3
    K = np.zeros((24, 24), dtype=np.float64)
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _strain_displacement(signs, xi, eta, zeta, h)
                K += B.T @ C @ B * det
    K = 0.5 * (K + K.T)
    return {"element_stiffness": K, "corner_signs": signs, "volume": h ** 3}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def digest(out):
    K = out["element_stiffness"]
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    # a free element has exactly six zero-energy rigid motions in three dimensions
    zero = int(np.sum(np.abs(w) < 1e-6 * max(abs(w[-1]), 1.0)))
    return (K.shape, round(float(np.abs(K - K.T).max()), 10), zero,
            round(float(np.trace(K)), 8), round(float(w[-1]), 8),
            round(float(K[0, 0]), 8), round(float(K[0, 3]), 8),
            # the corner ordering and the matrix must agree: permuting the offsets without
            # permuting the rows leaves the trace and the spectrum alone and is still wrong
            tuple(int(v) for v in out["corner_signs"].ravel()),
            round(float(sum((3 * a + c) * K[3 * a + c, (3 * a + c + 7) % 24]
                            for a in range(8) for c in range(3))), 6),
            round(float(out["volume"]), 15))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(hex_element_stiffness(1.5e6, 5.0e5, 0.02 / 24)))",
            "gold_call": "flat(digest(_oracle_hex_element_stiffness(1.5e6, 5.0e5, 0.02 / 24)))",
        },
        {
            "setup": """import numpy as np
def digest(out):
    K = out["element_stiffness"]
    # a uniform translation of the whole element must store no energy at all
    t = np.zeros(24); t[0::3] = 1.0
    return (round(float(t @ K @ t), 8), round(float(np.trace(K)), 8),
            round(float(np.linalg.eigvalsh(K)[-1]), 8))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat((digest(hex_element_stiffness(0.0, 1.0, 1.0)), digest(hex_element_stiffness(2.0e6, 8.0e5, 0.05))))",
            "gold_call": "flat((digest(_oracle_hex_element_stiffness(0.0, 1.0, 1.0)), digest(_oracle_hex_element_stiffness(2.0e6, 8.0e5, 0.05))))",
        },
        {
            "setup": """
def verdict(fn, lam=1.5e6, mu=5.0e5, h=1e-3):
    try:
        fn(lam, mu, h)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
def verdicts(fn):
    return flat((verdict(fn, mu=0.0), verdict(fn, h=-1.0), verdict(fn, lam=float('nan')), verdict(fn, mu=float('inf')), verdict(fn)))
""",
            'call': 'verdicts(hex_element_stiffness)',
            'gold_call': 'verdicts(_oracle_hex_element_stiffness)',
        },
    ]
