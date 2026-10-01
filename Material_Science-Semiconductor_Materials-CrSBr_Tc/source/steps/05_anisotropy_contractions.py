"""
Contract the single-ion anisotropy tensor into the local frame. Returns the three RPA anisotropy components as a three-element array.

Decoupling the single-ion anisotropy in the random phase approximation gives a
contribution built from three matrix elements,

    A^RPA_a = <S''z_a> u_a^T K_a u*_a,
    B^RPA_a = <S''z_a> u_a^T K_a u_a,
    C^RPA_a = 2 <S''z_a> v_a^T K_a v_a.

The local frame follows the ordered moment: [u_a]^mu = R^{mu x} + i R^{mu y} and
[v_a]^mu = R^{mu z} for a rotation R in SO(3) that carries the local z onto the moment.

CrSBr is a triaxial ferromagnet. Its point group is mmm, so every off-diagonal element of
K vanishes and K = diag(K^xx, K^yy, K^zz). The hard axis is c, the intermediate axis is a
and the easy axis is b; the moment therefore lies along the crystal y direction. Using the
isotropic invariance of the model to set the hard-axis component to zero,

    K^xx = 0.066,   K^yy = 0.087,   K^zz = 0.0      (meV)

These values belong to the RPA+CD treatment; the same spin-wave dispersion is reproduced by
a different fitted triple under plain RPA and another under Holstein-Primakoff.

All three are real here. A and C do not depend on how the local frame is fixed, but B does:
u is determined only up to an overall phase, and the two right-handed triads
(e1, e2, v) = (z, x, y) and (x, -z, y) differ by a factor -i, which flips the sign of
u^T K u. Only the modulus carries physical content - the magnon spectrum depends on the
anomalous element through |B| - so return

    [A^RPA, |B^RPA|, C^RPA].

Returns
-------
numpy.ndarray: shape (3,) array holding A, |B| and C in meV, the middle entry being the modulus of the anomalous element.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anisotropy_contractions(sz: float) -> "np.ndarray":
    """Contract the single-ion anisotropy tensor into the local frame. Returns the three RPA anisotropy components as a three-element array.

    Args:
        sz: sublattice magnetisation of the site.

    Returns:
        numpy.ndarray: shape (3,) array holding A, |B| and C in meV, the middle entry being the modulus of the anomalous element.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _uv():
    """Local frame, [u]^mu = R^{mu x} + i R^{mu y} and [v]^mu = R^{mu z}, R in SO(3).

    Moment along the easy axis b = crystal y, so v = y_hat; the right-handed triad
    (e1, e2, v) = (z_hat, x_hat, y_hat) gives u = e1 + i e2. Only u's phase is free.
    """
    return np.array([1j, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])


def _K_tensor():
    """Anisotropy tensor diag(K^xx, K^yy, K^zz) in meV, the RPA+CD fitted triple."""
    return np.diag([0.066, 0.087, 0.0]).astype(complex)


def _anisotropy(sz):
    """(A^RPA, B^RPA, C^RPA) for the single-ion anisotropy."""
    u, v = _uv()
    K = _K_tensor()
    return (float(np.real(sz * (u @ K @ np.conj(u)))),
            float(np.real(sz * (u @ K @ u))),
            float(np.real(2.0 * sz * (v @ K @ v))))


def _oracle_anisotropy_contractions(sz: float) -> "np.ndarray":
    """Contract the single-ion anisotropy tensor into the local frame."""
    a, b, c = _anisotropy(sz)
    # The sign of the anomalous element follows the phase convention chosen for u and
    # carries no physical content, so this step reports its modulus.
    return np.array([a, abs(b), c])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: the saturated moment
            "setup": "",
            "call": 'anisotropy_contractions(1.5)',
            "gold_call": '_oracle_anisotropy_contractions(1.5)',
        },
        {
            # normal: a partly demagnetised state
            "setup": "",
            "call": 'anisotropy_contractions(1.0)',
            "gold_call": '_oracle_anisotropy_contractions(1.0)',
        },
        {
            # boundary: zero magnetisation, where all three vanish
            "setup": "",
            "call": 'anisotropy_contractions(0.0)',
            "gold_call": '_oracle_anisotropy_contractions(0.0)',
        },
        {
            # edge: the moment reached just below the transition
            "setup": "",
            "call": 'anisotropy_contractions(0.1388530302)',
            "gold_call": '_oracle_anisotropy_contractions(0.1388530302)',
        },
    ]
