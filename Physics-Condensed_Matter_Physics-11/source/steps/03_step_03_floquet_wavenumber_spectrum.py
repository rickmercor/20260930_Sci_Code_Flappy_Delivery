"""
A quadratic matrix polynomial has no direct eigensolver, so it is reduced to a linear one. The standard device introduces the product of the wavenumber with the eigenvector as a second unknown, which doubles the dimension and leaves a generalised linear eigenvalue problem in block form, the companion pencil. A window truncated at n_order carries 2 n_order + 1 harmonics, so the pencil has side 4 n_order + 2 and returns that many wavenumber eigenvalues with their harmonic mode shapes. That count is the first thing worth checking against the physics: the unmodulated rod supports two waves, one running each way, at each of the 2 n_order + 1 shifted frequencies, which is the same number.

Because the segment is posed for the wavenumber at a prescribed real frequency, the supersonic regime returns eigenvalues that are real to working precision. Every mode the segment supports then propagates along the rod without growing or decaying in space, so the amplification this task measures is not the spatial growth of an evanescent or spatially amplifying mode. The largest magnitude of the imaginary part across the spectrum is returned so that this can be confirmed rather than assumed, and a large value is the signature of a configuration that has left the supersonic regime. Note what the reality of this spectrum does not establish. It is a statement about how a disturbance varies in space at a prescribed real frequency, and it decides nothing about whether the finite segment is stable in time; that is a separate question about the bounded system, and answering it needs a Floquet analysis of the finite segment or a time-domain simulation rather than this spectrum.

The eigenvectors are fixed only up to a scale by any solver, so each is normalised to unit Euclidean length before it is returned. The eigenvalues themselves come back in whatever order the underlying routine produced, which is not reproducible across libraries or across threading configurations, so they are sorted here by real part and then by imaginary part purely to make the returned arrays deterministic. That ordering carries no physical meaning and must not be used to decide which mode is which; the next step decides that from a property of the mode itself.

Returns
-------
dict holding a complex128 wavenumbers of shape (4 n_order + 2,) sorted by real part and then imaginary part; a complex128 mode_shapes of shape (2 n_order + 1, 4 n_order + 2) whose column j is the unit-norm harmonic mode shape of eigenvalue j; and a native float max_abs_imag, the largest magnitude of the imaginary part over the spectrum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def floquet_wavenumber_spectrum(a2: np.ndarray, a1: np.ndarray, a0: np.ndarray) -> dict:
    """Linearise the quadratic wavenumber problem and solve it for every mode the segment supports.

    Parameters
    ----------
    a2 : np.ndarray
        Coefficient matrix of the square of the wavenumber.
    a1 : np.ndarray
        Coefficient matrix of the first power of the wavenumber.
    a0 : np.ndarray
        Constant coefficient matrix.

    Returns
    -------
    dict
        Under the keys wavenumbers, mode_shapes and max_abs_imag. The wavenumbers
        entry is a complex128 array of shape (4 n_order + 2,) sorted by real part
        then imaginary part. The mode_shapes entry is a complex128 array of shape
        (2 n_order + 1, 4 n_order + 2) whose columns are unit-norm mode shapes in
        the same order. The max_abs_imag entry is a native float.

    Raises
    ------
    ValueError
        If the three matrices are not square, are not all of the same shape, do not
        have odd side length, are not finite, or if the pencil returns a number of
        finite eigenvalues other than twice the side length.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg as sla


def _oracle_floquet_wavenumber_spectrum(a2: np.ndarray, a1: np.ndarray, a0: np.ndarray) -> dict:
    mats = [np.asarray(m, dtype=np.float64) for m in (a2, a1, a0)]
    for m in mats:
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("each coefficient matrix must be square")
        if not np.all(np.isfinite(m)):
            raise ValueError("each coefficient matrix must be finite")
    if mats[0].shape != mats[1].shape or mats[1].shape != mats[2].shape:
        raise ValueError("the coefficient matrices must share one shape")
    side = mats[0].shape[0]
    if side % 2 != 1:
        raise ValueError("the coefficient matrices must have odd side length")
    a2m, a1m, a0m = mats

    zero = np.zeros((side, side), dtype=np.float64)
    identity = np.eye(side, dtype=np.float64)
    left = np.block([[zero, identity], [-a0m, -a1m]])
    right = np.block([[identity, zero], [zero, a2m]])
    values, vectors = sla.eig(left, right)

    keep = np.isfinite(values)
    values = values[keep]
    vectors = vectors[:, keep]
    if values.size != 2 * side:
        raise ValueError("the companion pencil did not return twice the side length in finite eigenvalues")

    shapes = np.asarray(vectors[:side, :], dtype=np.complex128)
    norms = np.linalg.norm(shapes, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("the pencil returned a null mode shape")
    shapes = shapes / norms

    order = np.lexsort((values.imag, values.real))
    values = np.asarray(values[order], dtype=np.complex128)
    shapes = np.ascontiguousarray(shapes[:, order])
    return {
        "wavenumbers": values,
        "mode_shapes": shapes,
        "max_abs_imag": float(np.abs(values.imag).max()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nC = np.array([[1.0, 0.15, 0.0], [0.15, 1.0, 0.15], [0.0, 0.15, 1.0]])\nS = np.diag([-10.0, 0.0, 10.0])\nW = np.diag([(15.0 + k * 20.0) ** 2 for k in (-1, 0, 1)])\nA2, A1, A0 = C, S @ C + C @ S, S @ C @ S - W\ndef pack(d):\n    values = np.asarray(d['wavenumbers'], dtype=complex)\n    modes = np.asarray(d['mode_shapes'], dtype=complex)\n    pieces = [np.asarray([values.ndim, values.size, modes.ndim] + list(modes.shape), dtype=float), values.real.ravel(), values.imag.ravel(), np.asarray([d['max_abs_imag']], dtype=float)]\n    if modes.ndim == 2:\n        norms = np.linalg.norm(modes, axis=0)\n        pieces.append(norms)\n        for mode, norm in zip(modes.T, norms):\n            unit = mode / norm if np.isfinite(norm) and norm > 0.0 else np.full(mode.shape, np.nan + 0j)\n            projector = np.outer(unit, np.conjugate(unit))\n            pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\n",
            "call": 'pack(floquet_wavenumber_spectrum(np.array(A2, copy=True),np.array(A1, copy=True),np.array(A0, copy=True)))',
            "gold_call": 'pack(_oracle_floquet_wavenumber_spectrum(np.array(A2, copy=True),np.array(A1, copy=True),np.array(A0, copy=True)))',
        },
        {
            "setup": "import numpy as np\nA2, A1, A0 = np.eye(1), np.zeros((1, 1)), np.array([[-4.0]])\ndef pack(d):\n    values = np.asarray(d['wavenumbers'], dtype=complex)\n    modes = np.asarray(d['mode_shapes'], dtype=complex)\n    pieces = [np.asarray([values.ndim, values.size, modes.ndim] + list(modes.shape), dtype=float), values.real.ravel(), values.imag.ravel(), np.asarray([d['max_abs_imag']], dtype=float)]\n    if modes.ndim == 2:\n        norms = np.linalg.norm(modes, axis=0)\n        pieces.append(norms)\n        for mode, norm in zip(modes.T, norms):\n            unit = mode / norm if np.isfinite(norm) and norm > 0.0 else np.full(mode.shape, np.nan + 0j)\n            projector = np.outer(unit, np.conjugate(unit))\n            pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\n",
            "call": 'pack(floquet_wavenumber_spectrum(np.array(A2, copy=True),np.array(A1, copy=True),np.array(A0, copy=True)))',
            "gold_call": 'pack(_oracle_floquet_wavenumber_spectrum(np.array(A2, copy=True),np.array(A1, copy=True),np.array(A0, copy=True)))',
        },
        {
            "setup": "import numpy as np\nA2, A1, A0 = np.eye(3), np.zeros((3, 3)), np.diag([1.0, 4.0, 9.0])\ndef pack(d):\n    values = np.asarray(d['wavenumbers'], dtype=complex)\n    modes = np.asarray(d['mode_shapes'], dtype=complex)\n    pieces = [np.asarray([values.ndim, values.size, modes.ndim] + list(modes.shape), dtype=float), values.real.ravel(), values.imag.ravel(), np.asarray([d['max_abs_imag']], dtype=float)]\n    if modes.ndim == 2:\n        norms = np.linalg.norm(modes, axis=0)\n        pieces.append(norms)\n        for mode, norm in zip(modes.T, norms):\n            unit = mode / norm if np.isfinite(norm) and norm > 0.0 else np.full(mode.shape, np.nan + 0j)\n            projector = np.outer(unit, np.conjugate(unit))\n            pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\n",
            "call": 'pack(floquet_wavenumber_spectrum(np.array(A2, copy=True),np.array(A1, copy=True),np.array(A0, copy=True)))',
            "gold_call": 'pack(_oracle_floquet_wavenumber_spectrum(np.array(A2, copy=True),np.array(A1, copy=True),np.array(A0, copy=True)))',
        },
        {
            "setup": "import numpy as np\nI3 = np.eye(3)\nZ3 = np.zeros((3, 3))\nI4 = np.eye(4)\nRECT = np.ones((3, 4))\nNAN = np.diag([1.0, float('nan'), 1.0])\ndef verdict(fn, a2=I3, a1=Z3, a0=I3):\n    try:\n        fn(np.array(a2, copy=True), np.array(a1, copy=True), np.array(a0, copy=True))\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": '(verdict(floquet_wavenumber_spectrum, a2=RECT), verdict(floquet_wavenumber_spectrum, a2=I4), verdict(floquet_wavenumber_spectrum, a0=NAN), verdict(floquet_wavenumber_spectrum))',
            "gold_call": '(verdict(_oracle_floquet_wavenumber_spectrum, a2=RECT), verdict(_oracle_floquet_wavenumber_spectrum, a2=I4), verdict(_oracle_floquet_wavenumber_spectrum, a0=NAN), verdict(_oracle_floquet_wavenumber_spectrum))',
        },
    ]
