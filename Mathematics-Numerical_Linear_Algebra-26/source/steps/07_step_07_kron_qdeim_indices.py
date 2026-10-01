"""
Implement the two-stage Kronecker QDEIM sampling construction.

Use the orthonormal factor bases left and right together with the reduced core basis to return exactly r distinct full Kronecker-row indices in pivot order. Preserve the source construction's reduced selection and C-order row mapping, and reject nonfinite, nonorthonormal, rank-deficient, or incompatible inputs with ValueError.

For a Kronecker-structured basis, separate QDEIM selections on the two factors reduce the candidate rows before a second selection is made on the sampled core. Mapping the reduced pivots back in C order yields the full tensor-product row indices used by the interpolatory projector.

Returns
-------
np.ndarray as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def kron_qdeim_indices(left: np.ndarray, right: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Return full Kronecker-row indices from the source's two-stage sampler.

    ``left`` and ``right`` have orthonormal columns and shapes ``(m1,r)`` and
    ``(m2,r)``. ``core`` has orthonormal columns and shape ``(r*r,r)``.
    Call the earlier ``qdeim_select`` on ``left`` and ``right``. If their
    ordered index arrays are ``p`` and ``q``, form
    ``B = kron(left[p,:], right[q,:]) @ core``. Compute its reduced QR,
    canonicalize each column so the corresponding diagonal of ``R`` is
    positive, and call ``qdeim_select`` on that orthonormal factor. For each
    selected reduced row ``s``, with ``i=s//r`` and ``j=s%r``, return the
    full C-order Kronecker row ``p[i]*m2+q[j]``. Real and complex factor
    bases are accepted. Reject non-finite, non-orthonormal, rank-deficient,
    or incompatible inputs.

    Returns
    -------
    numpy.ndarray
        Exactly ``r`` distinct integer indices in QDEIM pivot order.
    """
    return np.empty(0, dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_kron_qdeim_indices(left: np.ndarray, right: np.ndarray, core: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    core = np.asarray(core)
    if left.ndim != 2 or right.ndim != 2 or core.ndim != 2:
        raise ValueError('left, right, and core must be matrices')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or not np.all(np.isfinite(core)):
        raise ValueError('all inputs must be finite')
    m1, r = left.shape
    m2, r_right = right.shape
    if r < 1 or m1 < r or m2 < r or r_right != r or core.shape != (r * r, r):
        raise ValueError('incompatible Kronecker sampling dimensions')
    eye = np.eye(r)
    if not np.allclose(left.conj().T @ left, eye, rtol=1e-11, atol=1e-12):
        raise ValueError('left must have orthonormal columns')
    if not np.allclose(right.conj().T @ right, eye, rtol=1e-11, atol=1e-12):
        raise ValueError('right must have orthonormal columns')
    if not np.allclose(core.conj().T @ core, eye, rtol=1e-11, atol=1e-12):
        raise ValueError('core must have orthonormal columns')
    left_indices = _oracle_qdeim_select(left)
    right_indices = _oracle_qdeim_select(right)
    sampled = np.kron(left[left_indices, :], right[right_indices, :]) @ core
    qhat, triangular = np.linalg.qr(sampled, mode='reduced')
    diagonal = np.diag(triangular)
    scale = max(1.0, float(np.linalg.norm(sampled, ord=2)))
    if np.any(np.abs(diagonal) <= 1e-12 * scale):
        raise ValueError('the sampled Kronecker core must have full column rank')
    phases = diagonal / np.abs(diagonal)
    qhat = qhat * phases
    pair_indices = _oracle_qdeim_select(qhat)
    first = pair_indices // r
    second = pair_indices % r
    full = left_indices[first] * m2 + right_indices[second]
    if np.unique(full).size != r:
        raise RuntimeError('the mapped Kronecker indices must be distinct')
    return np.asarray(full, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """rng=np.random.default_rng(1401); left=np.linalg.qr(rng.standard_normal((7,3)),mode='reduced')[0]; right=np.linalg.qr(rng.standard_normal((6,3)),mode='reduced')[0]; core=np.linalg.qr(rng.standard_normal((9,3)),mode='reduced')[0]""",
            "call": "kron_qdeim_indices(left, right, core)",
            "gold_call": "_oracle_kron_qdeim_indices(left, right, core)",
        },
        # Case 2
        {
            "setup": """left=np.array([[2**-0.5,0.0],[2**-0.5,0.0],[0.0,1.0]]); right=np.array([[1.0,0.0],[0.0,2**-0.5],[0.0,2**-0.5],[0.0,0.0]]); core=np.linalg.qr(np.array([[1.0,2.0],[0.5,-1.0],[2.0,0.25],[-0.5,1.5]]),mode='reduced')[0]""",
            "call": "kron_qdeim_indices(left, right, core)",
            "gold_call": "_oracle_kron_qdeim_indices(left, right, core)",
        },
        # Case 3
        {
            "setup": """rng=np.random.default_rng(887); left=np.linalg.qr(rng.standard_normal((9,4))+1j*rng.standard_normal((9,4)),mode='reduced')[0]; right=np.linalg.qr(rng.standard_normal((8,4))+1j*rng.standard_normal((8,4)),mode='reduced')[0]; core=np.linalg.qr(rng.standard_normal((16,4))+1j*rng.standard_normal((16,4)),mode='reduced')[0]""",
            "call": "kron_qdeim_indices(left, right, core)",
            "gold_call": "_oracle_kron_qdeim_indices(left, right, core)",
        },
        # Case 4
        {
            "setup": """left=np.eye(3,2); right=np.eye(4,2); core=np.ones((4,2))
def run_model():
    try:
        kron_qdeim_indices(left,right,core); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_kron_qdeim_indices(left,right,core); return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
