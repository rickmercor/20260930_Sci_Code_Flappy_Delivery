"""
Transform a finite six-site open-boundary matrix-product state into the frozen right-canonical representation and retain the five internal Schmidt spectra.

For a normalized open-boundary state, successive right-to-left Schmidt decompositions give site tensors whose right virtual legs are isometric and bond spectra that describe the five bipartitions.  Physical indices are ordered before the left and right virtual indices, site 1 is the most-significant computational-basis index, and deterministic singular-vector phases make the returned representation unique on the nondegenerate rank-two domain.

Returns
-------
Return a two-tuple: six complex128 right-canonical tensors with shapes `(2,1,2)`, four `(2,2,2)`, and `(2,2,1)`, followed by five nonincreasing float64 spectra of shape `(2,)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def canonicalize_initial_mps(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]":
    """Return a normalized right-canonical six-site MPS and five spectra.

    Each input tensor has axis order ``(physical, left_bond, right_bond)``.
    The outer bond dimensions must be one and every internal Schmidt rank must
    be exactly two with nondegenerate positive singular values.  The returned
    tensors have shapes ``(2,1,2)``, four copies of ``(2,2,2)``, and
    ``(2,2,1)``; the five spectra are real arrays of shape ``(2,)``.  Raise
    ``ValueError`` when these shape, finiteness, rank, or nondegeneracy
    conditions are not satisfied.
    """
    return canonical_tensors, bond_singular_values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg


def _phase_fix_right_rows(
    left_vectors: "np.ndarray",
    right_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    left = np.asarray(left_vectors, dtype=np.complex128).copy()
    right_h = np.asarray(right_vectors_h, dtype=np.complex128).copy()
    for k in range(right_h.shape[0]):
        magnitudes = np.abs(right_h[k])
        pivot = int(np.argmax(magnitudes))
        if magnitudes[pivot] == 0.0:
            raise ValueError("a singular vector has no nonzero pivot")
        phase = right_h[k, pivot] / magnitudes[pivot]
        right_h[k] /= phase
        left[:, k] *= phase
    return left, right_h


def _phase_fix_left_columns(
    left_vectors: "np.ndarray",
    right_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    left = np.asarray(left_vectors, dtype=np.complex128).copy()
    right_h = np.asarray(right_vectors_h, dtype=np.complex128).copy()
    for k in range(left.shape[1]):
        magnitudes = np.abs(left[:, k])
        pivot = int(np.argmax(magnitudes))
        if magnitudes[pivot] == 0.0:
            raise ValueError("a singular vector has no nonzero pivot")
        phase = left[pivot, k] / magnitudes[pivot]
        left[:, k] /= phase
        right_h[k] *= phase
    return left, right_h


def _validated_six_site_tensors(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "tuple[np.ndarray, ...]":
    if not isinstance(site_tensors, tuple) or len(site_tensors) != 6:
        raise ValueError("site_tensors must be a tuple of six tensors")
    tensors = tuple(np.asarray(tensor, dtype=np.complex128) for tensor in site_tensors)
    if any(tensor.ndim != 3 or tensor.shape[0] != 2 for tensor in tensors):
        raise ValueError("every tensor must have shape (2, chi_left, chi_right)")
    if any(not np.all(np.isfinite(tensor)) for tensor in tensors):
        raise ValueError("site tensors must be finite")
    if tensors[0].shape[1] != 1 or tensors[-1].shape[2] != 1:
        raise ValueError("outer MPS bond dimensions must be one")
    if any(tensors[n].shape[2] != tensors[n + 1].shape[1] for n in range(5)):
        raise ValueError("adjacent virtual bond dimensions are incompatible")
    return tensors


def _contract_six_site_mps(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "np.ndarray":
    tensors = _validated_six_site_tensors(site_tensors)
    block = tensors[0][:, 0, :]
    for tensor in tensors[1:]:
        block = np.tensordot(block, tensor, axes=([-1], [1]))
    return np.asarray(block[..., 0], dtype=np.complex128).reshape(64)


def _oracle_canonicalize_initial_mps(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]":
    tensors = _validated_six_site_tensors(site_tensors)
    state = _contract_six_site_mps(tensors)
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("the contracted state must be finite and nonzero")
    work = (state / norm).reshape(32, 2)
    canonical = [None] * 6
    spectra = [None] * 5
    right_dimension = 1

    for site in range(5, 0, -1):
        matrix = work.reshape(2 ** site, 2 * right_dimension)
        left, singular_values, right_h = scipy.linalg.svd(
            matrix, full_matrices=False, lapack_driver="gesvd"
        )
        if singular_values.size < 2 or singular_values[1] <= 1.0e-12 * singular_values[0]:
            raise ValueError("each internal cut must have Schmidt rank exactly two")
        if singular_values.size > 2 and singular_values[2] > 1.0e-12 * singular_values[0]:
            raise ValueError("each internal cut must have Schmidt rank exactly two")
        active = singular_values[:2]
        if active[0] - active[1] <= 1.0e-12 * active[0]:
            raise ValueError("initial active Schmidt values must be nondegenerate")
        left, right_h = _phase_fix_right_rows(left[:, :2], right_h[:2])
        canonical[site] = right_h.reshape(2, 2, right_dimension).transpose(1, 0, 2)
        spectra[site - 1] = np.asarray(active, dtype=np.float64)
        work = left * active[np.newaxis, :]
        right_dimension = 2

    canonical[0] = np.asarray(work.reshape(2, 1, 2), dtype=np.complex128)
    result_tensors = tuple(np.asarray(tensor, dtype=np.complex128) for tensor in canonical)
    result_spectra = tuple(np.asarray(values, dtype=np.float64) for values in spectra)
    return result_tensors, result_spectra

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = '''import numpy as np
def _pack(value):
    parts=[]
    def _visit(item):
        if isinstance(item,(tuple,list)):
            parts.append(np.array([len(item)],dtype=np.complex128))
            for child in item: _visit(child)
        else:
            arr=np.asarray(item)
            parts.append(np.asarray([arr.ndim,*arr.shape],dtype=np.complex128))
            parts.append(arr.astype(np.complex128).ravel())
    _visit(value)
    return np.concatenate(parts)
def make_tensors(offset=0.0):
    dims = (1, 2, 2, 2, 2, 2, 1)
    out = []
    for n in range(6):
        tensor = np.empty((2, dims[n], dims[n + 1]), dtype=np.complex128)
        for sig in range(2):
            for a in range(dims[n]):
                for b in range(dims[n + 1]):
                    x = 11*(n+1)+7*(sig+1)+5*(a+1)+3*(b+1)+(n+1)*(sig+1)*(a+b+2)+offset
                    re = np.sin(0.37*x)+0.31*np.cos(0.19*(x*x+3*a+5*b))
                    im = np.cos(0.29*(x+2*a*b))+0.23*np.sin(0.41*(x*x+sig+b))
                    tensor[sig, a, b] = re + 1j*im
        out.append(tensor)
    return tuple(out)
'''
    return [
        {"setup": common + "site_tensors = make_tensors()", "call": "_pack(canonicalize_initial_mps(site_tensors))", "gold_call": "_pack(_oracle_canonicalize_initial_mps(site_tensors))", "tol": 1e-10},
        {"setup": common + "site_tensors = make_tensors(0.83)", "call": "_pack(canonicalize_initial_mps(site_tensors))", "gold_call": "_pack(_oracle_canonicalize_initial_mps(site_tensors))", "tol": 1e-10},
        {"setup": "import numpy as np\nsite_tensors = tuple(np.array([1.0, 0.0], dtype=np.complex128).reshape(2,1,1) for _ in range(6))\ndef candidate_result():\n    try:\n        canonicalize_initial_mps(site_tensors)\n        return 0\n    except ValueError:\n        return 1\ndef reference_result():\n    try:\n        _oracle_canonicalize_initial_mps(site_tensors)\n        return 0\n    except ValueError:\n        return 1", "call": "candidate_result()", "gold_call": "reference_result()"},
    ]
