"""
Represent every operation of the group by a two-by-two matrix on the valley doublet.

With the two valley Bloch functions of step 03 in hand, every operation of the group of the wave
vector acts inside their two-dimensional span and is therefore represented by a two-by-two
unitary matrix. Building those matrices explicitly is the step that converts a statement about
crystal symmetry into an algebraic object that can be manipulated.

The action is a substitution of coordinates. An operation {R | t} sends a plane wave exp(-i p . r)
to exp(-i p . (R r + t)), which is another plane wave of the same shell, exp(-i (R^T p) . r),
multiplied by the phase exp(-i p . t). Assembling those images into a matrix on the eight-plane-
wave space and sandwiching it between the two valley rows gives the required two-by-two matrix,
with entry (a, b) equal to the overlap of valley function a with the image of valley function b.

Every one of these matrices turns out to be a multiple of the identity or of one of the three
Pauli matrices in valley space. That is not an accident, and the pattern of which operation maps
to which Pauli matrix is precisely the information the next step needs. Note that the overall
phase in front of each matrix depends on the phase convention chosen for the basis in step 03,
so it is not by itself physically meaningful; what survives any such choice is how the Pauli
matrices are permuted under conjugation, which is what step 05 extracts.

Return the thirty-two matrices in the operation order fixed by step 02.

Returns
-------
np.ndarray of length 256: real parts of the 32x2x2 matrices flattened, then imaginary parts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def valley_representation(basis: np.ndarray, shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    '''Two-by-two matrices representing the group of the wave vector on the valley doublet.

    Parameters
    ----------
    basis : np.ndarray
        Output of step 03: length-32 real array packing the two-by-eight coefficient matrix.
    shell : np.ndarray
        Output of step 01 for the eight-fold shell: length-24 real array.
    group : np.ndarray
        Output of step 02: length-384 real array.

    Returns
    -------
    result : np.ndarray
        Real array of length 256: the real parts of the thirty-two two-by-two matrices flattened
        in operation order and then row-major, followed by the imaginary parts in the same order.

    Raises
    ------
    ValueError
        If the shell, group and basis arrays do not have matching, well-formed shapes.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _shell_from_flat(v):
    return np.rint(np.asarray(v, dtype=float)).astype(int).reshape(-1, 3)


def _group_from_flat(v):
    v = np.asarray(v, dtype=float).ravel()
    return np.rint(v[:288]).astype(int).reshape(32, 3, 3), v[288:384].reshape(32, 3) * np.pi


def _unpack_complex(v, shape):
    v = np.asarray(v, dtype=float).ravel()
    h = v.size // 2
    return (v[:h] + 1j * v[h:]).reshape(shape)


def _op_matrix(R, t, P, idx):
    n = len(P)
    M = np.zeros((n, n), dtype=complex)
    for j, p in enumerate(P):
        M[idx[tuple((np.asarray(R).T @ p).astype(int))], j] = np.exp(-1j * float(np.dot(p, t)))
    return M


def _oracle_valley_representation(basis: np.ndarray, shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    shell_arr = np.asarray(shell, dtype=float).ravel()
    group_arr = np.asarray(group, dtype=float).ravel()
    basis_arr = np.asarray(basis, dtype=float).ravel()
    if shell_arr.size % 3 or group_arr.size != 384 or basis_arr.size != 4 * (shell_arr.size // 3):
        raise ValueError("shell, group and basis must have matching, well-formed shapes")
    P = _shell_from_flat(shell)
    n = len(P)
    rot, tra = _group_from_flat(group)
    B = _unpack_complex(basis, (2, n))
    idx = {tuple(p): i for i, p in enumerate(P)}
    D = np.empty((32, 2, 2), dtype=complex)
    for g in range(32):
        D[g] = B.conj() @ (_op_matrix(rot[g], tra[g], P, idx) @ B.T)
    z = D.ravel()
    return np.concatenate([z.real, z.imag]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)",
            "call": 'valley_representation(_b, _s, _g)',
            "gold_call": '_oracle_valley_representation(_b, _s, _g)',
        },  # normal, the full representation of the group on the valley doublet
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\ndef _unpack(v):\n    v = np.asarray(v, float).ravel(); h = v.size//2\n    return (v[:h] + 1j*v[h:]).reshape(32,2,2)",
            "call": 'np.array([float(np.abs(_unpack(valley_representation(_b, _s, _g))[g][0, 1]) > 0.5) for g in range(32)])',
            "gold_call": 'np.array([float(np.abs(_unpack(_oracle_valley_representation(_b, _s, _g))[g][0, 1]) > 0.5) for g in range(32)])',
        },  # boundary, which operations exchange the two valleys and which do not
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\ndef _unpack(v):\n    v = np.asarray(v, float).ravel(); h = v.size//2\n    return (v[:h] + 1j*v[h:]).reshape(32,2,2)",
            "call": 'np.sort(np.abs(np.array([np.trace(M) for M in _unpack(valley_representation(_b, _s, _g))])))',
            "gold_call": 'np.sort(np.abs(np.array([np.trace(M) for M in _unpack(_oracle_valley_representation(_b, _s, _g))])))',
        },  # boundary, the multiset of traces is invariant under the basis phase convention
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\ndef _unpack(v, n):\n    v = np.asarray(v, float).ravel(); h = v.size//2\n    return (v[:h] + 1j*v[h:]).reshape(n)\n_B = _unpack(_b, (2, 8))\n_J = np.array([(_B[0] - _B[1])/(1j*np.sqrt(2)), (_B[0] + _B[1])/np.sqrt(2)]).ravel()\n_alt = np.concatenate([_J.real, _J.imag])",
            "call": 'valley_representation(_alt, _s, _g)',
            "gold_call": '_oracle_valley_representation(_alt, _s, _g)',
        },  # edge, an alternative basis of the same subspace must still be represented from overlaps
    ]
