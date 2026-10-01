"""
Project the plane-wave shell onto the X1 representation and impose compatibility with the bands running from X to the zone centre, returning the two valley basis functions.

Only some of the irreducible representations of the group built in step 02 can describe a Bloch
function at X. The reason is the face-diagonal translation Q. Acting on a Bloch function that
carries the X-point phase, Q returns the function multiplied by a definite factor fixed by the
periodicity of the lattice-periodic part, and that factor is not unity. Representations whose
character under Q has the wrong sign therefore cannot host a Bloch state at X. The survivors are
all two-dimensional, which is why the silicon conduction band is doubly degenerate at X and why
the two valleys exist at all.

The conduction band of silicon belongs to the representation conventionally called X1. Its two
basis functions must be extracted from the plane-wave shell of step 01 by the projection theorem,
that is, by summing the group operations weighted by the X1 characters. Only four symmorphic
operations of the group carry a non-zero X1 character, together with their partners obtained by
composing with Q, so the projector collapses to a short sum acting on a seed plane wave.

Projection alone fixes the two-dimensional subspace but not the basis inside it, which is free up
to a unitary rotation. The physical basis is fixed by a further requirement that is easy to state
and easy to get wrong: the pair must be compatible with the one-dimensional bands along the line
joining X to the zone centre. No operation of the little group of that line can reverse the sign
of z, so the z dependence of a state cannot be mixed by those operations; a state that is to
connect continuously onto one of the two bands along that line must therefore carry a single
value of the z component of its plane-wave index. Imposing that condition picks a unique pair,
up to ordering and an overall phase, and it is this pair, not any other unitary combination of
it, that represents the two valleys.

Return the two basis vectors as coefficient rows over the eight plane waves of step 01, in the
same lexicographic order those plane waves were returned in. Order the two rows so that the first
carries the plane waves with z index -1 and the second those with z index +1. Normalise each row
to unit length and fix the residual phase by making the first coefficient of largest modulus in
each row real and positive.

Returns
-------
np.ndarray of length 32: real parts of the 2x8 coefficient matrix row-major, then imaginary parts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def valleyor_basis(shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    '''The two X1 basis functions expressed over the plane-wave shell.

    Parameters
    ----------
    shell : np.ndarray
        Output of step 01 for the eight-fold shell: length-24 real array of plane-wave triples.
    group : np.ndarray
        Output of step 02: length-384 real array holding the thirty-two operations.

    Returns
    -------
    result : np.ndarray
        Real array of length 32: the real parts of the two-by-eight coefficient matrix flattened
        row-major, followed by its imaginary parts in the same order.

    Raises
    ------
    ValueError
        If the supplied shell does not contain exactly eight plane waves.
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


def _op_matrix(R, t, P, idx):
    n = len(P)
    M = np.zeros((n, n), dtype=complex)
    for j, p in enumerate(P):
        M[idx[tuple((np.asarray(R).T @ p).astype(int))], j] = np.exp(-1j * float(np.dot(p, t)))
    return M


def _oracle_valleyor_basis(shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    P = _shell_from_flat(shell)
    n = len(P)
    if n != 8:
        raise ValueError("the X1 construction expects the eight-fold shell of step 01")
    rot, tra = _group_from_flat(group)
    idx = {tuple(p): i for i, p in enumerate(P)}
    keep = {((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            ((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (-1, 0, 0), (0, 0, 1))}
    S = np.zeros((n, n), dtype=complex)
    Qm = None
    for R, t in zip(rot, tra):
        zero_t = bool(np.allclose(t, 0.0))
        if tuple(map(tuple, R.tolist())) in keep and zero_t:
            S = S + _op_matrix(R, t, P, idx)
        if np.array_equal(R, np.eye(3, dtype=int)) and not zero_t:
            Qm = _op_matrix(R, t, P, idx)
    Pi = S @ (np.eye(n) - Qm)
    U, sv, _ = np.linalg.svd(Pi)
    space = U[:, : int((sv > 1e-9).sum())]
    rows = []
    for p3 in (-1, +1):
        ind = (P[:, 2] == p3).astype(float)
        v = space @ (space.conj().T @ ind)
        v = v / np.linalg.norm(v)
        first = int(np.argmax(np.abs(v) > 1e-9))
        rows.append(v * np.exp(-1j * np.angle(v[first])))
    z = np.array(rows).ravel()
    return np.concatenate([z.real, z.imag]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)",
            "call": 'valleyor_basis(_s, _g)',
            "gold_call": '_oracle_valleyor_basis(_s, _g)',
        },  # normal, the valley doublet of the silicon conduction band
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 6)\n_g = _oracle_wave_vector_group_at_X(2)",
            "call": 'valleyor_basis(_s, _g)',
            "gold_call": '_oracle_valleyor_basis(_s, _g)',
        },  # boundary, a wider index search must reproduce the same shell and basis
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)",
            "call": '(lambda v: np.array([float(np.abs(np.reshape(v[:16], (2, 8)) + 1j*np.reshape(v[16:], (2, 8))).sum())]))(valleyor_basis(_s, _g))',
            "gold_call": '(lambda v: np.array([float(np.abs(np.reshape(v[:16], (2, 8)) + 1j*np.reshape(v[16:], (2, 8))).sum())]))(_oracle_valleyor_basis(_s, _g))',
        },  # boundary, total weight distinguishes the valley basis from other unitary choices
        {
            "setup": 'import numpy as np\n_g = _oracle_wave_vector_group_at_X(2)\n_s1 = _oracle_plane_wave_shell_at_X(1, 3)\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: valleyor_basis(_s1, _g))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_valleyor_basis(_s1, _g))])',
        },  # contract, a shell that is not eight-fold must raise ValueError
    ]
