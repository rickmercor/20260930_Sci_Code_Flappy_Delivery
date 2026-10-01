"""
Build the thirty-two operations of the group of the wave vector at X as point matrices with their translations, in a fixed deterministic order.

The diamond structure is non-symmorphic: its space group contains operations that combine a
point operation with a fractional translation that is not a lattice vector. The relevant one is
the quarter translation along the body diagonal, T = a(1, 1, 1)/4, which carries one atom of the
two-atom basis onto the other. Combining T with the mirror in the z = 0 plane produces a glide
operation, and it is the glide that makes the X point special.

The point group of the X point is D2d, of order eight. Adjoining the glide doubles this to
sixteen, but that set is not yet closed: the glide fails to commute with the two-fold axes, and
the mismatch is exactly a face-diagonal translation Q = a(0, -1, -1)/2. Q is a lattice vector of
the FCC lattice, so it acts trivially on the crystal, yet on a Bloch function carrying the phase
of the X point it does not act trivially at all. Including Q therefore doubles the count once
more, and the resulting set of thirty-two operations closes. This is the group of the wave vector
at X, and the fact that Q survives in it as a non-trivial element is what forces every physically
admissible Bloch representation at X to be two-dimensional.

There are three X points, along x, y and z. The threefold rotation about the body diagonal maps the
diamond structure onto itself with the origin on an atom, so it carries the group of one X point
onto the group of the next. The task works at the X point along z, cubic axis 2, but the routine
must return the group for whichever cubic axis is requested, rotating both the point matrices and
the translations; any other axis index is an error.

This step builds those thirty-two operations as pairs {R | t}, with R the three-by-three integer
point matrix acting on the coordinate column (x, y, z) and t the accompanying translation. Work
in units where a = 2*pi, so that T becomes (pi/2)(1, 1, 1) and Q becomes pi(0, -1, -1); the
translation is reported in units of pi. A deterministic ordering is imposed so that downstream
steps can index the operations unambiguously: sort ascending by the nine entries of R read
row-major, breaking ties on the three translation components rounded to six decimals.

Returns
-------
np.ndarray of length 384: 288 rotation entries followed by 96 translation entries in units of pi
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wave_vector_group_at_X(axis: int) -> np.ndarray:
    '''The thirty-two operations {R | t} of the group of the wave vector at the X point.

    Parameters
    ----------
    axis : int
        Cubic axis of the X point, 0 for x, 1 for y and 2 for z.

    Returns
    -------
    result : np.ndarray
        Real array of length 384. The first 288 entries are the thirty-two point matrices R
        flattened row-major in the sorted order described above; the remaining 96 entries are
        the matching translations t in units of pi, flattened row-major in the same order.

    Raises
    ------
    ValueError
        If axis is not 0, 1 or 2, or if the assembled complex does not close into thirty-two
        distinct operations.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_wave_vector_group_at_X(axis: int) -> np.ndarray:
    import numpy as np
    if axis not in (0, 1, 2) or isinstance(axis, bool):
        raise ValueError("axis must be 0, 1 or 2")
    d2d = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],        # E
           [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],      # C2 about z
           [[1, 0, 0], [0, -1, 0], [0, 0, -1]],      # C2 about x
           [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],      # C2 about y
           [[0, -1, 0], [1, 0, 0], [0, 0, -1]],      # S4z
           [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],      # S4z inverse
           [[0, 1, 0], [1, 0, 0], [0, 0, 1]],        # diagonal mirror m1
           [[0, -1, 0], [-1, 0, 0], [0, 0, 1]]]      # diagonal mirror m2
    mz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=int)
    T = np.array([0.5, 0.5, 0.5])                    # a(1,1,1)/4 in units of pi
    Q = np.array([0.0, -1.0, -1.0])                  # a(0,-1,-1)/2 in units of pi
    rot, tra = [], []
    for qt in (np.zeros(3), Q):
        for gR, gt in ((np.eye(3, dtype=int), np.zeros(3)), (mz, T)):
            for d in d2d:
                rot.append((gR @ np.array(d, dtype=int)).astype(int))
                tra.append(gt + qt)
    rot = np.array(rot, dtype=int)
    tra = np.array(tra, dtype=float)
    # threefold rotation about [111] taking the z axis onto the requested axis
    C = np.linalg.matrix_power(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int), (axis + 1) % 3)
    rot = np.array([C @ r @ C.T for r in rot], dtype=int)
    tra = np.array([C @ t for t in tra], dtype=float)
    if len({(tuple(r.ravel().tolist()), tuple(np.round(t, 6).tolist()))
            for r, t in zip(rot, tra)}) != 32:
        raise ValueError("the assembled complex does not close into thirty-two distinct operations")
    order = sorted(range(32), key=lambda i: (tuple(rot[i].ravel().tolist()),
                                             tuple(np.round(tra[i], 6).tolist())))
    return np.concatenate([rot[order].ravel().astype(float), tra[order].ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": 'wave_vector_group_at_X(2)',
            "gold_call": '_oracle_wave_vector_group_at_X(2)',
        },  # normal, the full ordered list of operations
        {
            "setup": "import numpy as np",
            "call": 'np.array([float(np.rint(wave_vector_group_at_X(2)[:288]).reshape(32,3,3).shape[0])])',
            "gold_call": 'np.array([float(np.rint(_oracle_wave_vector_group_at_X(2)[:288]).reshape(32,3,3).shape[0])])',
        },  # boundary, the order of the group is thirty-two
        {
            "setup": "import numpy as np",
            "call": 'np.sort(np.array([float(round(np.linalg.det(R))) for R in np.rint(wave_vector_group_at_X(2)[:288]).reshape(32,3,3)]))',
            "gold_call": 'np.sort(np.array([float(round(np.linalg.det(R))) for R in np.rint(_oracle_wave_vector_group_at_X(2)[:288]).reshape(32,3,3)]))',
        },  # boundary, proper and improper operations are equinumerous
        {
            "setup": "import numpy as np",
            "call": 'np.unique(np.round(wave_vector_group_at_X(2)[288:].reshape(32,3), 6), axis=0).ravel()',
            "gold_call": 'np.unique(np.round(_oracle_wave_vector_group_at_X(2)[288:].reshape(32,3), 6), axis=0).ravel()',
        },  # edge, exactly four distinct translations occur, including the face-diagonal one
        {
            "setup": "import numpy as np",
            "call": 'wave_vector_group_at_X(0)',
            "gold_call": '_oracle_wave_vector_group_at_X(0)',
        },  # normal, the X point along x, reached by the threefold rotation about the body diagonal
        {
            "setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: wave_vector_group_at_X(3)), _raises(lambda: wave_vector_group_at_X(-1))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_wave_vector_group_at_X(3)), _raises(lambda: _oracle_wave_vector_group_at_X(-1))])',
        },  # contract, an axis index outside 0, 1, 2 must raise ValueError
    ]
