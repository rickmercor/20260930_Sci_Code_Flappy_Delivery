"""
Return the sign of each valley Pauli matrix under each crystal operation together with its time-reversal parity.

The valley degree of freedom is described by the three Pauli matrices acting in the space spanned
by the two Bloch functions of step 03. It is tempting to treat these as if they were spin, but
they are not: their transformation law under the crystal symmetry is set by the two Bloch
functions themselves, and it differs from the spinor law. Establishing that law is the central
result this pipeline rests on.

Two independent pieces of information are needed about each valley Pauli matrix.

The first is its behaviour under the crystal operations. Conjugating a valley Pauli matrix by the
representation matrix of a group operation returns the same Pauli matrix multiplied by a sign,
and collecting those signs over all thirty-two operations assigns each valley operator a definite
symmetry character. Because conjugation cancels overall phases, this character is independent of
the phase convention adopted for the basis, and it is constant on each conjugacy class.

The second is behaviour under time reversal. For a spinless electron time reversal is complex
conjugation, which on plane waves sends the index triple p to its negative. Represented on the
valley doublet this becomes an antiunitary operation, a fixed unitary matrix followed by complex
conjugation, and conjugating each valley Pauli matrix through it again returns that matrix times
a sign. Here the contrast with spin is sharp and consequential: for spin every component is odd
under time reversal, whereas for the valley operators the parities are not all alike. Which of
them is odd is decided by the basis of step 03, and getting it wrong silently changes which
physical couplings are permitted later on.

This step returns both pieces: the sign of each of the three valley Pauli matrices under each of
the thirty-two operations, in the operation order of step 02, followed by the three time-reversal
parities.

Returns
-------
np.ndarray of length 99: 96 crystal signs (j slowest, operation fastest) then 3 time-reversal parities
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tau_symmetry_data(rep: np.ndarray, basis: np.ndarray, shell: np.ndarray) -> np.ndarray:
    '''Crystal signs and time-reversal parities of the three valley Pauli matrices.

    Parameters
    ----------
    rep : np.ndarray
        Output of step 04: length-256 real array packing the thirty-two representation matrices.
    basis : np.ndarray
        Output of step 03: length-32 real array packing the valley doublet.
    shell : np.ndarray
        Output of step 01 for the eight-fold shell: length-24 real array.

    Returns
    -------
    result : np.ndarray
        Real array of length 99. The first 96 entries are the signs s[j, g] for valley index
        j = 1, 2, 3 and operation g, flattened row-major with j varying slowest. The final three
        entries are the time-reversal parities of the three valley Pauli matrices in the same
        order. All entries are +1 or -1.

    Raises
    ------
    ValueError
        If some valley Pauli matrix is not mapped onto plus or minus itself, which signals that
        the supplied basis is not the X1 doublet.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _shell_from_flat(v):
    return np.rint(np.asarray(v, dtype=float)).astype(int).reshape(-1, 3)


def _unpack_complex(v, shape):
    v = np.asarray(v, dtype=float).ravel()
    h = v.size // 2
    return (v[:h] + 1j * v[h:]).reshape(shape)


def _oracle_tau_symmetry_data(rep: np.ndarray, basis: np.ndarray, shell: np.ndarray) -> np.ndarray:
    import numpy as np
    _TAU3 = [np.array([[0, 1], [1, 0]], dtype=complex),
             np.array([[0, -1j], [1j, 0]], dtype=complex),
             np.array([[1, 0], [0, -1]], dtype=complex)]
    D = _unpack_complex(rep, (32, 2, 2))
    P = _shell_from_flat(shell)
    n = len(P)
    B = _unpack_complex(basis, (2, n))
    signs = np.zeros((3, 32))
    for g in range(32):
        for j in range(3):
            Tj = D[g] @ _TAU3[j] @ D[g].conj().T
            s = float(np.trace(_TAU3[j].conj().T @ Tj).real) / 2.0
            if abs(abs(s) - 1.0) > 1e-6:
                raise ValueError("a valley Pauli matrix is not mapped onto plus or minus itself")
            signs[j, g] = round(s)
    idx = {tuple(p): i for i, p in enumerate(P)}
    Pi = np.zeros((n, n))
    for i, p in enumerate(P):
        Pi[idx[tuple(-p)], i] = 1.0
    U = B.conj() @ (Pi @ B.conj().T)
    tr = [round(float(np.trace(_TAU3[j].conj().T @ (U @ _TAU3[j].conj() @ U.conj().T)).real) / 2.0)
          for j in range(3)]
    return np.concatenate([signs.ravel(), np.array(tr, dtype=float)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\n_r = _oracle_valley_representation(_b, _s, _g)",
            "call": 'tau_symmetry_data(_r, _b, _s)',
            "gold_call": '_oracle_tau_symmetry_data(_r, _b, _s)',
        },  # normal, crystal signs and time-reversal parities of the valley operators
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\n_r = _oracle_valley_representation(_b, _s, _g)",
            "call": 'tau_symmetry_data(_r, _b, _s)[96:]',
            "gold_call": '_oracle_tau_symmetry_data(_r, _b, _s)[96:]',
        },  # boundary, the time-reversal parities alone, which the spin analogy gets wrong
        {
            "setup": "import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\n_r = _oracle_valley_representation(_b, _s, _g)",
            "call": '(lambda v: np.concatenate([(v[:96].reshape(3, 32) * np.arange(1, 33)).sum(axis=1), [float((v[:96].reshape(3, 32).prod(axis=0) > 0).sum())]]))(tau_symmetry_data(_r, _b, _s))',
            "gold_call": '(lambda v: np.concatenate([(v[:96].reshape(3, 32) * np.arange(1, 33)).sum(axis=1), [float((v[:96].reshape(3, 32).prod(axis=0) > 0).sum())]]))(_oracle_tau_symmetry_data(_r, _b, _s))',
        },  # boundary, position-weighted characters separate the three symmetry types
        {
            "setup": 'import numpy as np\n_s = _oracle_plane_wave_shell_at_X(5, 4)\n_g = _oracle_wave_vector_group_at_X(2)\n_b = _oracle_valleyor_basis(_s, _g)\n_r = _oracle_valley_representation(_b, _s, _g)\n_bad = _r.copy()\n_bad[:4] = _bad[:4] + 0.37\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: tau_symmetry_data(_bad, _b, _s))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_tau_symmetry_data(_bad, _b, _s))])',
        },  # contract, a representation that is not the X1 doublet must raise ValueError
    ]
