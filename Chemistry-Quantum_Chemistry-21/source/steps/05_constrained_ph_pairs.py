"""
Return the particle-hole space of the source's constrained RPA (its Eq. (2.14)) for n_orb spatial orbitals of which the first n_occ are doubly occupied and the orbitals listed in active form the active space, as an integer array of spin-orbital index pairs (i, a). Spin-orbitals are indexed 2p for the spin-up and 2p + 1 for the spin-down function of spatial orbital p; a pair is an occupied spin-orbital i and a virtual spin-orbital a of the same spin. List the pairs the source keeps in the constrained polarisability, ordered by i ascending and, within the same i, by a ascending.

Constrained RPA builds the polarisability that screens the active-space interaction from the particle-hole excitations the active-space solver will not treat explicitly; which excitations are kept is a definition the source makes once, and in a spin-orbital formulation each spatial excitation appears once per spin.

Returns
-------
numpy.ndarray of int64 with shape (m, 2), m the number of pairs kept.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constrained_ph_pairs(n_orb: int, n_occ: int, active: "list[int]") -> "np.ndarray":
    """Return the particle-hole space of the source's constrained RPA (its Eq. (2.14)) for n_orb spatial orbitals of which the first n_occ are doubly occupied and the orbitals listed in active form the active space, as an integer array of spin-orbital index pairs (i, a). Spin-orbitals are indexed 2p for the spin-up and 2p + 1 for the spin-down function of spatial orbital p; a pair is an occupied spin-orbital i and a virtual spin-orbital a of the same spin. List the pairs the source keeps in the constrained polarisability, ordered by i ascending and, within the same i, by a ascending.

    Parameters
    ----------
    n_orb : int
        Number of spatial orbitals (at least 2).
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= n_orb - 1; they are orbitals 0 to n_occ - 1.
    active : list[int]
        Distinct indices in [0, n_orb - 1] of the active spatial orbitals (non-empty).

    Returns
    -------
    pairs : numpy.ndarray
        Integer array of shape (m, 2); row k = (i, a) is the k-th kept spin-orbital particle-hole pair.

    Raises
    ------
    ValueError
        If n_orb is below 2, n_occ is not in [1, n_orb - 1], or active is empty, has repeated entries or entries outside [0, n_orb - 1].
    """
    return pairs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def _ph_pairs(n_occ, n_orb):
    """same-spin particle-hole pairs of spin-orbitals (2p = alpha, 2p+1 = beta), i outer, a inner."""
    occ = range(2 * n_occ)
    virt = range(2 * n_occ, 2 * n_orb)
    return [(i, a) for i in occ for a in virt if i % 2 == a % 2]


def _oracle_constrained_ph_pairs(n_orb: int, n_occ: int, active: "list[int]") -> "np.ndarray":
    n_orb = _check_int(n_orb, "n_orb", 2)
    n_occ = _check_int(n_occ, "n_occ", 1)
    if n_occ >= n_orb:
        raise ValueError("n_occ must be smaller than n_orb")
    act = set(_check_active(active, n_orb))
    # cRPA, Eq. (2.14): every same-spin spin-orbital particle-hole pair except those with both orbitals
    # in the active space (their rows and columns of A +/- B are zero and the modes drop out of W)
    pairs = [(i, a) for (i, a) in _ph_pairs(n_occ, n_orb) if not (i // 2 in act and a // 2 in act)]
    return np.array(pairs, dtype=np.int64).reshape(len(pairs), 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "n_orb, n_occ, active = 6, 3, [2, 3]\n",
            "call": "constrained_ph_pairs(n_orb, n_occ, active)",
            "gold_call": "_oracle_constrained_ph_pairs(n_orb, n_occ, active)",
        },
        {
            "setup": "n_orb, n_occ, active = 8, 4, [2, 3, 4, 5]\n",
            "call": "constrained_ph_pairs(n_orb, n_occ, active)",
            "gold_call": "_oracle_constrained_ph_pairs(n_orb, n_occ, active)",
        },
        {
            "setup": "n_orb, n_occ, active = 10, 5, [3, 4, 5, 6]\n",
            "call": "constrained_ph_pairs(n_orb, n_occ, active)",
            "gold_call": "_oracle_constrained_ph_pairs(n_orb, n_occ, active)",
        },
        {
            "setup": "n_orb, n_occ, active = 4, 2, [0, 1, 2, 3]\n",
            "call": "constrained_ph_pairs(n_orb, n_occ, active)",
            "gold_call": "_oracle_constrained_ph_pairs(n_orb, n_occ, active)",
        },
        {
            "setup": "n_orb, n_occ, active = 6, 3, [2, 2, 3]\ndef run_model():\n    try:\n        constrained_ph_pairs(n_orb, n_occ, active)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_constrained_ph_pairs(n_orb, n_occ, active)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
