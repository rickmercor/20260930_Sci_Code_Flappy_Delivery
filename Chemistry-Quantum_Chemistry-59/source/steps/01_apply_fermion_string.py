"""
Apply an ordered fermionic operator string in factorized spin channels.

Fermionic creation and annihilation operators acquire a sign from the parity of occupied orbitals preceding the orbital being modified. In this factorized representation, alpha and beta occupations are stored in separate spatial-orbital bitmasks. Parity must therefore be accumulated independently inside each spin mask; occupations of the opposite spin do not contribute.

Returns
-------
An integer array [alpha_mask, beta_mask, phase]; a Pauli-forbidden action returns [-1, -1, 0].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_fermion_string(
    determinant: 'np.ndarray',
    annihilators: 'np.ndarray',
    creators: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':
    """Apply annihilation and creation operators in chronological action order.

    A determinant stores separate alpha and beta spatial-orbital masks. Input
    operator labels use alpha ``2*i`` and beta ``2*i+1`` only as an integral-
    tensor lookup convention. Fermionic parity is accumulated independently
    inside each spin mask: occupied orbitals of the opposite spin never enter
    a parity count. The entries of ``annihilators`` act first from left to
    right, followed by ``creators`` from left to right.

    Returns
    -------
    np.ndarray
        ``[alpha_mask, beta_mask, phase]``. A Pauli-forbidden action returns
        ``[-1, -1, 0]``.

    Raises
    ------
    ValueError
        If masks, orbital indices, dimensions or dtypes are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_apply_fermion_string(
    determinant: 'np.ndarray',
    annihilators: 'np.ndarray',
    creators: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':

    if not isinstance(n_orb, (int, np.integer)) or not 1 <= int(n_orb) <= 31:
        raise ValueError("n_orb must be an integer in [1,31]")

    n_orb = int(n_orb)
    det = np.asarray(determinant)
    ann = np.asarray(annihilators)
    cre = np.asarray(creators)

    if det.shape != (2,) or ann.ndim != 1 or cre.ndim != 1:
        raise ValueError("invalid determinant or operator shape")
    if not np.issubdtype(det.dtype, np.integer):
        raise ValueError("determinant masks must be integers")
    if not np.issubdtype(ann.dtype, np.integer) or not np.issubdtype(
        cre.dtype, np.integer
    ):
        raise ValueError("operator indices must be integers")

    alpha, beta = map(int, det)
    if (
        alpha < 0
        or beta < 0
        or alpha >= (1 << n_orb)
        or beta >= (1 << n_orb)
    ):
        raise ValueError("determinant mask is outside n_orb")

    if (
        np.any(ann < 0)
        or np.any(ann >= 2 * n_orb)
        or np.any(cre < 0)
        or np.any(cre >= 2 * n_orb)
    ):
        raise ValueError("spin-orbital index outside range")

    phase = 1

    for q in ann:
        q = int(q)
        orbital, spin = q // 2, q & 1
        mask = alpha if spin == 0 else beta
        bit = 1 << orbital

        if (mask & bit) == 0:
            return np.array([-1, -1, 0], dtype=np.int64)

        if (mask & (bit - 1)).bit_count() & 1:
            phase = -phase

        mask ^= bit
        if spin == 0:
            alpha = mask
        else:
            beta = mask

    for p in cre:
        p = int(p)
        orbital, spin = p // 2, p & 1
        mask = alpha if spin == 0 else beta
        bit = 1 << orbital

        if mask & bit:
            return np.array([-1, -1, 0], dtype=np.int64)

        if (mask & (bit - 1)).bit_count() & 1:
            phase = -phase

        mask |= bit
        if spin == 0:
            alpha = mask
        else:
            beta = mask

    return np.array([alpha, beta, phase], dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nd=np.array([3,1],dtype=np.int64); a=np.array([0]); c=np.array([4])",
            "call": "apply_fermion_string(d.copy(),a.copy(),c.copy(),4)",
            "gold_call": "_oracle_apply_fermion_string(d,a,c,4)",
        },
        {
            "setup": "import numpy as np\nd=np.array([3,3],dtype=np.int64); a=np.array([0,3]); c=np.array([5,6])",
            "call": "apply_fermion_string(d.copy(),a.copy(),c.copy(),4)",
            "gold_call": "_oracle_apply_fermion_string(d,a,c,4)",
        },
        {
            "setup": "import numpy as np\nd=np.array([5,2],dtype=np.int64); a=np.array([6]); c=np.array([0])",
            "call": "apply_fermion_string(d.copy(),a.copy(),c.copy(),4)",
            "gold_call": "_oracle_apply_fermion_string(d,a,c,4)",
        },
        {
            "setup": "import numpy as np\nd=np.array([5,3],dtype=np.int64); a=np.array([0,4]); c=np.array([6,2])",
            "call": "apply_fermion_string(d.copy(),a.copy(),c.copy(),4)",
            "gold_call": "_oracle_apply_fermion_string(d,a,c,4)",
        },
        {
            "setup": "import numpy as np\nd=np.array([1,0],dtype=np.int64); a=np.array([],dtype=np.int64); c=np.array([0],dtype=np.int64)",
            "call": "apply_fermion_string(d.copy(),a.copy(),c.copy(),2)",
            "gold_call": "_oracle_apply_fermion_string(d,a,c,2)",
        },
        {
            "setup": "import numpy as np\nd=np.array([1,0],dtype=np.int64); a=np.array([4]); c=np.array([],dtype=np.int64)\ndef check(fn):\n try: fn(d.copy(),a.copy(),c.copy(),2)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(apply_fermion_string)",
            "gold_call": "check(_oracle_apply_fermion_string)",
        },
    ]
