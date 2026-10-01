"""
Extract the source Pauli commutator and its PSC parity relation.

The controlled-error propagation theorem uses Q' = U R U^dagger R^dagger and the fact that the resulting Pauli has a definite commute/anticommute relation with the PSC. This step returns both pieces in a compact integer representation.

Returns
-------
np.ndarray: length-(2+2k) vector [q_phase, q_x..., q_z..., parity]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def psc_commutator_signature(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    '''Return Q' together with whether Q' commutes or anticommutes with U.

    Parameters
    ----------
    p_code, q_codes
        Canonical PSC Pauli data in the same encoding as earlier steps.
    pauli_code : np.ndarray
        Encoded target Pauli R.

    Returns
    -------
    np.ndarray
        Length-(2+2k) integer vector [q_phase, q_x..., q_z..., parity], where
        parity is 0 if U Q' U^dagger = Q' and 1 if it equals -Q'.
    '''
    return signature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _psc2_relation_with_u(p_code, q_codes, pauli_code):
    r = np.asarray(pauli_code, dtype=int)
    image = _oracle_psc_conjugate_pauli(p_code, q_codes, r)
    if not np.array_equal(image[1:], r[1:]):
        raise ValueError('Pauli is not a +/- eigenoperator of conjugation by this PSC')
    delta = (int(image[0]) - int(r[0])) % 4
    if delta not in (0, 2):
        raise ValueError('unexpected Pauli phase relation')
    return delta // 2


def _oracle_psc_commutator_signature(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    r = np.asarray(pauli_code, dtype=int)
    ur = _oracle_psc_conjugate_pauli(p_code, q_codes, r)
    qprime = _psc2_mul(ur, _psc2_dagger(r))
    parity = _psc2_relation_with_u(p_code, q_codes, qprime)
    return np.concatenate((qprime, [parity])).astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    hcase = '''import numpy as np\np_code=np.array([0,1,0],int); q_codes=np.array([[3,1,1]],int)'''
    return [
        {
            'setup': hcase + '\npauli_code=np.array([0,1,0],int)',
            'call': 'psc_commutator_signature(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'gold_call': '_oracle_psc_commutator_signature(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'tol': 0,
        },
        {
            'setup': '''import numpy as np\np_code=np.array([0,1,0,0,0],int); q_codes=np.array([[3,1,0,1,0],[2,0,0,0,1]],int); pauli_code=np.array([0,0,1,0,0],int)''',
            'call': 'psc_commutator_signature(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'gold_call': '_oracle_psc_commutator_signature(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'tol': 0,
        },
        {
            'setup': '''import numpy as np\np_code=np.array([1,1,0,1,0,0,1],int); q_codes=np.array([[2,0,0,0,1,0,0],[0,0,0,0,0,1,0],[2,0,0,0,0,0,1],[0,0,0,0,1,1,0]],int); pauli_code=np.array([1,1,0,1,1,0,1],int)''',
            'call': 'psc_commutator_signature(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'gold_call': '_oracle_psc_commutator_signature(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'tol': 0,
        },
    ]
