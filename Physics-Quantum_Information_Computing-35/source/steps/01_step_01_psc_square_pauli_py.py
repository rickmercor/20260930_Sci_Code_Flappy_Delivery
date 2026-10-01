"""
Recover the Pauli square of a Pauli-square-root Clifford from canonical data.

Use the source canonical PSC equation



$$

U=\alpha P\exp\!\left(\frac{\pi i}{4}\sum_j Q_j\right)

=\alpha P\prod_j\exp\!\left(\frac{\pi iQ_j}{4}\right),

$$



with factor order $\alpha$, then $P$, then the commuting positive quarter-turns. Here $\alpha=\exp(i\pi a/4)$, where $a$ is $alpha_power$, and the $Q_j$ are mutually commuting Hermitian Paulis. The plus sign in the rotation exponent is part of the convention. Squaring this ordered representation can be done entirely in Pauli-group arithmetic, avoiding any dense $2^k\times2^k$ unitary construction.

Returns
-------
np.ndarray: length-(1+2k) integer Pauli-group code for U^2, with phase modulo 4
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def psc_square_pauli(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray") -> "np.ndarray":
    '''Return the encoded Pauli equal to U^2 for the canonical PSC U.

    Parameters
    ----------
    alpha_power : int
        Integer a in {0,...,7} for alpha = exp(i*pi*a/4).
    p_code : np.ndarray
        Length-(1+2k) integer Pauli code [phase, x..., z...] for P, using
        i^phase X^x Z^z.
    q_codes : np.ndarray
        Integer array of shape (m, 1+2k) containing mutually commuting
        Hermitian Pauli codes Q_j. Use the canonical convention
        U = alpha * P * exp(+pi*i/4 * sum_j Q_j), with P to the left of
        the positive quarter-turn exponential.

    Returns
    -------
    np.ndarray
        Length-(1+2k) integer Pauli-group code for U^2, with phase modulo 4.
    '''
    return square_code

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _psc2_mul(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = (a.size - 1) // 2
    phase = (int(a[0]) + int(b[0]) + 2 * int(np.dot(a[1+k:], b[1:1+k]) % 2)) % 4
    x = np.bitwise_xor(a[1:1+k], b[1:1+k])
    z = np.bitwise_xor(a[1+k:], b[1+k:])
    return np.concatenate(([phase], x, z)).astype(int)


def _psc2_dagger(a):
    a = np.asarray(a, dtype=int)
    k = (a.size - 1) // 2
    phase = (-int(a[0]) + 2 * int(np.dot(a[1:1+k], a[1+k:]) % 2)) % 4
    return np.concatenate(([phase], a[1:])).astype(int)


def _psc2_symp(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = (a.size - 1) // 2
    return int((np.dot(a[1:1+k], b[1+k:]) + np.dot(a[1+k:], b[1:1+k])) % 2)


def _psc2_labels_to_code(labels, extra_phase=0):
    labels = np.asarray(labels, dtype=int)
    x = np.isin(labels, [1, 2]).astype(int)
    z = np.isin(labels, [2, 3]).astype(int)
    phase = (int(extra_phase) + int(np.sum(labels == 2))) % 4
    return np.concatenate(([phase], x, z)).astype(int)


def _psc2_is_identity(code):
    code = np.asarray(code, dtype=int)
    return int(code[0]) % 4 == 0 and not np.any(code[1:])


def _oracle_psc_square_pauli(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray") -> "np.ndarray":
    p = np.asarray(p_code, dtype=int)
    qs = np.asarray(q_codes, dtype=int)
    out = np.zeros_like(p)
    out[0] = int(alpha_power) % 4  # alpha^2 = i^a
    out = _psc2_mul(out, _psc2_mul(p, p))
    for q in qs:
        if _psc2_symp(p, q) == 0:
            iq = np.array(q, dtype=int, copy=True)
            iq[0] = (int(iq[0]) + 1) % 4
            out = _psc2_mul(out, iq)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '''import numpy as np\nalpha_power=1; p_code=np.array([0,0,0,0,0],int); q_codes=np.array([[2,0,0,1,0],[2,0,0,0,1],[0,0,0,1,1]],int)''',
            'call': 'psc_square_pauli(alpha_power,p_code.copy(),q_codes.copy())',
            'gold_call': '_oracle_psc_square_pauli(alpha_power,p_code.copy(),q_codes.copy())',
            'tol': 0,
        },
        {
            'setup': '''import numpy as np\nalpha_power=0; p_code=np.array([0,1,0],int); q_codes=np.array([[3,1,1]],int)''',
            'call': 'psc_square_pauli(alpha_power,p_code.copy(),q_codes.copy())',
            'gold_call': '_oracle_psc_square_pauli(alpha_power,p_code.copy(),q_codes.copy())',
            'tol': 0,
        },
        {
            'setup': '''import numpy as np\nalpha_power=3; p_code=np.array([1,1,0,1,0,0,1],int); q_codes=np.array([[2,0,0,0,1,0,0],[0,0,0,0,0,1,0],[2,0,0,0,0,0,1],[0,0,0,0,1,1,0]],int)''',
            'call': 'psc_square_pauli(alpha_power,p_code.copy(),q_codes.copy())',
            'gold_call': '_oracle_psc_square_pauli(alpha_power,p_code.copy(),q_codes.copy())',
            'tol': 0,
        },
    ]
