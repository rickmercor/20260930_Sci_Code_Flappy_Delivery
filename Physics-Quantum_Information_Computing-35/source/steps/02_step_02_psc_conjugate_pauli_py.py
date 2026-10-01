"""
Conjugate a Pauli by a canonically represented Pauli-square-root Clifford.

Use the same source canonical PSC equation and order as Step 01,



$$

U=\alpha P\exp\!\left(\frac{\pi i}{4}\sum_j Q_j\right)

=\alpha P\prod_j\exp\!\left(\frac{\pi iQ_j}{4}\right).

$$



The rotation sign is positive and $P$ is the left prefactor. For a Hermitian Pauli $Q_j$, the positive quarter-turn acts on a Pauli $R$ as



$$

\exp\!\left(\frac{\pi iQ_j}{4}\right)R

\exp\!\left(-\frac{\pi iQ_j}{4}\right)

=

\begin{cases}

R,& Q_jR=RQ_j,\\

iQ_jR,& Q_jR=-RQ_j.

\end{cases}

$$



After all commuting quarter-turns, conjugation by $P$ supplies the final sign. This step performs that action directly in binary Pauli-group arithmetic.

Returns
-------
np.ndarray: length-(1+2k) integer Pauli-group code for U R U^dagger.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def psc_conjugate_pauli(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    '''Return the encoded Pauli U R U^dagger, omitting the irrelevant alpha phase.

    Parameters
    ----------
    p_code : np.ndarray
        Length-(1+2k) encoded Pauli prefactor P.
    q_codes : np.ndarray
        Shape-(m,1+2k) mutually commuting Hermitian Pauli codes Q_j for
        U = alpha * P * exp(+pi*i/4 * sum_j Q_j), with this exact positive
        rotation sign and factor order.
    pauli_code : np.ndarray
        Length-(1+2k) Pauli-group code for R.

    Returns
    -------
    np.ndarray
        Length-(1+2k) integer code for U R U^dagger.
    '''
    return conjugated_code

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_psc_conjugate_pauli(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    p = np.asarray(p_code, dtype=int)
    out = np.array(pauli_code, dtype=int, copy=True)
    for q in np.asarray(q_codes, dtype=int):
        if _psc2_symp(q, out):
            out = _psc2_mul(q, out)
            out[0] = (int(out[0]) + 1) % 4
    if _psc2_symp(p, out):
        out[0] = (int(out[0]) + 2) % 4
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = '''import numpy as np\np_code=np.array([0,1,0,0,0],int); q_codes=np.array([[3,1,0,1,0],[2,0,0,0,1]],int)'''
    return [
        {
            'setup': common + '\npauli_code=np.array([0,1,0,0,0],int)',
            'call': 'psc_conjugate_pauli(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'gold_call': '_oracle_psc_conjugate_pauli(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'tol': 0,
        },
        {
            'setup': common + '\npauli_code=np.array([1,1,0,1,0],int)',
            'call': 'psc_conjugate_pauli(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'gold_call': '_oracle_psc_conjugate_pauli(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'tol': 0,
        },
        {
            'setup': '''import numpy as np\np_code=np.array([1,1,0,1,0,0,1],int); q_codes=np.array([[2,0,0,0,1,0,0],[0,0,0,0,0,1,0],[2,0,0,0,0,0,1],[0,0,0,0,1,1,0]],int); pauli_code=np.array([2,1,1,0,0,1,1],int)''',
            'call': 'psc_conjugate_pauli(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'gold_call': '_oracle_psc_conjugate_pauli(p_code.copy(),q_codes.copy(),pauli_code.copy())',
            'tol': 0,
        },
    ]
