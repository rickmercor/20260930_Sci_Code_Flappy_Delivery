"""
Factor an arbitrary Pauli fault propagated through a controlled PSC.

The source theorem covers all four Pauli possibilities on the control, including the nontrivial X/Y branches that generate both a controlled-Pauli factor and a residual target PSC. This step returns the theorem factorization without building the controlled unitary as a dense matrix.

Returns
-------
np.ndarray: compact integer factorization of the propagated controlled-PSC fault.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def controlled_psc_fault_factor(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", control_label: int, target_code: "np.ndarray") -> "np.ndarray":
    '''Return a compact factorization of G F G^dagger for G=controlled-U.

    The returned factors use the fixed product order
    P_prime * controlled(Q) * (I_control tensor U)^b.

    Parameters
    ----------
    alpha_power : int
        Integer eighth-root phase exponent of the canonical PSC U.
    p_code, q_codes
        Canonical PSC Pauli data.
    control_label : int
        Fault Pauli on the fresh control: 0=I, 1=X, 2=Y, 3=Z.
    target_code : np.ndarray
        Length-(1+2k) encoded Pauli fault on the target register.

    Returns
    -------
    np.ndarray
        Integer vector [control_label, P'_target_code, Q_code, b]. Its length is
        2*(1+2k)+2; b is 0 or 1.
    '''
    return factor_code

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _psc2_conjugate_by_pauli(a, b):
    return _psc2_mul(_psc2_dagger(a), _psc2_mul(b, a))


def _oracle_controlled_psc_fault_factor(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", control_label: int, target_code: "np.ndarray") -> "np.ndarray":
    r = np.asarray(target_code, dtype=int)
    sig = _oracle_psc_commutator_signature(p_code, q_codes, r)
    qprime = sig[:-1]
    if int(control_label) in (0, 3):
        ptarget = r.copy()
        q = _psc2_conjugate_by_pauli(r, qprime)
        b = 0
    else:
        ptarget = _oracle_psc_conjugate_pauli(p_code, q_codes, r)
        u2 = _oracle_psc_square_pauli(alpha_power, p_code, q_codes)
        q = _psc2_mul(qprime, _psc2_dagger(u2))
        b = 1
    return np.concatenate(([int(control_label)], ptarget, q, [b])).astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    hs = '''import numpy as np\nalpha_power=1; p_code=np.array([0,1,0,0,0],int); q_codes=np.array([[3,1,0,1,0],[2,0,0,0,1]],int)'''
    return [
        {
            'setup': hs + '\ncontrol_label=0; target_code=np.array([0,1,0,0,0],int)',
            'call': 'controlled_psc_fault_factor(alpha_power,p_code.copy(),q_codes.copy(),control_label,target_code.copy())',
            'gold_call': '_oracle_controlled_psc_fault_factor(alpha_power,p_code.copy(),q_codes.copy(),control_label,target_code.copy())',
            'tol': 0,
        },
        {
            'setup': hs + '\ncontrol_label=1; target_code=np.array([0,1,0,0,0],int)',
            'call': 'controlled_psc_fault_factor(alpha_power,p_code.copy(),q_codes.copy(),control_label,target_code.copy())',
            'gold_call': '_oracle_controlled_psc_fault_factor(alpha_power,p_code.copy(),q_codes.copy(),control_label,target_code.copy())',
            'tol': 0,
        },
        {
            'setup': hs + '\ncontrol_label=2; target_code=np.array([1,1,1,1,1],int)',
            'call': 'controlled_psc_fault_factor(alpha_power,p_code.copy(),q_codes.copy(),control_label,target_code.copy())',
            'gold_call': '_oracle_controlled_psc_fault_factor(alpha_power,p_code.copy(),q_codes.copy(),control_label,target_code.copy())',
            'tol': 0,
        },
    ]
