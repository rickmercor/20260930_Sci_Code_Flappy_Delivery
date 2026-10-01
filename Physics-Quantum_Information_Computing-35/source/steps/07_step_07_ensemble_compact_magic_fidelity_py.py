"""
Combine exact compact-propagation branches into an accepted-ensemble magic-state fidelity.

The paper samples Pauli configurations, propagates each to an end Clifford, and renormalizes fidelity by the accepted-shot probability. This deterministic orchestrator replaces Monte Carlo sampling by a finite supplied probability distribution while retaining the same acceptance-weighted contraction.

Returns
-------
float: exact accepted-ensemble conditional fidelity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ensemble_compact_magic_fidelity(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedules: "np.ndarray", probabilities: "np.ndarray") -> float:
    '''Return the exact conditional fidelity of the accepted finite ensemble.

    Parameters
    ----------
    alpha_power, p_code, q_codes
        Canonical PSC data.
    target_state : np.ndarray
        Normalized k-qubit target state.
    fault_schedules : np.ndarray
        Shape-(N,r,k+1) collection of pre-gate Pauli schedules.
    probabilities : np.ndarray
        Length-N nonnegative schedule probabilities summing to one.

    Returns
    -------
    float
        Probability-weighted accepted target overlap divided by total acceptance.
    '''
    return fidelity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ensemble_compact_magic_fidelity(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedules: "np.ndarray", probabilities: "np.ndarray") -> float:
    schedules = np.asarray(fault_schedules, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    total_a = 0.0
    total_n = 0.0
    for schedule, prob in zip(schedules, probs):
        stats = _oracle_postselected_pauli_rank_statistics(alpha_power, p_code, q_codes, target_state, schedule)
        total_a += float(prob) * float(stats[0])
        total_n += float(prob) * float(stats[1])
    return float(total_n / total_a)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    main = '''import numpy as np\nalpha_power=1; p_code=np.array([0,1,0,0,0],int); q_codes=np.array([[3,1,0,1,0],[2,0,0,0,1]],int); h=np.array([np.cos(np.pi/8),np.sin(np.pi/8)],complex); target_state=np.kron(h,np.array([1,0],complex)); fault_schedules=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[1,1,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[2,3,0],[0,0,0],[0,0,0]],[[3,1,1],[0,2,0],[0,0,0],[0,0,0]],[[0,0,2],[1,0,0],[0,3,1],[0,0,0]],[[2,1,3],[3,0,1],[1,2,0],[0,3,3]],[[0,2,2],[0,0,0],[2,1,0],[3,0,2]],[[1,3,0],[2,0,1],[3,1,2],[1,2,3]]],int); probabilities=np.array([0.3271,0.1437,0.1079,0.0913,0.0842,0.0746,0.0934,0.0778],float)'''
    hcase = '''import numpy as np\nalpha_power=0; p_code=np.array([0,1,0],int); q_codes=np.array([[3,1,1]],int); target_state=np.array([np.cos(np.pi/8),np.sin(np.pi/8)],complex); fault_schedules=np.array([[[0,0],[0,0],[0,0]],[[1,1],[0,0],[0,0]],[[2,3],[0,1],[0,0]],[[3,2],[1,3],[2,1]]],int); probabilities=np.array([0.47,0.21,0.18,0.14],float)'''
    scase = '''import numpy as np\nalpha_power=1; p_code=np.array([0,0,0],int); q_codes=np.array([[2,0,1]],int); target_state=np.array([1,0],complex); fault_schedules=np.array([[[0,0],[0,0]],[[1,1],[0,0]],[[2,3],[3,1]],[[3,2],[1,3]]],int); probabilities=np.array([0.52,0.18,0.17,0.13],float)'''
    return [
        {'setup': main, 'call': 'ensemble_compact_magic_fidelity(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedules.copy(),probabilities.copy())', 'gold_call': '_oracle_ensemble_compact_magic_fidelity(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedules.copy(),probabilities.copy())', 'tol': 8e-11},
        {'setup': hcase, 'call': 'ensemble_compact_magic_fidelity(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedules.copy(),probabilities.copy())', 'gold_call': '_oracle_ensemble_compact_magic_fidelity(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedules.copy(),probabilities.copy())', 'tol': 8e-11},
        {'setup': scase, 'call': 'ensemble_compact_magic_fidelity(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedules.copy(),probabilities.copy())', 'gold_call': '_oracle_ensemble_compact_magic_fidelity(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedules.copy(),probabilities.copy())', 'tol': 8e-11},
    ]
