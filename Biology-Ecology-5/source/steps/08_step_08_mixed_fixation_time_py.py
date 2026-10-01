"""
Orchestrator: calculate the mixed supply sensitivity of conditional mean fixation time.

This pipeline combines nonlinear stationary ecology, the generalized parent-mutant reduction, and conditional first-passage statistics. Both supply directions affect the stationary state and every ecological coefficient before conditioning.

Returns
-------
Native Python float: mixed epsilon-zeta derivative of the natural log conditional mean time to fixation. Compose all earlier steps, including the scale-integral step reached through hitting_moment_jet. Use the stated quadrature convention. A zero supply direction or identical focal uptake, saturation and mortality gives zero sensitivity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixed_fixation_time(config: dict, order: int = 64) -> float:
    """Orchestrator: calculate the mixed supply sensitivity of conditional mean fixation time.

    Parameters
    ----------
    config : dict
        Required keys: U, half, B, leakage, N, R, v, w, N0, mp, mm, D, f0.
        U and half have shape (S+2,M), with S residents followed by parent
        and mutant; 1 <= S <= M. U is nonnegative, half positive; N is
        positive (S,), R positive (M,); v,w are signed (M,) supply directions.
        B is nonnegative (M,M), columns sum to one to tolerance 1e-12;
        leakage lies in [0,1). N0,D are positive, mp,mm nonnegative, 0<f0<1.
        All data are finite and real. Resident mortality and baseline supply
        are implied by the cavity state; only supply changes by epsilon*v
        plus zeta*w. Fixed resident set, combined focal abundance and noise.
        Use the specified Monod model and the generalized cavity reduction,
        including parent correction. Extra keys are ignored. The stationary
        Jacobian must be nonsingular.
    order : int
        Inner and outer Gauss-Legendre order in [16,256], not boolean.

    Returns
    -------
    result : float
        Native Python float: mixed epsilon-zeta derivative of the natural
        log conditional mean time to fixation. Compose all earlier steps,
        including the scale-integral step reached through hitting_moment_jet.
        Use the stated quadrature convention. A zero supply direction or
        identical focal uptake, saturation and mortality gives zero sensitivity.

    Raises
    ------
    ValueError
        If config is not a dictionary or a required key is absent, any stated
        shape, real/finiteness, sign, leakage, conversion, frequency or order
        condition fails, the stationary Jacobian is singular, or intermediate
        arithmetic/results are nonfinite or a required probability or moment
        is nonpositive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_mixed_fixation_time(config: dict, order: int = 64) -> float:
    keys=('U','half','B','leakage','N','R','v','w','N0','mp','mm','D','f0')
    if not isinstance(config,dict) or any(k not in config for k in keys):raise ValueError('missing configuration')
    c=config;N=_arr(c['N'],1);s=len(N);U=_arr(c['U'],2)
    if U.shape[0]!=s+2:raise ValueError('U must contain residents then parent and mutant')
    t=_oracle_uptake_tensors(U,c['half'],c['R'])
    state=_oracle_cavity_state_jet(t[:,:s],c['B'],c['leakage'],N,c['R'],c['v'],c['w'])
    H=_oracle_susceptibility_jet(t[:,:s],c['B'],c['leakage'],state)
    sel=_oracle_selection_mixed_jet(t[:,s:],c['B'],c['leakage'],state[:,s:],H,c['N0'],c['mp'],c['mm'])
    mom=_oracle_hitting_moment_jet(sel,c['N0'],c['D'],c['f0'],order)
    return float(_oracle_conditional_time_statistics(mom)[4])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge and declared-invalid test cases."""
    return [{'setup': "import numpy as np\nimport copy\n# normal\na0={'U': np.array([[1.0, 0.2, 0.1, 0.05], [0.1, 0.9, 0.25, 0.1], [0.15, 0.1, 0.8, 0.3], [0.4, 0.3, 0.2, 0.1], [0.425, 0.28, 0.215, 0.09]], dtype=float), 'half': np.array([[0.4, 0.7, 0.3, 0.8], [0.6, 0.2, 0.9, 0.5], [0.3, 0.8, 0.4, 0.6], [0.5, 0.4, 0.7, 0.3], [0.52, 0.38, 0.72, 0.28]], dtype=float), 'B': np.array([[0.0, 0.1, 0.2, 0.3], [0.5, 0.0, 0.3, 0.2], [0.3, 0.6, 0.0, 0.5], [0.2, 0.3, 0.5, 0.0]], dtype=float), 'leakage': 0.35, 'N': np.array([1.2, 0.9, 1.1], dtype=float), 'R': np.array([1.0, 0.8, 1.2, 0.9], dtype=float), 'v': np.array([0.3, -0.2, 0.15, -0.1], dtype=float), 'w': np.array([-0.1, 0.25, 0.2, -0.15], dtype=float), 'N0': 0.1, 'mp': 0.4, 'mm': 0.4029148343, 'D': 2e-08, 'f0': 0.07}\n", 'call': 'mixed_fixation_time(copy.deepcopy(a0))', 'gold_call': '_oracle_mixed_fixation_time(copy.deepcopy(a0))', 'tol': 1e-08}, {'setup': "import numpy as np\nimport copy\n# boundary\na0={'U': np.array([[1.0, 0.2, 0.1, 0.05], [0.1, 0.9, 0.25, 0.1], [0.15, 0.1, 0.8, 0.3], [0.4, 0.3, 0.2, 0.1], [0.425, 0.28, 0.215, 0.09]], dtype=float), 'half': np.array([[0.4, 0.7, 0.3, 0.8], [0.6, 0.2, 0.9, 0.5], [0.3, 0.8, 0.4, 0.6], [0.5, 0.4, 0.7, 0.3], [0.52, 0.38, 0.72, 0.28]], dtype=float), 'B': np.array([[0.0, 0.1, 0.2, 0.3], [0.5, 0.0, 0.3, 0.2], [0.3, 0.6, 0.0, 0.5], [0.2, 0.3, 0.5, 0.0]], dtype=float), 'leakage': 0.35, 'N': np.array([1.2, 0.9, 1.1], dtype=float), 'R': np.array([1.0, 0.8, 1.2, 0.9], dtype=float), 'v': np.array([0.0, 0.0, 0.0, 0.0], dtype=float), 'w': np.array([-0.1, 0.25, 0.2, -0.15], dtype=float), 'N0': 0.1, 'mp': 0.4, 'mm': 0.4029148343, 'D': 2e-08, 'f0': 0.07}\n", 'call': 'mixed_fixation_time(copy.deepcopy(a0))', 'gold_call': '_oracle_mixed_fixation_time(copy.deepcopy(a0))', 'tol': 1e-08}, {'setup': "import numpy as np\nimport copy\n# edge\na0={'U': np.array([[0.8, 0.3], [0.4, 0.2], [0.45, 0.18]], dtype=float), 'half': np.array([[0.3, 0.7], [0.5, 0.4], [0.6, 0.35]], dtype=float), 'B': np.array([[0.2, 0.7], [0.8, 0.3]], dtype=float), 'leakage': 0.2, 'N': np.array([0.8], dtype=float), 'R': np.array([1.1, 0.9], dtype=float), 'v': np.array([0.2, -0.1], dtype=float), 'w': np.array([-0.15, 0.3], dtype=float), 'N0': 0.08, 'mp': 0.1, 'mm': 0.105, 'D': 0.0002, 'f0': 0.3}\n", 'call': 'mixed_fixation_time(copy.deepcopy(a0))', 'gold_call': '_oracle_mixed_fixation_time(copy.deepcopy(a0))', 'tol': 1e-08}, {'setup': "import numpy as np\nimport copy\n# invalid_declared_condition\na0={'U': np.array([[1.0, 0.2, 0.1, 0.05], [0.1, 0.9, 0.25, 0.1], [0.15, 0.1, 0.8, 0.3], [0.4, 0.3, 0.2, 0.1], [0.425, 0.28, 0.215, 0.09]], dtype=float), 'half': np.array([[0.4, 0.7, 0.3, 0.8], [0.6, 0.2, 0.9, 0.5], [0.3, 0.8, 0.4, 0.6], [0.5, 0.4, 0.7, 0.3], [0.52, 0.38, 0.72, 0.28]], dtype=float), 'B': np.array([[0.0, 0.1, 0.2, 0.3], [0.5, 0.0, 0.3, 0.2], [0.3, 0.6, 0.0, 0.5], [0.2, 0.3, 0.5, 0.0]], dtype=float), 'leakage': 0.35, 'N': np.array([1.2, 0.9, 1.1], dtype=float), 'R': np.array([1.0, 0.8, 1.2, 0.9], dtype=float), 'v': np.array([0.3, -0.2, 0.15, -0.1], dtype=float), 'w': np.array([-0.1, 0.25, 0.2, -0.15], dtype=float), 'N0': 0.1, 'mp': 0.4, 'mm': 0.4029148343, 'f0': 0.07}\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    return 0\n", 'call': 'expect_value_error(mixed_fixation_time, copy.deepcopy(a0))', 'gold_call': 'expect_value_error(_oracle_mixed_fixation_time, copy.deepcopy(a0))', 'tol': 0}]
