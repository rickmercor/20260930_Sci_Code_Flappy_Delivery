"""
Construct a two-qutrit synthetic Markovian control model in the basis |i,j> with i first. On one qutrit define X=|0><1|+|1><0|, Y=-i|0><1|+i|1><0|, Z=diag(1,-1,0), N=diag(0,1,2), a=|0><1|+sqrt(2)|1><2| and ell=|2><1|. Return Hamiltonians [1.3 X tensor X, 0.9 Z tensor I+0.7 I tensor Y, 1.1 Y tensor Z+0.4 N tensor I], the feedforward Hamiltonian feedforward*X tensor I, jumps [a tensor I,I tensor a,Z tensor I,I tensor Z,ell tensor I,I tensor ell], rates noise_scale*[1,1,0.7,0.7,0.5,0.5], projectors [diag(1,0,0) tensor I,diag(0,1,1) tensor I], rho=|00><00| and observable=rho as a seven-tuple. Both arguments are finite nonboolean real scalars; the synthetic benchmark domain is noise_scale in [0,1] and feedforward in [-2,2]. Invalid inputs raise ValueError.

Level 2 is retained as a leakage level, not discarded by post-selection. The lowering jump a allows amplitude relaxation, the Z jump dephases computational coherence and ell transfers population from level 1 to level 2. These operators define the benchmark model rather than a fitted experimental calibration.

Returns
-------
tuple    Hamiltonians, feedback, jumps, rates, projectors, density and observable.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qutrit_model(noise_scale: float = 0.06, feedforward: float = 0.8) -> tuple:
    """Build the fixed operators with variable noise and feedback strengths.

    Parameters
    ----------
    noise_scale : float
        Finite nonboolean rate multiplier in [0,1].
    feedforward : float
        Finite nonboolean feedback amplitude in [-2,2].

    Returns
    -------
    tuple
        Hamiltonians, feedback, jumps, rates, projectors, density and observable.

    Raises
    ------
    ValueError
        If a scalar is not real, finite or in its stated domain.
    """
    return None


# EXPECTED RETURN LINE
# tuple of seven numeric arrays defining the two-qutrit model

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qutrit_model(noise_scale: float = 0.06, feedforward: float = 0.8) -> tuple:
    vals = []
    for value, low, high in ((noise_scale, 0, 1), (feedforward, -2, 2)):
        v = np.asarray(value)
        if v.ndim != 0 or v.dtype.kind not in 'iuf' or not np.isfinite(v) or not low <= float(v) <= high:
            raise ValueError('invalid scalar')
        vals.append(float(v))
    xi, f = vals
    eye = np.eye(3, dtype=complex)
    x = np.zeros((3, 3), complex)
    x[0, 1] = x[1, 0] = 1
    y = np.zeros((3, 3), complex)
    y[0, 1], y[1, 0] = -1j, 1j
    z = np.diag([1, -1, 0]).astype(complex)
    n = np.diag([0, 1, 2]).astype(complex)
    a = np.diag([1, np.sqrt(2)], 1).astype(complex)
    ell = np.zeros((3, 3), complex)
    ell[2, 1] = 1
    h = np.array([1.3*np.kron(x, x), 0.9*np.kron(z, eye)+0.7*np.kron(eye, y), 1.1*np.kron(y, z)+0.4*np.kron(n, eye)])
    hf = f*np.kron(x, eye)
    jumps = np.array([np.kron(a, eye), np.kron(eye, a), np.kron(z, eye), np.kron(eye, z), np.kron(ell, eye), np.kron(eye, ell)])
    rates = xi*np.array([1, 1, 0.7, 0.7, 0.5, 0.5])
    projectors = np.array([np.kron(np.diag([1, 0, 0]), eye), np.kron(np.diag([0, 1, 1]), eye)])
    rho = np.zeros((9, 9), complex)
    rho[0, 0] = 1
    return h, hf, jumps, rates, projectors, rho, rho.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases: list[dict] = [
        {'setup': '', 'call': 'qutrit_model(.06,.8)', 'gold_call': '_oracle_qutrit_model(.06,.8)'},
        {'setup': '', 'call': 'qutrit_model(0.,0.)', 'gold_call': '_oracle_qutrit_model(0.,0.)'},
        {'setup': '', 'call': 'qutrit_model(.2,-1.7)', 'gold_call': '_oracle_qutrit_model(.2,-1.7)'},
    ]
    for args in ('-0.1,.8', '.1,np.inf', 'True,.8'):
        setup = ''
        for name, fn in (('run_model', 'qutrit_model'), ('run_gold', '_oracle_qutrit_model')):
            setup += f'def {name}():\n    try:\n        {fn}({args})\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup': setup, 'call': 'run_model()', 'gold_call': 'run_gold()'})
    packing = 'def pack(values):\n    arrays=[np.asarray(v,dtype=complex) for v in values]\n    shape=np.array([x for a in arrays for x in (a.ndim,)+a.shape],dtype=complex)\n    return np.concatenate([shape]+[a.ravel() for a in arrays])\n'
    for case in cases:
        if case['call'].startswith('qutrit_model('):
            case['setup'] = packing + case['setup']
            case['call'] = 'pack(' + case['call'] + ')'
            case['gold_call'] = 'pack(' + case['gold_call'] + ')'
        case['setup'] = 'import numpy as np\n' + case['setup']
        case['tol'] = 1e-11
    return cases
