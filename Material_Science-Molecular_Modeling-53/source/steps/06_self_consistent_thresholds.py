"""
Determine the source's rupture thresholds for a chain of N bonds self-consistently (the numbered procedure in the paper's Sec. S2.2). Each bond's PMF depends parametrically on the thresholds of the OTHER bonds, so the source alternates between rebuilding the angular weights for the current threshold set and updating each threshold to the stationary point that the variational transition-state principle selects, until the set stops changing. Start every threshold at 2.4 and run exactly 8 sweeps, which is far past convergence here. Locate stationary points by scanning bond lengths on [0.6, 6.0] with 24001 uniform points, taking the first decreasing-to-increasing sign change and then the next increasing-to-decreasing one, refining each by 80 ternary-search iterations on half-width 0.02. Raise ValueError unless N is an integer of at least 2. Assemble by calling the earlier sub-problem functions.

Because the thresholds enter each other's PMFs, they are not known in advance and must be found together with the free-energy landscapes they define.

Returns
-------
return (N,) float64: the converged rupture thresholds
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e):
    """N: number of bonds (>= 2); f: applied force; beta: inverse temperature;
    De, a, le: Morse parameters; kphi, phi_e: bending parameters. Returns (N,)
    float64 with the source's self-consistent rupture thresholds. Raises
    ValueError on an invalid N."""
    return np.zeros(int(N))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: self-consistent rupture thresholds (Sec. S2.2 procedure)."""

import numpy as np

_NSCAN, _LO, _HI, _KITER = 24001, 0.6, 6.0, 8


def _stationary(w_i, f, beta, De, a, le):
    l = np.linspace(_LO, _HI, _NSCAN)
    W = _oracle_tm_pmf(l, w_i, f, beta, De, a, le)
    d = np.diff(W)
    s = np.sign(d)
    idx = np.where(s[:-1] != s[1:])[0] + 1
    lm = lb = None
    for k in idx:
        if s[k - 1] < 0 and s[k] > 0 and lm is None:
            lm = l[k]
        elif s[k - 1] > 0 and s[k] < 0 and lm is not None and lb is None:
            lb = l[k]
    if lm is None or lb is None:
        raise ValueError("no minimum/threshold pair; force may exceed f_c")

    def ref(l0, kind):
        A, B = l0 - 0.02, l0 + 0.02
        for _ in range(80):
            m1 = A + (B - A) / 3.0
            m2 = B - (B - A) / 3.0
            f1 = float(_oracle_tm_pmf(m1, w_i, f, beta, De, a, le)[0])
            f2 = float(_oracle_tm_pmf(m2, w_i, f, beta, De, a, le)[0])
            if kind == 'min':
                if f1 < f2: B = m2
                else: A = m1
            else:
                if f1 > f2: B = m2
                else: A = m1
        return 0.5 * (A + B)

    return ref(lm, 'min'), ref(lb, 'max')


def _oracle_self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e):
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2")
    Q = _oracle_bending_kernel(beta, kphi, phi_e)
    thr = np.full(int(N), 2.4, dtype=np.float64)
    for _ in range(_KITER):
        w = _oracle_tm_angular_weights(thr, f, beta, De, a, le, Q)
        thr = np.array([_stationary(w[i], f, beta, De, a, le)[1] for i in range(int(N))])
    return thr

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nN=5;f=0.20', "call": 'self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)', "gold_call": '_oracle_self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)', "tol": 1e-07},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nN=4;f=0.15', "call": 'self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)', "gold_call": '_oracle_self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)', "tol": 1e-07},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nN=3;f=0.25', "call": 'self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)', "gold_call": '_oracle_self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)', "tol": 1e-07},
    ]
