"""
Return the source's bond-resolved activation barriers, in units of kT, for a chain of N bonds at the given force. For each bond the barrier is the PMF difference between the two stationary points of that bond's converged PMF, taken in the source's sense. Because the chain ends truncate the orientational correlations, the barriers depend on bond position and the source's construction makes them symmetric about the chain midpoint. Assemble by calling the earlier sub-problem functions.

The bond-resolved barriers are what make rupture position dependent along the chain, with the ends behaving differently from the interior.

Returns
-------
return (N,) float64: activation barrier of every bond in units of kT
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bond_barriers(N, f, beta, De, a, le, kphi, phi_e):
    """N: number of bonds; f: applied force; beta: inverse temperature;
    De, a, le: Morse parameters; kphi, phi_e: bending parameters. Returns (N,)
    float64 with the source's activation barrier of each bond in units of kT.
    Assembled by calling the earlier sub-problem functions."""
    return np.zeros(int(N))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: bond-resolved activation barriers in kT."""

import numpy as np


_NSCAN, _LO, _HI = 24001, 0.6, 6.0


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


def _oracle_bond_barriers(N, f, beta, De, a, le, kphi, phi_e):
    thr = _oracle_self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)
    Q = _oracle_bending_kernel(beta, kphi, phi_e)
    w = _oracle_tm_angular_weights(thr, f, beta, De, a, le, Q)
    out = np.empty(int(N))
    for i in range(int(N)):
        lm, lb = _stationary(w[i], f, beta, De, a, le)
        out[i] = beta * float(_oracle_tm_pmf(lb, w[i], f, beta, De, a, le)[0]
                              - _oracle_tm_pmf(lm, w[i], f, beta, De, a, le)[0])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nN=5;f=0.20', "call": 'bond_barriers(N, f, beta, De, a, le, kphi, phi_e)', "gold_call": '_oracle_bond_barriers(N, f, beta, De, a, le, kphi, phi_e)', "tol": 1e-07},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nN=4;f=0.15', "call": 'bond_barriers(N, f, beta, De, a, le, kphi, phi_e)', "gold_call": '_oracle_bond_barriers(N, f, beta, De, a, le, kphi, phi_e)', "tol": 1e-07},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nN=3;f=0.25', "call": 'bond_barriers(N, f, beta, De, a, le, kphi, phi_e)', "gold_call": '_oracle_bond_barriers(N, f, beta, De, a, le, kphi, phi_e)', "tol": 1e-07},
    ]
