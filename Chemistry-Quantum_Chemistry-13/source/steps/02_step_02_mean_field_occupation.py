"""
Mean-field occupation of a level that feels the charge on its partner

The density-density coupling between the two molecules produces, already at mean-field level, a

static shift of each molecular level proportional to how far its partner departs from charge

neutrality. Because the two molecules here are identical and identically biased, they carry the

same occupation, and the mean-field problem closes on a single unknown: the occupation of a level

is a function of the shift it feels, and the shift is a function of that same occupation.



That is a nonlinear scalar equation, and it is not benign. Near resonance the derivative of the

occupation with respect to the level position is large enough that the obvious repeated

substitution does not settle: it oscillates, and the value one reads off after a fixed number of

sweeps depends on where the sweep was stopped. The equation must instead be solved as a root

problem on the physically allowed interval. More than one root can exist; the convention adopted

throughout this task is that the physical solution is the smallest occupation in [0, 1] that

satisfies the equation.



The occupation itself is obtained from the lesser Green's function of the level, built from the

retarded Green's function of the shifted level and the electrode self-energies alone. No

correlation self-energy enters at this stage; this step is the starting point that the later

self-consistency is built on, not the final answer for the charge.

Returns
-------
#     np.ndarray of length 2: [mean-field occupation (dimensionless), static level shift (eV)]  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def mean_field_occupation(w: np.ndarray, dw: float, eps: float, gamma_total: float,
                          sigma_lesser: np.ndarray, u_coupling: float, q_core: float) -> np.ndarray:
    '''Self-consistent mean-field occupation of a molecular level and the shift it produces.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV, uniformly spaced.
    dw : float
        Grid spacing in eV; positive.
    eps : float
        Bare molecular level energy in eV.
    gamma_total : float
        Total hybridisation width of the junction in eV; positive.
    sigma_lesser : np.ndarray
        Lesser electrode self-energy on the same grid, in eV.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.
    q_core : float
        Positive ionic core charge q of the partner molecule.

    Returns
    -------
    result : np.ndarray
        Real array of length 2: the mean-field occupation, then the static level shift in eV
        that this occupation produces on the partner.

    Raises
    ------
    ValueError
        If dw or gamma_total is not positive, if u_coupling is negative, or if the mean-field
        equation admits no root in [0, 1].
    '''
    return result  # placeholder

# ============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mean_field_occupation(w, dw, eps, gamma_total, sigma_lesser, u_coupling, q_core):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if not (gamma_total > 0.0):
        raise ValueError("gamma_total must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    w = np.asarray(w, dtype=float)
    sl = np.asarray(sigma_lesser)

    def occ_of(nv):
        gr = 1.0 / (w - eps - u_coupling * (nv - q_core) + 0.5j * gamma_total)
        gl = gr * sl * np.conj(gr)
        return float((-1j * gl).sum().real * dw / (2.0 * np.pi))

    def resid(nv):
        return occ_of(nv) - nv

    xs = np.linspace(0.0, 1.0, 241)
    vs = np.array([resid(x) for x in xs])
    root = None
    for i in range(len(xs) - 1):
        if vs[i] == 0.0:
            root = float(xs[i])
            break
        if vs[i] * vs[i + 1] < 0.0:
            lo, hi, flo = xs[i], xs[i + 1], vs[i]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = resid(mid)
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
                if hi - lo < 1e-15:
                    break
            root = float(0.5 * (lo + hi))
            break
    if root is None:
        raise ValueError("mean-field equation has no root in [0, 1]")
    return np.array([root, u_coupling * (root - q_core)], dtype=float)

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_KB = 8.617333262e-5\n"
          "_n = 1024\n"
          "_dw = 20.0 / _n\n"
          "_w = (np.arange(_n) - _n // 2) * _dw\n"
          "def _sig_l(V, T, gl, gr, fr, ef=0.0):\n"
          "    mu_l = ef + fr * V\n"
          "    mu_r = ef - (1.0 - fr) * V\n"
          "    f1 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_l) / (2.0 * _KB * T), -400, 400)))\n"
          "    f2 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_r) / (2.0 * _KB * T), -400, 400)))\n"
          "    return 1j * (gl * f1 + gr * f2)\n"
          "_S = _sig_l(2.0, 300.0, 0.050, 0.030, 0.70)")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'mean_field_occupation(_w, _dw, -0.75, 0.080, _S, 0.90, 1.0)',
            "gold_call": '_oracle_mean_field_occupation(_w, _dw, -0.75, 0.080, _S, 0.90, 1.0)',
            "note": 'normal, the operating point of the task',
        },
        {
            "setup": _SETUP,
            "call": 'mean_field_occupation(_w, _dw, -0.75, 0.080, _S, 0.0, 1.0)',
            "gold_call": '_oracle_mean_field_occupation(_w, _dw, -0.75, 0.080, _S, 0.0, 1.0)',
            "note": 'boundary, zero coupling removes the nonlinearity and the shift must vanish',
        },
        {
            "setup": _SETUP,
            "call": 'mean_field_occupation(_w, _dw, -0.20, 0.120, _sig_l(2.0, 300.0, 0.070, 0.050, 0.70), 1.80, 1.0)',
            "gold_call": '_oracle_mean_field_occupation(_w, _dw, -0.20, 0.120, _sig_l(2.0, 300.0, 0.070, 0.050, 0.70), 1.80, 1.0)',
            "note": 'edge, loop gain near three so repeated substitution oscillates instead of settling',
        },
        {
            "setup": _SETUP,
            "call": 'mean_field_occupation(_w, _dw, -1.50, 0.090, _sig_l(1.0, 200.0, 0.045, 0.045, 0.5), 0.60, 1.0)',
            "gold_call": '_oracle_mean_field_occupation(_w, _dw, -1.50, 0.090, _sig_l(1.0, 200.0, 0.045, 0.045, 0.5), 0.60, 1.0)',
            "note": 'normal, a deep level under a small symmetric bias',
        },
        {
            "setup": _SETUP,
            "call": 'mean_field_occupation(_w, _dw, 0.40, 0.080, _S, 1.00, 0.0)',
            "gold_call": '_oracle_mean_field_occupation(_w, _dw, 0.40, 0.080, _S, 1.00, 0.0)',
            "note": 'edge, an empty-orbital convention with the core charge set to zero',
        },
        {
            "setup": _SETUP,
            "call": 'mean_field_occupation(_w, _dw, -0.10, 0.120, _sig_l(1.2, 300.0, 0.070, 0.050, 0.5), 2.20, 1.0)',
            "gold_call": '_oracle_mean_field_occupation(_w, _dw, -0.10, 0.120, _sig_l(1.2, 300.0, 0.070, 0.050, 0.5), 2.20, 1.0)',
            "note": 'edge, the largest loop gain reachable here, where iteration wanders by a third of the range',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: mean_field_occupation(_w, -1.0, -0.75, 0.08, _S, 1.0, 1.0)),'
                     ' _raises(lambda: mean_field_occupation(_w, _dw, -0.75, 0.0, _S, 1.0, 1.0)),'
                     ' _raises(lambda: mean_field_occupation(_w, _dw, -0.75, 0.08, _S, -1.0, 1.0))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_mean_field_occupation(_w, -1.0, -0.75, 0.08, _S, 1.0, 1.0)),'
                          ' _raises(lambda: _oracle_mean_field_occupation(_w, _dw, -0.75, 0.0, _S, 1.0, 1.0)),'
                          ' _raises(lambda: _oracle_mean_field_occupation(_w, _dw, -0.75, 0.08, _S, -1.0, 1.0))])'),
            "note": 'contract, non-positive spacing or width and negative coupling must raise ValueError',
        },
    ]
