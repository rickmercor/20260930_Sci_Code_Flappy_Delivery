"""
Closing the loop between the Green's functions and the correlation self-energy

Steps 03 to 06 form a cycle. Green's functions produce charge-fluctuation propagators, those

propagators dress the partner and produce a correlation self-energy in all three of its

components, and that self-energy feeds back into the Green's functions. Nothing in the problem

says the cycle is to be traversed once. Freezing the propagators at their mean-field values and

using them only to evaluate the final energy is a perturbative shortcut, and it is not what is

asked for here: the electronic structure has to be determined at the same order to which the

energy is evaluated, which means the cycle is iterated until it stops changing.



Because the two molecules are identical and identically driven, they carry identical Green's

functions at every stage, so one set of functions can be propagated and used both as the object

being dressed and as the source of the dressing. The static mean-field shift from step 02 is held

fixed throughout; it is the starting point that the correlation dressing is applied on top of, not

a quantity to be re-solved inside the loop.



Convergence is measured on the Green's functions themselves, as the largest absolute change in any

component of either function between one pass and the next. Any stable update scheme is

acceptable, from simple repeated substitution to a damped or accelerated mixing of the old and new

functions; the converged result does not depend on which is used, only on the tolerance reached.

Implementations should return the converged functions rather than the last unconverged pass, and

should stop after a bounded number of passes rather than looping forever.

Returns
-------
#     np.ndarray of shape (2, n), complex: converged lesser and greater Green's functions, eV^-1  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def self_consistent_greens(w: np.ndarray, dw: float, eps_shifted: float, gamma_total: float,
                           sigma_lesser: np.ndarray, sigma_greater: np.ndarray,
                           u_coupling: float, tol: float, max_iter: int) -> np.ndarray:
    '''Green's functions of one molecule with the correlation dressing iterated to convergence.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV of the form w_j = (j - n/2) * dw.
    dw : float
        Grid spacing in eV; positive.
    eps_shifted : float
        Molecular level energy in eV including the static mean-field shift, held fixed.
    gamma_total : float
        Total hybridisation width in eV; positive.
    sigma_lesser, sigma_greater : np.ndarray
        Lesser and greater electrode self-energies on the same grid, in eV.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.
    tol : float
        Convergence threshold on the largest absolute change in any Green's function component
        between successive passes, in eV^-1; positive.
    max_iter : int
        Maximum number of passes; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, n): row 0 the converged lesser Green's function and row 1 the
        converged greater Green's function, both in eV^-1.

    Raises
    ------
    ValueError
        If dw, gamma_total, tol or max_iter is not positive, or if u_coupling is negative.
    '''
    return result  # placeholder

# ============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_self_consistent_greens(w, dw, eps_shifted, gamma_total, sigma_lesser,
                                   sigma_greater, u_coupling, tol, max_iter):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if not (gamma_total > 0.0):
        raise ValueError("gamma_total must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    if not (tol > 0.0):
        raise ValueError("tol must be positive")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive")
    wv = np.asarray(w, dtype=float)
    n = len(wv)
    sl = np.asarray(sigma_lesser, dtype=complex)
    sg = np.asarray(sigma_greater, dtype=complex)
    zero = np.zeros(n, dtype=complex)

    g = _oracle_keldysh_greens(wv, eps_shifted, gamma_total, sl, sg, zero, zero, zero)
    g_l, g_g = g[1], g[2]
    mix = 0.5
    for _ in range(int(max_iter)):
        pi = _oracle_polarisation_bubbles(g_l, g_g, dw)
        sc = _oracle_correlation_selfenergies(g_l, g_g, pi[0], pi[1], u_coupling, dw)
        sc_ret = _oracle_retarded_from_keldysh(sc[0], sc[1], wv, dw)
        nxt = _oracle_keldysh_greens(wv, eps_shifted, gamma_total, sl, sg,
                                     sc_ret, sc[0], sc[1])
        new_l, new_g = nxt[1], nxt[2]
        change = max(float(np.abs(new_l - g_l).max()), float(np.abs(new_g - g_g).max()))
        g_l = g_l + mix * (new_l - g_l)
        g_g = g_g + mix * (new_g - g_g)
        g_l = 1j * g_l.imag
        g_g = 1j * g_g.imag
        if change < tol:
            break
    return np.vstack([g_l, g_g])

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_KB = 8.617333262e-5\n"
          "_n = 1024\n"
          "_dw = 20.0 / _n\n"
          "_w = (np.arange(_n) - _n // 2) * _dw\n"
          "def _sig(V, T, gl, gr, fr, ef=0.0):\n"
          "    mu_l = ef + fr * V\n"
          "    mu_r = ef - (1.0 - fr) * V\n"
          "    f1 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_l) / (2.0 * _KB * T), -400, 400)))\n"
          "    f2 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_r) / (2.0 * _KB * T), -400, 400)))\n"
          "    return 1j * (gl * f1 + gr * f2), -1j * (gl * (1 - f1) + gr * (1 - f2))\n"
          "_SL, _SG = _sig(2.0, 300.0, 0.050, 0.030, 0.70)\n"
          "_QL, _QG = _sig(0.0, 300.0, 0.025, 0.025, 0.5)")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 0.90, 1e-11, 4000)',
            "gold_call": '_oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 0.90, 1e-11, 4000)',
            "note": 'normal, the operating point of the task',
        },
        {
            "setup": _SETUP,
            "call": 'self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 0.0, 1e-11, 4000)',
            "gold_call": '_oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 0.0, 1e-11, 4000)',
            "note": 'boundary, zero coupling must return the undressed electrode-only functions',
        },
        {
            "setup": _SETUP,
            "call": 'self_consistent_greens(_w, _dw, -0.50, 0.050, _QL, _QG, 0.60, 1e-11, 4000)',
            "gold_call": '_oracle_self_consistent_greens(_w, _dw, -0.50, 0.050, _QL, _QG, 0.60, 1e-11, 4000)',
            "note": 'normal, an unbiased molecule at moderate coupling',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 1.00, 1e-12, 6000)'
                     ' - self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 1.00, 1e-10, 6000)).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 1.00, 1e-12, 6000)'
                          ' - _oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 1.00, 1e-10, 6000)).max()])'),
            "note": 'contract, the converged result must not depend on where the tolerance is set',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([float((-1j * self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 0.90, 1e-11, 4000)[0]).sum().real'
                     ' * _dw / (2 * np.pi))])'),
            "gold_call": ('np.array([float((-1j * _oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.080, _SL, _SG, 0.90, 1e-11, 4000)[0]).sum().real'
                          ' * _dw / (2 * np.pi))])'),
            "note": 'contract, the occupation implied by the converged lesser function',
        },
        {
            "setup": _SETUP,
            "call": 'self_consistent_greens(_w, _dw, -1.10, 0.140, *_sig(2.8, 400.0, 0.090, 0.050, 0.40), 1.30, 1e-11, 4000)',
            "gold_call": '_oracle_self_consistent_greens(_w, _dw, -1.10, 0.140, *_sig(2.8, 400.0, 0.090, 0.050, 0.40), 1.30, 1e-11, 4000)',
            "note": 'normal, a broader level under stronger drive and coupling',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: self_consistent_greens(_w, 0.0, -0.7800536, 0.08, _SL, _SG, 1.0, 1e-11, 100)),'
                     ' _raises(lambda: self_consistent_greens(_w, _dw, -0.7800536, 0.08, _SL, _SG, 1.0, 0.0, 100)),'
                     ' _raises(lambda: self_consistent_greens(_w, _dw, -0.7800536, 0.08, _SL, _SG, 1.0, 1e-11, 0))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_self_consistent_greens(_w, 0.0, -0.7800536, 0.08, _SL, _SG, 1.0, 1e-11, 100)),'
                          ' _raises(lambda: _oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.08, _SL, _SG, 1.0, 0.0, 100)),'
                          ' _raises(lambda: _oracle_self_consistent_greens(_w, _dw, -0.7800536, 0.08, _SL, _SG, 1.0, 1e-11, 0))])'),
            "note": 'contract, non-positive spacing, tolerance or iteration cap must raise ValueError',
        },
    ]
