"""
Retarded, lesser and greater Green's functions of a driven level

With the electrodes summarised by their self-energies and the level shifted by its mean-field

partner interaction, the single-particle description of one molecule reduces to three frequency-

dependent functions. The retarded function follows from the shifted level position together with

every retarded self-energy acting on it, the electrode contribution entering as the usual

imaginary damping built from the total hybridisation. The lesser and greater functions are then

fixed by the retarded and advanced functions and by the corresponding lesser and greater

self-energies, with no additional freedom: they are not independent objects to be modelled but

are determined once the retarded function and the injection and extraction rates are known.



The routine has to accept an optional correlation self-energy in all three of its components, so

that the same code serves both the starting point, where those contributions are absent, and every

later pass of the self-consistency, where they are not. When they are absent the result must reduce

exactly to the electrode-only case rather than approximately.



Two structural properties are worth preserving in the implementation because later steps rely on

them. The lesser function is i times a real non-negative quantity and the greater function is

minus i times a real non-negative quantity, and their difference is fixed by the spectral weight

of the level. Round-off that leaks a real part into either of them will propagate into the

polarisation and the energy.

Returns
-------
#     np.ndarray of shape (3, len(w)), complex: retarded, lesser and greater Green's functions, eV^-1  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def keldysh_greens(w: np.ndarray, eps_shifted: float, gamma_total: float, sigma_lesser: np.ndarray,
                   sigma_greater: np.ndarray, sig_c_ret: np.ndarray, sig_c_lesser: np.ndarray,
                   sig_c_greater: np.ndarray) -> np.ndarray:
    '''Retarded, lesser and greater Green's functions of one molecular level.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV.
    eps_shifted : float
        Molecular level energy in eV, already including the static mean-field shift.
    gamma_total : float
        Total hybridisation width of the junction in eV; positive.
    sigma_lesser, sigma_greater : np.ndarray
        Lesser and greater electrode self-energies on the same grid, in eV.
    sig_c_ret, sig_c_lesser, sig_c_greater : np.ndarray
        Retarded, lesser and greater correlation self-energies on the same grid, in eV. Pass
        arrays of zeros to obtain the electrode-only result.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (3, len(w)): row 0 the retarded Green's function, row 1 the
        lesser and row 2 the greater, all in eV^-1.

    Raises
    ------
    ValueError
        If gamma_total is not positive, or if any input array length differs from len(w).
    '''
    return result  # placeholder

# ============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_keldysh_greens(w, eps_shifted, gamma_total, sigma_lesser, sigma_greater,
                           sig_c_ret, sig_c_lesser, sig_c_greater):
    if not (gamma_total > 0.0):
        raise ValueError("gamma_total must be positive")
    w = np.asarray(w, dtype=float)
    n = len(w)
    arrs = [np.asarray(a) for a in (sigma_lesser, sigma_greater, sig_c_ret,
                                    sig_c_lesser, sig_c_greater)]
    for a in arrs:
        if a.shape[-1] != n:
            raise ValueError("all self-energy arrays must match the length of w")
    sl, sg, scr, scl, scg = arrs
    g_ret = 1.0 / (w - eps_shifted - scr + 0.5j * gamma_total)
    g_adv = np.conj(g_ret)
    g_lesser = g_ret * (sl + scl) * g_adv
    g_greater = g_ret * (sg + scg) * g_adv
    return np.vstack([g_ret, g_lesser, g_greater])

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_KB = 8.617333262e-5\n"
          "_n = 1024\n"
          "_dw = 20.0 / _n\n"
          "_w = (np.arange(_n) - _n // 2) * _dw\n"
          "_Z = np.zeros(_n, dtype=complex)\n"
          "def _sig(V, T, gl, gr, fr, ef=0.0):\n"
          "    mu_l = ef + fr * V\n"
          "    mu_r = ef - (1.0 - fr) * V\n"
          "    f1 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_l) / (2.0 * _KB * T), -400, 400)))\n"
          "    f2 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_r) / (2.0 * _KB * T), -400, 400)))\n"
          "    return 1j * (gl * f1 + gr * f2), -1j * (gl * (1 - f1) + gr * (1 - f2))\n"
          "_SL, _SG = _sig(2.0, 300.0, 0.050, 0.030, 0.70)\n"
          "_CR = 0.03 * np.exp(-(_w / 1.5) ** 2) - 0.02j * np.exp(-(_w / 2.0) ** 2)\n"
          "_CL = 1j * 0.01 * np.exp(-((_w - 0.3) / 1.1) ** 2)\n"
          "_CG = -1j * 0.014 * np.exp(-((_w + 0.2) / 1.3) ** 2)")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)',
            "gold_call": '_oracle_keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)',
            "note": 'normal, the electrode-only starting point of the task',
        },
        {
            "setup": _SETUP,
            "call": 'keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _CR, _CL, _CG)',
            "gold_call": '_oracle_keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _CR, _CL, _CG)',
            "note": 'normal, a dressed pass with all three correlation components present',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[2]'
                     ' - keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[1]'
                     ' - (keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[0]'
                     ' - np.conj(keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[0]))).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[2]'
                          ' - _oracle_keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[1]'
                          ' - (_oracle_keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[0]'
                          ' - np.conj(_oracle_keldysh_greens(_w, -0.7800536, 0.080, _SL, _SG, _Z, _Z, _Z)[0]))).max()])'),
            "note": 'contract, greater minus lesser must equal retarded minus advanced',
        },
        {
            "setup": _SETUP,
            "call": 'keldysh_greens(_w, 0.45, 0.020, *_sig(0.0, 120.0, 0.010, 0.010, 0.5), _Z, _Z, _Z)',
            "gold_call": '_oracle_keldysh_greens(_w, 0.45, 0.020, *_sig(0.0, 120.0, 0.010, 0.010, 0.5), _Z, _Z, _Z)',
            "note": 'edge, an empty level in equilibrium with a narrow width',
        },
        {
            "setup": _SETUP,
            "call": 'keldysh_greens(_w, -2.10, 0.300, _SL, _SG, 0.5 * _CR, _CL, _CG)',
            "gold_call": '_oracle_keldysh_greens(_w, -2.10, 0.300, _SL, _SG, 0.5 * _CR, _CL, _CG)',
            "note": 'normal, a deep level with a broad hybridisation',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: keldysh_greens(_w, -0.7800536, 0.0, _SL, _SG, _Z, _Z, _Z)),'
                     ' _raises(lambda: keldysh_greens(_w, -0.7800536, 0.08, _SL[:10], _SG, _Z, _Z, _Z))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_keldysh_greens(_w, -0.7800536, 0.0, _SL, _SG, _Z, _Z, _Z)),'
                          ' _raises(lambda: _oracle_keldysh_greens(_w, -0.7800536, 0.08, _SL[:10], _SG, _Z, _Z, _Z))])'),
            "note": 'contract, zero width and a mismatched array length must raise ValueError',
        },
    ]
