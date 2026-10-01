"""
Charge-fluctuation propagators of one molecule

A molecule held in a steady state does not carry a static charge; its occupation fluctuates, and

the spectrum of those fluctuations is what the partner molecule couples to. The object that

carries this information is the particle-hole propagator of the molecule, and it has two

components. One of them measures the rate at which the molecule can take up a charge fluctuation

of a given frequency, the other the rate at which it can give one off. Each is obtained from the

lesser and greater Green's functions of that same molecule by a single frequency integral in which

the two functions are evaluated at arguments differing by the fluctuation frequency, so that a

process of frequency w is built from all the ways of removing weight at one energy and supplying

it at an energy lower by w. The two components differ in which Green's function supplies which

end of that pair.



Both components are pure imaginary, and the sign of each is fixed by the requirement that the

corresponding physical spectral weight be non-negative at every frequency. A shift of the

integration variable relates the two components at opposite frequencies, and that relation is

exact rather than approximate; it is worth using as a check on an implementation, and it is what

makes the energy integral of step 09 finite.



On the grid used throughout this task the difference of any two grid frequencies is itself a grid

frequency, so the integral is a plain index shift and needs no interpolation. A discrete

convolution formed with a complex conjugation of one argument, which is what the textbook fast

correlation identity produces, is not the same object and will give a silently wrong answer.

Returns
-------
#     np.ndarray of shape (2, n), complex: lesser then greater charge-fluctuation propagator, eV^-1  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def polarisation_bubbles(g_lesser: np.ndarray, g_greater: np.ndarray, dw: float) -> np.ndarray:
    '''Lesser and greater charge-fluctuation propagators of one molecule.

    Parameters
    ----------
    g_lesser, g_greater : np.ndarray
        Lesser and greater Green's functions of the molecule on the task grid, in eV^-1.
        The grid is w_j = (j - n/2) * dw with n = len(g_lesser).
    dw : float
        Grid spacing in eV; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, n); row 0 is the lesser propagator and row 1 the greater
        propagator, both in eV^-1. Frequencies outside the grid contribute nothing.

    Raises
    ------
    ValueError
        If dw is not positive, or if the two input arrays have different lengths.
    '''
    return result  # placeholder

# ============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fft_linear_convolve(a, b):
    m = len(a) + len(b) - 1
    size = 1
    while size < m:
        size *= 2
    return np.fft.ifft(np.fft.fft(a, size) * np.fft.fft(b, size))[:m]


def _shifted_correlation(a, b, dw):
    """(1/2pi) * int dw' a(w') b(w' - w), as an exact index shift on this grid."""
    n = len(a)
    half = n // 2
    full = _fft_linear_convolve(np.asarray(a, dtype=complex),
                               np.asarray(b, dtype=complex)[::-1])
    return full[np.arange(n) + (n - 1 - half)] * dw / (2.0 * np.pi)


def _oracle_polarisation_bubbles(g_lesser, g_greater, dw):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    gl = np.asarray(g_lesser, dtype=complex)
    gg = np.asarray(g_greater, dtype=complex)
    if gl.shape != gg.shape:
        raise ValueError("g_lesser and g_greater must have the same shape")
    pi_lesser = -1j * _shifted_correlation(gl, gg, dw)
    pi_greater = -1j * _shifted_correlation(gg, gl, dw)
    return np.vstack([pi_lesser, pi_greater])

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_KB = 8.617333262e-5\n"
          "_n = 1024\n"
          "_dw = 20.0 / _n\n"
          "_w = (np.arange(_n) - _n // 2) * _dw\n"
          "def _greens(eps, gam, V, T, gl, gr, fr):\n"
          "    mu_l = fr * V\n"
          "    mu_r = -(1.0 - fr) * V\n"
          "    f1 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_l) / (2.0 * _KB * T), -400, 400)))\n"
          "    f2 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_r) / (2.0 * _KB * T), -400, 400)))\n"
          "    sl = 1j * (gl * f1 + gr * f2)\n"
          "    sg = -1j * (gl * (1 - f1) + gr * (1 - f2))\n"
          "    gret = 1.0 / (_w - eps + 0.5j * gam)\n"
          "    return gret * sl * np.conj(gret), gret * sg * np.conj(gret)\n"
          "_GL, _GG = _greens(-0.7800536, 0.080, 2.0, 300.0, 0.050, 0.030, 0.70)\n"
          "_EL, _EG = _greens(-0.50, 0.050, 0.0, 300.0, 0.025, 0.025, 0.5)")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'polarisation_bubbles(_GL, _GG, _dw)',
            "gold_call": '_oracle_polarisation_bubbles(_GL, _GG, _dw)',
            "note": 'normal, the driven molecule of the task',
        },
        {
            "setup": _SETUP,
            "call": 'polarisation_bubbles(_EL, _EG, _dw)',
            "gold_call": '_oracle_polarisation_bubbles(_EL, _EG, _dw)',
            "note": 'normal, an unbiased molecule in equilibrium',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(polarisation_bubbles(_GL, _GG, _dw)[0]'
                     ' - polarisation_bubbles(_GL, _GG, _dw)[1][(_n - np.arange(_n)) % _n]).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_polarisation_bubbles(_GL, _GG, _dw)[0]'
                          ' - _oracle_polarisation_bubbles(_GL, _GG, _dw)[1][(_n - np.arange(_n)) % _n]).max()])'),
            "note": 'contract, the two components must map onto one another under reversal of the frequency',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(polarisation_bubbles(_GL, _GG, _dw)[0].real).max(),'
                     ' np.abs(polarisation_bubbles(_GL, _GG, _dw)[1].real).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_polarisation_bubbles(_GL, _GG, _dw)[0].real).max(),'
                          ' np.abs(_oracle_polarisation_bubbles(_GL, _GG, _dw)[1].real).max()])'),
            "note": 'contract, both propagators must be pure imaginary',
        },
        {
            "setup": _SETUP,
            "call": 'polarisation_bubbles(*_greens(-1.20, 0.200, 3.5, 450.0, 0.120, 0.080, 0.35), _dw)',
            "gold_call": '_oracle_polarisation_bubbles(*_greens(-1.20, 0.200, 3.5, 450.0, 0.120, 0.080, 0.35), _dw)',
            "note": 'normal, a broad deep level under a strongly asymmetric large bias',
        },
        {
            "setup": _SETUP,
            "call": 'polarisation_bubbles(*_greens(-0.60, 0.080, 1.8, -300.0, 0.050, 0.030, 0.70), _dw)',
            "gold_call": '_oracle_polarisation_bubbles(*_greens(-0.60, 0.080, 1.8, -300.0, 0.050, 0.030, 0.70), _dw)',
            "note": 'edge, a population-inverted reservoir reverses which component dominates',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: polarisation_bubbles(_GL, _GG, 0.0)),'
                     ' _raises(lambda: polarisation_bubbles(_GL[:10], _GG, _dw))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_polarisation_bubbles(_GL, _GG, 0.0)),'
                          ' _raises(lambda: _oracle_polarisation_bubbles(_GL[:10], _GG, _dw))])'),
            "note": 'contract, non-positive spacing and mismatched lengths must raise ValueError',
        },
    ]
