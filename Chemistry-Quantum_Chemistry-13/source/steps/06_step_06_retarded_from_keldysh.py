"""
Retarded component of a self-energy from its lesser and greater parts

The Dyson equation needs the retarded correlation self-energy, but the diagrammatic construction

of step 05 delivers only the lesser and greater components. The three are not independent. A

self-energy that is retarded is causal, and causality ties its real part to its imaginary part

through a dispersion relation over the whole frequency axis. Applied to the difference of the

greater and lesser components, that relation reconstructs the retarded function completely: one

contribution is a principal-value integral of that difference against a simple pole, and the other

is the local term that the pole leaves behind at coincident frequencies.

The numerical treatment is where this step is won or lost. Regularising the pole by giving it a

small imaginary part is the obvious move and it fails: as soon as that regulator becomes smaller

than the grid spacing, the kernel at coincident frequencies takes a value of order the inverse

regulator, and that single spike leaves the self-consistency of step 07 converging on a self-energy that is wrong by orders of 

magnitude. Taking the principal value directly on the grid avoids the problem and removes the

regulator as a parameter, because the kernel is odd and the point that would diverge is the one

point that drops out by symmetry.

One further detail matters at the accuracy required here. The kernel must be evaluated at every

frequency difference that can occur between two grid points, which spans twice the range of the

grid itself. Building it only over the grid range silently discards the far pairs, and because

the kernel decays only as the reciprocal of the frequency difference, that truncation is worth

several parts in a thousand rather than nothing.

Returns
-------
#     np.ndarray of length n, complex: the retarded self-energy in eV  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def retarded_from_keldysh(sigma_lesser: np.ndarray, sigma_greater: np.ndarray, w: np.ndarray,
                          dw: float) -> np.ndarray:
    '''Retarded self-energy reconstructed from its lesser and greater components.

    Parameters
    ----------
    sigma_lesser, sigma_greater : np.ndarray
        Lesser and greater components of the self-energy on the task grid, in eV.
    w : np.ndarray
        Real frequency grid in eV, of the form w_j = (j - n/2) * dw.
    dw : float
        Grid spacing in eV; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of length n holding the retarded self-energy in eV.

    Raises
    ------
    ValueError
        If dw is not positive, or if the input arrays do not all have the same length.
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


def _oracle_retarded_from_keldysh(sigma_lesser, sigma_greater, w, dw):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    sl = np.asarray(sigma_lesser, dtype=complex)
    sg = np.asarray(sigma_greater, dtype=complex)
    wv = np.asarray(w, dtype=float)
    if not (len(sl) == len(sg) == len(wv)):
        raise ValueError("all input arrays must have the same length")
    n = len(wv)
    num = sg - sl
    offsets = np.arange(2 * n - 1) - (n - 1)
    kernel = np.zeros(2 * n - 1, dtype=complex)
    nonzero = offsets != 0
    kernel[nonzero] = 1.0 / (offsets[nonzero] * dw)
    principal = _fft_linear_convolve(num, kernel)[(n - 1) + np.arange(n)] * dw
    return 1j * principal / (2.0 * np.pi) + 0.5 * num

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_n = 512\n"
          "_dw = 20.0 / _n\n"
          "_w = (np.arange(_n) - _n // 2) * _dw\n"
          "_A = 1j * 0.020 * np.exp(-((_w - 0.35) / 0.9) ** 2)\n"
          "_B = -1j * 0.031 * np.exp(-((_w + 0.22) / 1.2) ** 2)\n"
          "_C = 1j * 0.005 * (1.0 / (1.0 + ((_w - 1.1) / 0.4) ** 2))\n"
          "_D = -1j * 0.009 * (1.0 / (1.0 + ((_w + 0.8) / 0.6) ** 2))")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'retarded_from_keldysh(_A, _B, _w, _dw)',
            "gold_call": '_oracle_retarded_from_keldysh(_A, _B, _w, _dw)',
            "note": 'normal, two gaussian components of opposite sign',
        },
        {
            "setup": _SETUP,
            "call": 'retarded_from_keldysh(_C, _D, _w, _dw)',
            "gold_call": '_oracle_retarded_from_keldysh(_C, _D, _w, _dw)',
            "note": 'normal, lorentzian components with slowly decaying tails',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(retarded_from_keldysh(_A, _B, _w, _dw).imag'
                     ' - 0.5 * (_B - _A).imag).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_retarded_from_keldysh(_A, _B, _w, _dw).imag'
                          ' - 0.5 * (_B - _A).imag).max()])'),
            "note": 'contract, the imaginary part must equal half the difference of the two components',
        },
        {
            "setup": _SETUP,
            "call": 'retarded_from_keldysh(np.zeros(_n, complex), np.zeros(_n, complex), _w, _dw)',
            "gold_call": '_oracle_retarded_from_keldysh(np.zeros(_n, complex), np.zeros(_n, complex), _w, _dw)',
            "note": 'boundary, vanishing input must give a vanishing retarded function',
        },
        {
            "setup": _SETUP,
            "call": 'retarded_from_keldysh(_A + _C, _B + _D, _w, _dw)',
            "gold_call": '_oracle_retarded_from_keldysh(_A + _C, _B + _D, _w, _dw)',
            "note": 'normal, superposed structure exercising linearity across the whole grid',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([float(retarded_from_keldysh(_A, _B, _w, _dw)[_n // 2].real),'
                     ' float(retarded_from_keldysh(_A, _B, _w, _dw)[_n // 4].real)])'),
            "gold_call": ('np.array([float(_oracle_retarded_from_keldysh(_A, _B, _w, _dw)[_n // 2].real),'
                          ' float(_oracle_retarded_from_keldysh(_A, _B, _w, _dw)[_n // 4].real)])'),
            "note": 'contract, the dispersive real part at two separated frequencies',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: retarded_from_keldysh(_A, _B, _w, 0.0)),'
                     ' _raises(lambda: retarded_from_keldysh(_A[:10], _B, _w, _dw))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_retarded_from_keldysh(_A, _B, _w, 0.0)),'
                          ' _raises(lambda: _oracle_retarded_from_keldysh(_A[:10], _B, _w, _dw))])'),
            "note": 'contract, non-positive spacing and mismatched lengths must raise ValueError',
        },
    ]
