"""
The dispersion interaction energy of the driven pair

Everything the interaction energy needs is now available. The energy of the coupling between the

two molecules follows from the correlation self-energy that the coupling generates, and once that

self-energy is written out at second order the Green's functions reassemble into the

charge-fluctuation propagators of the two molecules. What survives is a double frequency integral:

one frequency belongs to a fluctuation on the first molecule, the other to a fluctuation on the

second, the two propagators enter bilinearly, and the pair is weighted by the reciprocal of the

sum of the two frequencies, which is the energy denominator of the correlated fluctuation. The

whole expression carries two powers of the coupling constant, as it must at this order.



Which components of the two propagators pair with which, and with what relative sign, is the

entire content of this step and is what separates a correct nonequilibrium treatment from an

equilibrium one. In thermal equilibrium the two propagators of a molecule are locked together by

detailed balance, the combination collapses onto the familiar attractive result, and the sign of

the energy is fixed no matter what the molecules are. Away from equilibrium that lock is gone, the

two contributions are free to compete, and neither the magnitude nor the sign of the answer can be

taken for granted.



The weight diverges on the surface where the two frequencies sum to zero. That surface is

integrable rather than singular: the frequency-reversal relation established in step 04 makes the

numerator vanish there identically, for identical molecules, so no regulator is needed and none

should be introduced. On the task grid that surface is a set of grid points along one

anti-diagonal, and those points carry no weight. Note also that the double sum has the same value

as a single sum over the frequency sum, because the weight depends on the two frequencies only

through their total; exploiting that turns an expensive quadratic double loop into one convolution.

Returns
-------
#     float: the dispersion interaction energy in eV, negative when attractive  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dispersion_energy(pi_lesser_a: np.ndarray, pi_greater_a: np.ndarray, pi_lesser_b: np.ndarray,
                      pi_greater_b: np.ndarray, w: np.ndarray, dw: float,
                      u_coupling: float) -> float:
    '''Dispersion interaction energy of two Coulomb-coupled molecules in a steady state.

    Parameters
    ----------
    pi_lesser_a, pi_greater_a : np.ndarray
        Lesser and greater charge-fluctuation propagators of molecule A, in eV^-1.
    pi_lesser_b, pi_greater_b : np.ndarray
        Lesser and greater charge-fluctuation propagators of molecule B, in eV^-1.
    w : np.ndarray
        Real frequency grid in eV of the form w_j = (j - n/2) * dw.
    dw : float
        Grid spacing in eV; positive.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.

    Returns
    -------
    result : float
        The dispersion interaction energy in eV. Negative values are attractive.

    Raises
    ------
    ValueError
        If dw is not positive, if u_coupling is negative, or if the four propagator arrays and w
        do not all have the same length.
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


def _oracle_dispersion_energy(pi_lesser_a, pi_greater_a, pi_lesser_b, pi_greater_b,
                              w, dw, u_coupling):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    arrs = [np.asarray(a, dtype=complex) for a in
            (pi_lesser_a, pi_greater_a, pi_lesser_b, pi_greater_b)]
    wv = np.asarray(w, dtype=float)
    if len({a.shape[-1] for a in arrs} | {len(wv)}) != 1:
        raise ValueError("all propagator arrays and w must have the same length")
    pla, pga, plb, pgb = arrs
    n = len(wv)
    numerator = _fft_linear_convolve(pla, plb) - _fft_linear_convolve(pga, pgb)
    total = np.arange(2 * n - 1) - n
    keep = total != 0
    acc = (numerator[keep] / (total[keep] * dw)).sum()
    return float((-(u_coupling ** 2) * acc * dw * dw / (2.0 * np.pi) ** 2).real)

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_KB = 8.617333262e-5\n"
          "_n = 1024\n"
          "_dw = 20.0 / _n\n"
          "_w = (np.arange(_n) - _n // 2) * _dw\n"
          "def _fc(a, b):\n"
          "    m = len(a) + len(b) - 1\n"
          "    s = 1\n"
          "    while s < m: s *= 2\n"
          "    return np.fft.ifft(np.fft.fft(a, s) * np.fft.fft(b, s))[:m]\n"
          "def _corr(a, b):\n"
          "    h = _n // 2\n"
          "    return _fc(np.asarray(a, complex), np.asarray(b, complex)[::-1])[np.arange(_n) + (_n - 1 - h)] * _dw / (2 * np.pi)\n"
          "def _bub(eps, gam, V, T, gl, gr, fr):\n"
          "    mu_l = fr * V\n"
          "    mu_r = -(1.0 - fr) * V\n"
          "    f1 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_l) / (2.0 * _KB * T), -400, 400)))\n"
          "    f2 = 0.5 * (1.0 - np.tanh(np.clip((_w - mu_r) / (2.0 * _KB * T), -400, 400)))\n"
          "    sl = 1j * (gl * f1 + gr * f2)\n"
          "    sg = -1j * (gl * (1 - f1) + gr * (1 - f2))\n"
          "    gret = 1.0 / (_w - eps + 0.5j * gam)\n"
          "    a = gret * sl * np.conj(gret)\n"
          "    b = gret * sg * np.conj(gret)\n"
          "    return -1j * _corr(a, b), -1j * _corr(b, a)\n"
          "_PL, _PG = _bub(-0.7800536, 0.080, 2.0, 300.0, 0.050, 0.030, 0.70)\n"
          "_EL, _EG = _bub(-0.50, 0.050, 0.0, 300.0, 0.025, 0.025, 0.5)\n"
          "_IL, _IG = _bub(-0.60, 0.080, 1.8, -300.0, 0.050, 0.030, 0.70)")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'np.array([dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 0.90)])',
            "gold_call": 'np.array([_oracle_dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 0.90)])',
            "note": 'normal, the driven pair of the task',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([dispersion_energy(_EL, _EG, _EL, _EG, _w, _dw, 1.00)])',
            "gold_call": 'np.array([_oracle_dispersion_energy(_EL, _EG, _EL, _EG, _w, _dw, 1.00)])',
            "note": 'boundary, an unbiased pair in equilibrium must come out attractive',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([dispersion_energy(_IL, _IG, _IL, _IG, _w, _dw, 1.00)])',
            "gold_call": 'np.array([_oracle_dispersion_energy(_IL, _IG, _IL, _IG, _w, _dw, 1.00)])',
            "note": 'edge, an inverted population reverses the sign of the interaction',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 0.0)])',
            "gold_call": 'np.array([_oracle_dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 0.0)])',
            "note": 'boundary, zero coupling must give exactly zero energy',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 2.00)'
                     ' / dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 0.90)])'),
            "gold_call": ('np.array([_oracle_dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 2.00)'
                          ' / _oracle_dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, 0.90)])'),
            "note": 'contract, the energy must scale as the square of the coupling',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([dispersion_energy(*_bub(-1.20, 0.200, 3.5, 450.0, 0.120, 0.080, 0.35), *_bub(-1.20, 0.200, 3.5, 450.0, 0.120, 0.080, 0.35), _w, _dw, 1.40)])',
            "gold_call": 'np.array([_oracle_dispersion_energy(*_bub(-1.20, 0.200, 3.5, 450.0, 0.120, 0.080, 0.35), *_bub(-1.20, 0.200, 3.5, 450.0, 0.120, 0.080, 0.35), _w, _dw, 1.40)])',
            "note": 'normal, a broad strongly driven pair at large coupling',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: dispersion_energy(_PL, _PG, _PL, _PG, _w, 0.0, 1.0)),'
                     ' _raises(lambda: dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, -1.0)),'
                     ' _raises(lambda: dispersion_energy(_PL[:10], _PG, _PL, _PG, _w, _dw, 1.0))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_dispersion_energy(_PL, _PG, _PL, _PG, _w, 0.0, 1.0)),'
                          ' _raises(lambda: _oracle_dispersion_energy(_PL, _PG, _PL, _PG, _w, _dw, -1.0)),'
                          ' _raises(lambda: _oracle_dispersion_energy(_PL[:10], _PG, _PL, _PG, _w, _dw, 1.0))])'),
            "note": 'contract, bad spacing, negative coupling and mismatched lengths must raise ValueError',
        },
    ]
