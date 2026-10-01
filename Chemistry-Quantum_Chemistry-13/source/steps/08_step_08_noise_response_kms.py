"""
Charge noise, dissipative response, and how far the drive pushes the molecule off balance

The two components of the charge-fluctuation propagator carry more physics than their difference

suggests. Read as spectral weights, one of them counts the processes in which the molecule absorbs

energy from a fluctuation at a given frequency and the other the processes in which it releases

energy; both are real and non-negative. Their symmetric combination is the charge noise spectrum,

which measures the total fluctuation activity at that frequency, and their antisymmetric

combination is the dissipative part of the charge response, which measures the imbalance between

absorption and emission.



In thermal equilibrium these two quantities are not independent: detailed balance ties the ratio

of the absorbing to the emitting weight to a single universal function of frequency and

temperature, the same for every molecule regardless of its level position or its coupling to the

electrodes. That is precisely the constraint a bias voltage destroys. The ratio of the two weights

therefore serves as a direct, dimensionless measure of how far the driven steady state has been

pushed from detailed balance, and it is worth computing on its own account: it takes the

equilibrium value where the drive has no influence, and departs from it, often by orders of

magnitude, where the drive dominates. A ratio that falls below one signals that emission has

overtaken absorption, which is the signature of an inverted population.



This step returns those quantities as functions of frequency. It does no further physics; the sign

conventions established in step 04 fix everything, and the only real work is extracting real

non-negative weights from pure imaginary propagators without dropping a factor or a sign.

Returns
-------
#     np.ndarray of shape (5, n), real: absorbing weight, emitting weight, noise, response, ratio  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def noise_response_kms(pi_lesser: np.ndarray, pi_greater: np.ndarray) -> np.ndarray:
    '''Absorbing and emitting spectral weights, noise, response, and their ratio.

    Parameters
    ----------
    pi_lesser, pi_greater : np.ndarray
        Lesser and greater charge-fluctuation propagators of one molecule, in eV^-1.

    Returns
    -------
    result : np.ndarray
        Real array of shape (5, n): row 0 the absorbing spectral weight, the greater propagator
        component multiplied by the imaginary unit; row 1 the emitting spectral weight, the lesser
        component multiplied by the same prefactor; row 2 their half-sum, the charge noise
        spectrum; row 3 their half-difference, the dissipative charge response, all in eV^-1; and
        row 4 the dimensionless ratio of the absorbing to the emitting weight. Both weights are
        non-negative by construction. The ratio is reported as 0.0 wherever the emitting weight
        falls below one ten-thousandth of its own peak, where its value would be set by round-off
        rather than by physics.

    Raises
    ------
    ValueError
        If the two input arrays have different lengths.
    '''
    return result  # placeholder

# ============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_noise_response_kms(pi_lesser, pi_greater):
    pl = np.asarray(pi_lesser, dtype=complex)
    pg = np.asarray(pi_greater, dtype=complex)
    if pl.shape != pg.shape:
        raise ValueError("pi_lesser and pi_greater must have the same shape")
    s_plus = np.maximum((1j * pg).real, 0.0)
    s_minus = np.maximum((1j * pl).real, 0.0)
    noise = 0.5 * (s_plus + s_minus)
    response = 0.5 * (s_plus - s_minus)
    ratio = np.zeros_like(s_minus)
    good = s_minus > 1e-4 * s_minus.max()
    ratio[good] = s_plus[good] / s_minus[good]
    return np.vstack([s_plus, s_minus, noise, response, ratio])

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
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
              "_EL, _EG = _bub(-0.50, 0.050, 0.0, 300.0, 0.025, 0.025, 0.5)")

    return [
        {
            "setup": _SETUP,
            "call": 'noise_response_kms(_PL, _PG)',
            "gold_call": '_oracle_noise_response_kms(_PL, _PG)',
            "note": 'normal, the driven molecule of the task',
        },
        {
            "setup": _SETUP,
            "call": 'noise_response_kms(_EL, _EG)',
            "gold_call": '_oracle_noise_response_kms(_EL, _EG)',
            "note": 'normal, an unbiased molecule where detailed balance still holds',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([float(noise_response_kms(_EL, _EG)[4][int(np.argmin(np.abs(_w - 0.05)))]),'
                     ' float(noise_response_kms(_PL, _PG)[4][int(np.argmin(np.abs(_w - 0.05)))])])'),
            "gold_call": ('np.array([float(_oracle_noise_response_kms(_EL, _EG)[4][int(np.argmin(np.abs(_w - 0.05)))]),'
                          ' float(_oracle_noise_response_kms(_PL, _PG)[4][int(np.argmin(np.abs(_w - 0.05)))])])'),
            "note": 'contract, the equilibrium ratio must follow detailed balance while the driven one need not',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([float(noise_response_kms(_PL, _PG)[0].min()),'
                     ' float(noise_response_kms(_PL, _PG)[1].min())])'),
            "gold_call": ('np.array([float(_oracle_noise_response_kms(_PL, _PG)[0].min()),'
                          ' float(_oracle_noise_response_kms(_PL, _PG)[1].min())])'),
            "note": 'contract, both spectral weights must be non-negative everywhere',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(noise_response_kms(_PL, _PG)[2] + noise_response_kms(_PL, _PG)[3]'
                     ' - noise_response_kms(_PL, _PG)[0]).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_noise_response_kms(_PL, _PG)[2] + _oracle_noise_response_kms(_PL, _PG)[3]'
                          ' - _oracle_noise_response_kms(_PL, _PG)[0]).max()])'),
            "note": 'contract, noise plus response must reconstruct the absorbing weight',
        },
        {
            "setup": _SETUP,
            "call": 'noise_response_kms(*_bub(-0.60, 0.080, 1.8, -300.0, 0.050, 0.030, 0.70))',
            "gold_call": '_oracle_noise_response_kms(*_bub(-0.60, 0.080, 1.8, -300.0, 0.050, 0.030, 0.70))',
            "note": 'edge, an inverted population drives the ratio below one',
        },
        {
            "setup": _SETUP,
            "call": 'noise_response_kms(_PL, _PG)[:4]',
            "gold_call": '_oracle_noise_response_kms(_PL, _PG)[:4]',
            "note": 'contract, the four spectral quantities compared at their own scale',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: noise_response_kms(_PL[:10], _PG))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_noise_response_kms(_PL[:10], _PG))])',
            "note": 'contract, mismatched lengths must raise ValueError',
        },
    ]
