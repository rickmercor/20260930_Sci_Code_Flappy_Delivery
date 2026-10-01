"""
Collect the bath components that were excluded from the lineshape and turn them

into a single inhomogeneous width for the donor-acceptor energy gap.



A component whose correlation time reaches or exceeds the threshold does not dephase the

donor-acceptor coherence; it simply offsets the gap by an amount that is frozen for the whole

transfer event. Because the donor and acceptor baths are uncorrelated, the variances of those

frozen offsets add, and because the transfer depends only on the difference of the two site

energies, the donor and acceptor contributions add on the same footing.



The result is the standard deviation of a Gaussian distribution of the gap. Every quantity

here stays in cm^-1: no conversion to angular frequency is applied at this step, because the

width is later added directly to the mean gap, which is also quoted in cm^-1.



Formulas

--------

sigma_static = sqrt( sum_{j : tau_j >= tau_static} ( sigma_D,j^2 + sigma_A,j^2 ) )



Returns

-------

float, the RMS static spread of the donor-acceptor gap in cm^-1 (zero when no component

is static)

Returns
-------
float, the RMS static spread of the donor-acceptor gap in cm^-1 (zero when no component is static)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def static_gap_width(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float) -> float:
    '''RMS inhomogeneous width of the donor-acceptor gap from the near-static bath modes.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitude of each bath component, in cm^-1.
        Same length as tau_fs; entries must be finite and >= 0.
    tau_fs : array_like
        Bath correlation time of each component, in fs; entries must be finite and > 0.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs count as static. Must be finite and > 0.

    Returns
    -------
    float
        Standard deviation of the gap distribution in cm^-1; exactly 0.0 when no
        component reaches the threshold.

    Raises
    ------
    ValueError
        If the three component lists differ in length, if any list is empty, if any
        entry is non-finite, if any sigma is negative, if any tau is non-positive, or
        if tau_static_fs is non-positive.
    '''
    return width_cm


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s02_check(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd = np.atleast_1d(np.asarray(sigma_donor, dtype=float))
    sa = np.atleast_1d(np.asarray(sigma_acceptor, dtype=float))
    ta = np.atleast_1d(np.asarray(tau_fs, dtype=float))
    if sd.size == 0 or sd.size != sa.size or sd.size != ta.size:
        raise ValueError("sigma_donor, sigma_acceptor and tau_fs must be non-empty and equal length")
    if not (np.all(np.isfinite(sd)) and np.all(np.isfinite(sa)) and np.all(np.isfinite(ta))):
        raise ValueError("bath parameters must all be finite")
    if np.any(sd < 0.0) or np.any(sa < 0.0):
        raise ValueError("fluctuation amplitudes must be non-negative")
    if np.any(ta <= 0.0):
        raise ValueError("correlation times must be strictly positive")
    tstat = float(tau_static_fs)
    if not np.isfinite(tstat) or tstat <= 0.0:
        raise ValueError("tau_static_fs must be finite and strictly positive")
    return sd, sa, ta, tstat


def _oracle_static_gap_width(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd, sa, ta, tstat = _s02_check(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    mask = ta >= tstat
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.sum(sd[mask] ** 2 + sa[mask] ** 2)))


#

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = (
        'import copy\n'
        'def _fp(a):\n'
        '    a = np.asarray(a)\n'
        '    if np.iscomplexobj(a):\n'
        '        a = np.concatenate([np.real(a).ravel(), np.imag(a).ravel()])\n'
        '    a = np.asarray(a, dtype=float).ravel()\n'
        '    w = 1.0 + (np.arange(a.size) % 97) / 97.0\n'
        '    return float(np.sum(w * a / (1.0 + np.abs(a))))\n'
        'def _v(fn):\n'
        '    try:\n'
        '        out = fn()\n'
        '        if isinstance(out, (tuple, list)):\n'
        '            return float(sum(_fp(x) * (i + 1) for i, x in enumerate(out)))\n'
        '        return _fp(out)\n'
        '    except ValueError:\n'
        '        return -1.2345e6\n'
        '    except Exception:\n'
        '        return -9.8765e6\n'
        'SD = [90.0, 110.0, 70.0]\n'
        'SA = [90.0, 110.0, 70.0]\n'
        'TAU = [40.0, 180.0, 3000.0]\n'
    )
    base = "import numpy as np\n" + guard
    return [
        # normal: the three-component bath of the main problem, one static mode
        {"setup": base,
         "call": '_v(lambda: static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
        # normal: unequal donor and acceptor static amplitudes
        {"setup": base,
         "call": '_v(lambda: static_gap_width([90.0, 55.0], [90.0, 130.0], [40.0, 4000.0], 1000.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width([90.0, 55.0], [90.0, 130.0], [40.0, 4000.0], 1000.0))'},
        # boundary: threshold exactly on a component tau, which therefore counts as static
        {"setup": base,
         "call": '_v(lambda: static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 180.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 180.0))'},
        # boundary: no component reaches the threshold, so the width is exactly zero
        {"setup": base,
         "call": '_v(lambda: static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 5000.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 5000.0))'},
        # edge: every component is static
        {"setup": base,
         "call": '_v(lambda: static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1.0))'},
        # edge: two static components add in quadrature, not linearly
        {"setup": base,
         "call": '_v(lambda: static_gap_width([30.0, 40.0], [0.0, 0.0], [2000.0, 5000.0], 1000.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width([30.0, 40.0], [0.0, 0.0], [2000.0, 5000.0], 1000.0))'},
        # invalid: a negative fluctuation amplitude
        {"setup": base,
         "call": '_v(lambda: static_gap_width([-90.0, 110.0, 70.0], copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width([-90.0, 110.0, 70.0], copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
        # invalid: a non-positive threshold
        {"setup": base,
         "call": '_v(lambda: static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 0.0))',
         "gold_call": '_v(lambda: _oracle_static_gap_width(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 0.0))'},
    ]
