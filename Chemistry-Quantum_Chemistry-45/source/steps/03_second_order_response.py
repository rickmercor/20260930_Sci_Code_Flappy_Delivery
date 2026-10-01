"""
Assemble the complex response function whose time integral gives the standard

lowest-order incoherent transfer rate between the two chromophores.



Once the donor-acceptor coherence has been reduced to a single dephasing lineshape, the

response of the pair is the product of two factors: a phase that winds at the mean gap

between the two site energies, and the decay envelope exp(-g(t)) built in step 1. The

response is normalised so that R(0) = 1 exactly, and it carries no factor of the coupling:

the coupling enters later, where the rate is formed.



The sign convention matters and is fixed here: the gap is defined as acceptor minus donor,

and the phase advances as exp(-i omega t) with omega = 2 pi c times that gap. Reversing the

sign of the gap conjugates the response, which is what makes the backward rate equal to the

forward rate at infinite temperature.



Formulas

--------

omega = 2 pi c * gap_cm,  c = 2.99792458e-5 cm/fs

R(t) = exp(-g(t)) * exp(-i omega t)



Returns

-------

np.ndarray of shape (len(t_fs),), complex dtype

Returns
-------
np.ndarray of shape (len(t_fs),), complex dtype
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def second_order_response(t_fs, sigma_donor, sigma_acceptor, tau_fs,
                          tau_static_fs: float, gap_cm: float) -> np.ndarray:
    '''Complex second-order transfer response of the donor-acceptor pair.

    Parameters
    ----------
    t_fs : array_like
        Times in fs; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor excitation energy gap, in cm^-1. May be any finite
        real number, including negative.

    Returns
    -------
    np.ndarray
        Shape (len(t_fs),) complex array with R(0) = 1.

    Raises
    ------
    ValueError
        On any of the bath or time faults listed for the lineshape step, or if
        gap_cm is not finite.
    '''
    return response


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_S03_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _oracle_second_order_response(t_fs, sigma_donor, sigma_acceptor, tau_fs,
                                  tau_static_fs, gap_cm):
    gap = float(gap_cm)
    if not np.isfinite(gap):
        raise ValueError("gap_cm must be finite")
    t = np.atleast_1d(np.asarray(t_fs, dtype=float))
    g = _oracle_dephasing_lineshape(t, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    return np.exp(-g) * np.exp(-1j * gap * _S03_TWO_PI_C * t)


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
        'TG = np.arange(0.0, 220.0 + 0.5, 1.0)\n'
    )
    base = "import numpy as np\n" + guard
    return [
        # normal: the main problem's bath and mean gap on the coherence grid
        {"setup": base,
         "call": '_v(lambda: second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # normal: a gap displaced far into the wing of the static distribution
        {"setup": base,
         "call": '_v(lambda: second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 366.5))',
         "gold_call": '_v(lambda: _oracle_second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 366.5))'},
        # boundary: R(0) must be exactly 1 + 0j
        {"setup": base,
         "call": '_v(lambda: second_order_response([0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_second_order_response([0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # boundary: a zero gap leaves a purely real, purely decaying response
        {"setup": base,
         "call": '_v(lambda: second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0))',
         "gold_call": '_v(lambda: _oracle_second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0))'},
        # edge: a negative gap conjugates the response relative to the positive gap
        {"setup": base,
         "call": '_v(lambda: second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0))',
         "gold_call": '_v(lambda: _oracle_second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0))'},
        # edge: with every mode static the envelope is 1 and only the phase survives
        {"setup": base,
         "call": '_v(lambda: second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1.0, 120.0))'},
        # invalid: a non-finite gap
        {"setup": base,
         "call": '_v(lambda: second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, float("nan")))',
         "gold_call": '_v(lambda: _oracle_second_order_response(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, float("nan")))'},
        # invalid: a negative time
        {"setup": base,
         "call": '_v(lambda: second_order_response([-5.0, 0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_second_order_response([-5.0, 0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
    ]
