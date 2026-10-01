"""
Evaluate the constant that the fourth-order rate profile approaches once the two

coherence intervals stop knowing about each other.



Push the waiting time to infinity and every lineshape term that couples the two coherence

intervals drops out, so the doubly integrated kernel separates into a product of

single-interval integrals and the constant follows in closed form, without touching the

numerical waiting-time grid at all. Work out what each kernel leaves behind in that limit,

sum the two, and apply the same scale factor and unit conversion the profile uses. Evaluating

the profile at a large waiting time instead is not an acceptable substitute: the point of this

step is the closed form.



Physically this plateau is not a fourth-order transfer event at all. It is a forward hop

followed by an independent backward hop, two lowest-order processes that happen to occur in

sequence, and the kinetic model already accounts for it. It grows without bound as the

waiting window is extended, so it has to be removed before the waiting-time integral is taken.



The single-interval integral uses exactly the same grid and trapezoidal rule as the

second-order rate, so that the plateau computed here and the profile computed in the

previous step agree in the long-waiting-time limit to within the discretisation error.



Formulas

--------

S  = trapz(R(t), t) over the coherence grid          [fs, complex]

J_ang = 2 pi c * j_cm

C = A * J_ang^4 * Re( ... ) * 1e6                    [ps^-2]

with A the same signed integer as in the previous step and the bracket built from S alone.



Returns

-------

float, the plateau of the fourth-order rate profile in ps^-2 (negative for any non-zero

coupling)

Returns
-------
float, the plateau of the fourth-order rate profile in ps^-2 (negative for any non-zero coupling)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                     gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float) -> float:
    '''Long-waiting-time limit of the fourth-order rate profile.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of the coherence-time integral, in fs.

    Returns
    -------
    float
        The plateau in ps^-2.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if t_max_fs or dt_fs is non-positive or dt_fs exceeds t_max_fs.
    '''
    return plateau_ps2


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_S08_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _oracle_plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                             gap_cm, j_cm, t_max_fs, dt_fs):
    j = float(j_cm)
    if not np.isfinite(j):
        raise ValueError("j_cm must be finite")
    tmax, dt = float(t_max_fs), float(dt_fs)
    if not np.isfinite(tmax) or tmax <= 0.0:
        raise ValueError("t_max_fs must be finite and strictly positive")
    if not np.isfinite(dt) or dt <= 0.0 or dt > tmax:
        raise ValueError("dt_fs must be finite, strictly positive and at most t_max_fs")
    t = np.linspace(0.0, tmax, int(round(tmax / dt)) + 1)
    resp = _oracle_second_order_response(t, sigma_donor, sigma_acceptor, tau_fs,
                                         tau_static_fs, gap_cm)
    s = np.trapezoid(resp, t)
    j_ang = j * _S08_TWO_PI_C
    return float(-4.0 * j_ang ** 4 * (s * s + np.abs(s) ** 2).real * 1.0e6)


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
        # normal: the main problem at its mean gap
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 2.0))'},
        # normal: a gap out in the static wing
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 366.5, 26.0, 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 366.5, 26.0, 200.0, 2.0))'},
        # boundary: flipping the gap sign leaves the plateau unchanged
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0, 26.0, 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0, 26.0, 200.0, 2.0))'},
        # boundary: a zero coupling gives exactly zero plateau
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 0.0, 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 0.0, 200.0, 2.0))'},
        # edge: quartic in the coupling, so doubling j multiplies the plateau by sixteen
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 52.0, 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 52.0, 200.0, 2.0))'},
        # edge: a zero gap, where the response integral is purely real
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0, 26.0, 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0, 26.0, 200.0, 2.0))'},
        # edge: a coarser coherence grid shifts the plateau only slightly
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 10.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 10.0))'},
        # invalid: a non-finite coupling
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, float("inf"), 200.0, 2.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, float("inf"), 200.0, 2.0))'},
        # invalid: a non-positive grid spacing
        {"setup": base,
         "call": '_v(lambda: plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 0.0))',
         "gold_call": '_v(lambda: _oracle_plateau_constant(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 0.0))'},
    ]
