"""
Run the whole audit at the prescribed configuration and return the eleven reported quantities in the order given below. The refinement multiplies both the number of grid points and the number of propagation steps; the 2001 equally spaced distances at which the adiabatic reference is sampled for the adiabatic-following measure do not change with it. Raise ValueError if refine is not a positive integer.

The audit compares, on one footing, what each of the three control knobs achieves over the same shortened propagation distance, and contrasts them with the linear ramp that the same medium would follow without any inverse engineering. The two equilibrium widths fix the endpoints of the prescribed history and are reported so that the compression ratio the protocols are asked to deliver is explicit.

Returns
-------
ndarray of shape (11,), float64: the final beam width under the nonlocal-length knob, under the Kerr knob and under the confinement knob; the initial and target equilibrium widths; the final width under a linear nonlocal-length ramp; the adiabatic reference width at the midpoint of that ramp; and the largest value, over the interior of those 2001 samples, of the magnitude of the centred three-point second difference of the adiabatic reference width divided by the product of that width with its squared oscillation frequency; the change in the effective potential from the initial to the target equilibrium width at the initial response length; and the accuracy of the nonlocal-length protocol against the medium's own self-trapped profile at the target response length, relaxed over 2000 imaginary-distance steps of size 0.01 on the same grid; and the shortest protocol length, found as in the previous step with its default bracket, for which the squared confinement strength demanded by the confinement-knob protocol stays non-negative everywhere.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shortcut_audit(refine: int) -> 'np.ndarray':
    """Run the whole audit at the prescribed configuration and return the eleven reported quantities in the order given below. The refinement multiplies both the number of grid points and the number of propagation steps; the 2001 equally spaced distances at which the adiabatic reference is sampled for the adiabatic-following measure do not change with it. Raise ValueError if refine is not a positive integer.

    Returns
    -------
    ndarray of shape (11,), float64: the final beam width under the nonlocal-length knob, under the Kerr knob and under the confinement knob; the initial and target equilibrium widths; the final width under a linear nonlocal-length ramp; the adiabatic reference width at the midpoint of that ramp; and the largest value, over the interior of those 2001 samples, of the magnitude of the centred three-point second difference of the adiabatic reference width divided by the product of that width with its squared oscillation frequency; the change in the effective potential from the initial to the target equilibrium width at the initial response length; and the accuracy of the nonlocal-length protocol against the medium's own self-trapped profile at the target response length, relaxed over 2000 imaginary-distance steps of size 0.01 on the same grid; and the shortest protocol length, found as in the previous step with its default bracket, for which the squared confinement strength demanded by the confinement-knob protocol stays non-negative everywhere.

    Raises
    ------
    ValueError
        If refine is not a positive integer.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _grid(npts: int, half_width: float) -> "np.ndarray":
    npts = int(npts); half_width = float(half_width)
    if npts < 8 or npts % 2 != 0:
        raise ValueError("npts must be an even integer of at least 8")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    dx = 2.0 * half_width / npts
    x = -half_width + dx * np.arange(npts)
    k = 2.0 * np.pi * np.fft.fftfreq(npts, d=dx)
    return np.stack([x, k])

def _oracle_shortcut_audit(refine: int) -> "np.ndarray":
    """End-to-end audit of the three inverse-engineered shortcut protocols.

    Slots, in order: the final beam width under the nonlocal-length knob, under the
    Kerr knob and under the confinement knob; the initial and target equilibrium
    widths; the final width under a linear nonlocal-length ramp; the adiabatic
    reference width at the midpoint of that ramp; the adiabatic-following ratio; the
    effective-potential change; the fidelity of the nonlocal-length protocol; and the shortest confinement-knob protocol length that stays everywhere guiding.
    """
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    power, gamma, alpha = 12.0, 0.2, 0.15
    sigma_i, sigma_f, zf = 4.0, 0.8, 2.2
    npts, half_width = 1024 * refine, 30.0
    nz = 2000 * refine

    g = _grid(npts, half_width)
    x, k = g[0], g[1]
    dx = x[1] - x[0]
    a_i = _oracle_equilibrium_width(power, gamma, alpha, sigma_i)
    a_f = _oracle_equilibrium_width(power, gamma, alpha, sigma_f)

    amp = np.sqrt(power / (np.sqrt(np.pi) * a_i))
    u0 = (amp * np.exp(-x ** 2 / (2.0 * a_i ** 2))).astype(complex)

    dz = zf / nz
    zc = (np.arange(nz) + 0.5) * dz
    widths = []
    u_sigma = None
    for knob in ("sigma", "gamma", "alpha2"):
        ctrl = np.array([_oracle_inverse_control(knob,
                                                 *_oracle_minimum_jerk_width(z, zf, a_i, a_f),
                                                 power, gamma, alpha, sigma_i) for z in zc])
        u = _oracle_propagate_field(u0, npts, half_width, ctrl, dz, gamma, alpha, sigma_i, knob)
        widths.append(_oracle_beam_width(u, x, dx))
        if knob == "sigma":
            u_sigma = u

    ramp = sigma_i + (sigma_f - sigma_i) * zc / zf
    u_ramp = _oracle_propagate_field(u0, npts, half_width, ramp, dz, gamma, alpha, sigma_i, "sigma")
    a_ref = _oracle_adiabatic_reference(0.5 * zf, zf, power, gamma, alpha, sigma_i, sigma_f)

    zs = np.linspace(0.0, zf, 2001)
    hs = zs[1] - zs[0]
    ac = np.array([_oracle_adiabatic_reference(z, zf, power, gamma, alpha, sigma_i, sigma_f)
                   for z in zs])
    acdd = (ac[2:] - 2.0 * ac[1:-1] + ac[:-2]) / hs ** 2
    s_mid = sigma_i + (sigma_f - sigma_i) * zs[1:-1] / zf
    om2 = np.array([_oracle_width_curvature(a, power, gamma, alpha, s)
                    for a, s in zip(ac[1:-1], s_mid)])
    ratio = float(np.max(np.abs(acdd) / (om2 * ac[1:-1])))

    depth = _oracle_potential_change(a_i, a_f, power, gamma, alpha, sigma_i)

    target = _oracle_stationary_soliton(npts, half_width, power, gamma, alpha,
                                        sigma_f, 0.01, 2000)
    fid = _oracle_overlap_fidelity(u_sigma, target, dx)

    onset = _oracle_antiguiding_onset(power, gamma, alpha, sigma_i, sigma_f)

    return np.array([widths[0], widths[1], widths[2], a_i, a_f,
                     _oracle_beam_width(u_ramp, x, dx), a_ref, ratio, depth, fid,
                     onset], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "shortcut_audit(1)",
                    "gold_call": "_oracle_shortcut_audit(1)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "shortcut_audit(2)",
                    "gold_call": "_oracle_shortcut_audit(2)"
            },
            {
                    "setup": "import numpy as np\ndef ordering(fn):\n    v = fn(1)\n    return float(v[3] > v[4] and v[1] > v[0] > v[2])",
                    "call": "ordering(shortcut_audit)",
                    "gold_call": "ordering(_oracle_shortcut_audit)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        shortcut_audit(0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_shortcut_audit(0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
