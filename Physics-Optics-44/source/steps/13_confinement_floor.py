"""
Return the smallest value the squared confinement strength takes anywhere along a compression protocol of length zf. The width is driven from the equilibrium width the medium supports at sigma_i to the equilibrium width it supports at sigma_f, following the minimum-jerk interpolation between them on the closed interval from zero to zf. At every propagation distance the squared confinement strength is the one value that reproduces that width together with its second derivative, with the nonlocal length held fixed at sigma_i and the Kerr coefficient held at gamma throughout: the confinement is the only parameter that moves. Search the closed interval including both endpoints and locate the minimum to an absolute accuracy of 1e-10 in the propagation distance. A negative result means the protocol demands a locally antiguiding profile somewhere along the way. Raise ValueError if zf is not strictly positive.

A parabolic transverse index profile guides light only while its curvature has the focusing sign. When a compression is demanded over a short distance the width must be decelerated hard near the end of the protocol, and the confinement required to supply that deceleration can overshoot through zero and change sign. Where that happens the index profile has a minimum on axis rather than a maximum, so it expels the beam instead of confining it, and the protocol is no longer physically realisable as a guiding structure. The smallest value reached along the protocol is therefore the quantity that decides realisability, and because the demanded profile is smooth and turns over in the interior of the interval, reading it off a coarse sample of propagation distances is not enough.

Returns
-------
float: the minimum squared confinement strength over the protocol, negative when the protocol turns antiguiding.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def confinement_floor(zf: float, power: float, gamma: float, alpha: float, sigma_i: float, sigma_f: float) -> float:
    """Return the smallest value the squared confinement strength takes anywhere along a compression protocol of length zf. The width is driven from the equilibrium width the medium supports at sigma_i to the equilibrium width it supports at sigma_f, following the minimum-jerk interpolation between them on the closed interval from zero to zf. At every propagation distance the squared confinement strength is the one value that reproduces that width together with its second derivative, with the nonlocal length held fixed at sigma_i and the Kerr coefficient held at gamma throughout: the confinement is the only parameter that moves. Search the closed interval including both endpoints and locate the minimum to an absolute accuracy of 1e-10 in the propagation distance. A negative result means the protocol demands a locally antiguiding profile somewhere along the way. Raise ValueError if zf is not strictly positive.

    Returns
    -------
    float: the minimum squared confinement strength over the protocol, negative when the protocol turns antiguiding.

    Raises
    ------
    ValueError
        If zf is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_confinement_floor(zf: float, power: float, gamma: float, alpha: float,
                              sigma_i: float, sigma_f: float) -> float:
    """Smallest waveguide strength the confinement-knob shortcut of length zf demands."""
    zf = float(zf)
    if zf <= 0.0:
        raise ValueError("zf must be positive")
    a_i = _oracle_equilibrium_width(power, gamma, alpha, sigma_i)
    a_f = _oracle_equilibrium_width(power, gamma, alpha, sigma_f)

    val = lambda z: _oracle_inverse_control("alpha2",
                                           *_oracle_minimum_jerk_width(z, zf, a_i, a_f),
                                           power, gamma, alpha, sigma_i)
    zs = np.linspace(0.0, zf, 401)
    vs = np.array([val(z) for z in zs])
    j = int(np.argmin(vs))
    lo = zs[max(j - 1, 0)]
    hi = zs[min(j + 1, zs.size - 1)]
    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    for _ in range(200):                      # golden section: no scipy, deterministic
        if hi - lo <= 1e-12:
            break
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        if val(c) < val(d):
            hi = d
        else:
            lo = c
    return float(min(val(0.5 * (lo + hi)), vs[0], vs[-1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "confinement_floor(2.2, 12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_confinement_floor(2.2, 12.0, 0.2, 0.15, 4.0, 0.8)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "confinement_floor(1.0, 12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_confinement_floor(1.0, 12.0, 0.2, 0.15, 4.0, 0.8)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "confinement_floor(0.6, 12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_confinement_floor(0.6, 12.0, 0.2, 0.15, 4.0, 0.8)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "confinement_floor(0.35, 20.0, 0.1, 0.1, 5.0, 0.5)",
                    "gold_call": "_oracle_confinement_floor(0.35, 20.0, 0.1, 0.1, 5.0, 0.5)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "confinement_floor(9.0, 20.0, 0.1, 0.1, 5.0, 0.5)",
                    "gold_call": "_oracle_confinement_floor(9.0, 20.0, 0.1, 0.1, 5.0, 0.5)"
            },
            {
                    "setup": "import numpy as np\ndef long_protocol(fn):\n    return float(fn(40.0, 12.0, 0.2, 0.15, 4.0, 0.8))",
                    "call": "long_protocol(confinement_floor)",
                    "gold_call": "long_protocol(_oracle_confinement_floor)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        confinement_floor(0.0, 12.0, 0.2, 0.15, 4.0, 0.8)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_confinement_floor(0.0, 12.0, 0.2, 0.15, 4.0, 0.8)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
