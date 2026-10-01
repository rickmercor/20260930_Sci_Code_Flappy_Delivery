"""
Step 08 (FINAL ORCHESTRATOR): run the whole chain and return the single scalar
the task asks for.

Step 08 (FINAL ORCHESTRATOR): run the whole chain and return the single scalar
the task asks for.

## Contract

Given the bare steering strength, the sensor accuracy, the Peclet number and
the three-valued sensory modulation profile, return the steady-state mean rate
of entropy production associated with the sensor's alignment torque. The
return is a single Python float in the units of the problem statement.

## The profile

The modulation takes the value 0 on the disc r < r_inner, the value gamma_mid
on the annulus r_inner <= r < r_outer, and the value 1 for r >= r_outer. The
three regions are ordered from the origin outwards throughout the pipeline, so
the modulation array handed to step 01 is exactly

[0.0, gamma_mid, 1.0]

and the interface array handed to steps 04 to 07 is exactly

[r_inner, r_outer]

## The chain

Step 01 turns the modulation profile into the per-region sensor statistics;
step 03 turns the mean cosines into the per-region transport parameters,
evaluating the closure coefficients of step 02 at the renormalised strengths;
steps 04 and 05 match and normalise the profile; step 06 evaluates that
normalised profile at any radius, and step 07 supplies the inverse-square
moment of each region beyond the innermost, which is the radial integral of
the very profile step 06 returns. The rate is then assembled from those
moments and the per-region contribution weights that step 01 returns, divided
by the product of the sensor accuracy and the Peclet number. The innermost
region carries no weight and contributes nothing.

## Admissible range

The closure coefficients of step 02 are defined only for renormalised
strengths in the closed interval [0, 1.2]. Each region's renormalised strength
is the bare strength multiplied by that region's own mean cosine, which never
exceeds one, so the outermost region carries the largest of them. A bare
strength whose renormalised value leaves that interval in any region is
outside the admissible range and raises ValueError instead of returning a
number.

## Defaults

The signature carries the benchmark configuration as its default arguments, so
calling the function with no arguments reproduces the reported answer.

## Inputs

kappa_tilde : float, bare steering strength, finite and strictly positive.
sigma_phi_sq : float, sensor accuracy, finite and strictly positive.
peclet : float, Peclet number, finite and strictly positive.
r_inner : float, inner interface radius, finite and strictly positive.
r_outer : float, outer interface radius, finite and strictly greater than
r_inner.
gamma_mid : float, annulus modulation amplitude, finite and in [0, 1].

## Returns

rate : float, the steady-state sensing entropy production rate.

Returns
-------
rate : float, the steady-state sensing entropy production rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sensing_entropy_production_rate(kappa_tilde=0.48, sigma_phi_sq=0.30,
                                    peclet=2.5, r_inner=0.06, r_outer=0.55,
                                    gamma_mid=0.50):
    """Return the steady-state sensing entropy production rate.

    Runs the whole chain on the three-region modulation profile
    [0, gamma_mid, 1] with interfaces [r_inner, r_outer] and returns the
    single scalar the task asks for.

    Parameters
    ----------
    kappa_tilde : float
        Bare steering strength. Must be finite and strictly positive.
    sigma_phi_sq : float
        Sensor accuracy. Must be finite and strictly positive.
    peclet : float
        Peclet number. Must be finite and strictly positive.
    r_inner : float
        Radius at which the alignment torque switches on. Must be finite and
        strictly positive.
    r_outer : float
        Radius at which the alignment torque reaches full strength. Must be
        finite and strictly greater than r_inner.
    gamma_mid : float
        Modulation amplitude on the annulus. Must be finite and lie in the
        closed interval [0, 1].

    Returns
    -------
    rate : float
        The steady-state mean rate of entropy production associated with the
        sensor's alignment torque, in the units of the problem statement.

    Raises
    ------
    ValueError
        If kappa_tilde, sigma_phi_sq, peclet or r_inner is not a finite,
        strictly positive real number; if r_outer is not finite and strictly
        greater than r_inner; if gamma_mid is not finite and in [0, 1]; or if
        the renormalised strength of any region leaves the closed interval
        [0, 1.2] on which the closure coefficients are defined.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _profile_quadrature(transport, radii):
    """Integrate each sensing region on panels uniform in log radius."""
    nodes, gl_weights = np.polynomial.legendre.leggauss(48)
    tr = np.asarray(transport, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    rules = []
    for j in range(1, tr.shape[0]):
        lo = float(rad[j - 1])
        if j < tr.shape[0] - 1:
            hi = float(rad[j])
        else:
            decay = float(tr[j, 0])
            if decay <= 0.0:
                raise ValueError("the outermost region does not decay")
            hi = lo + 100.0 / decay
        log_lo, log_hi = np.log(lo), np.log(hi)
        panels = max(1, int(np.ceil(log_hi - log_lo)))
        edges = np.linspace(log_lo, log_hi, panels + 1)
        r_all, w_all = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            half = 0.5 * (b - a)
            r = np.exp(0.5 * (a + b) + half * nodes)
            r_all.append(r)
            w_all.append(half * gl_weights * r)
        rules.append((np.concatenate(r_all), np.concatenate(w_all)))
    return rules


def _oracle_sensing_entropy_production_rate(kappa_tilde=0.48,
                                            sigma_phi_sq=0.30, peclet=2.5,
                                            r_inner=0.06, r_outer=0.55,
                                            gamma_mid=0.50):
    """Reference implementation of sensing_entropy_production_rate."""
    scalars = (("kappa_tilde", kappa_tilde), ("sigma_phi_sq", sigma_phi_sq),
               ("peclet", peclet), ("r_inner", r_inner), ("r_outer", r_outer),
               ("gamma_mid", gamma_mid))
    values = {}
    for name, value in scalars:
        if isinstance(value, bool):
            raise ValueError("%s must be a real number, got a bool" % name)
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be a real number" % name)
        if not np.isfinite(values[name]):
            raise ValueError("%s must be finite" % name)
    for name in ("kappa_tilde", "sigma_phi_sq", "peclet", "r_inner"):
        if values[name] <= 0.0:
            raise ValueError("%s must be strictly positive" % name)
    if values["r_outer"] <= values["r_inner"]:
        raise ValueError("r_outer must be strictly greater than r_inner")
    if not (0.0 <= values["gamma_mid"] <= 1.0):
        raise ValueError("gamma_mid must lie in [0, 1]")

    gamma_values = np.array([0.0, values["gamma_mid"], 1.0], dtype=np.float64)
    radii = np.array([values["r_inner"], values["r_outer"]], dtype=np.float64)

    stats = _oracle_sensor_alignment_statistics(values["sigma_phi_sq"],
                                                gamma_values)
    mean_cos = stats[:, 1]

    transport = _oracle_region_transport_parameters(values["kappa_tilde"],
                                                    mean_cos, values["peclet"])
    amplitudes = _oracle_matched_density_amplitudes(transport, radii)
    summary = _oracle_radial_normalisation(transport, amplitudes, radii)
    normalisation = float(summary[0])
    moments = _oracle_conditional_inverse_square_moments(
        transport, amplitudes, radii, normalisation)

    # The two restricted averages are integrals of the matched profile that
    # step 06 evaluates, so the density readout is what certifies them before
    # the rate is assembled from them. Integrating that readout region by
    # region must reproduce step 07's closed-form values; a profile that is
    # wrong anywhere, replaced by zeros, or replaced by any constant fails
    # here and no rate is returned. Step 02 is called directly as well, since
    # it is otherwise reached only through step 03.
    _oracle_closure_series_coefficients(values["kappa_tilde"] * mean_cos)
    profile_moments = np.empty(moments.size, dtype=np.float64)
    for j, (r_nodes, r_weights) in enumerate(
            _profile_quadrature(transport, radii)):
        density = _oracle_radial_density_profile(transport, amplitudes, radii,
                                                 normalisation, r_nodes)
        profile_moments[j] = float(np.sum(r_weights * density
                                          / (r_nodes * r_nodes)))
    if not np.all(np.isfinite(profile_moments)):
        raise ValueError("the matched profile is not integrable over a "
                         "sensing region")
    if not np.all(np.abs(profile_moments - moments)
                  <= 1e-7 * np.abs(moments) + 1e-15):
        raise ValueError("the radial integral of the matched density does not "
                         "reproduce the restricted inverse-square averages")

    weights = stats[1:, 3]
    rate = float(np.sum(weights * moments)
                 / (values["sigma_phi_sq"] * values["peclet"]))

    return rate

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the cases for :func:`sensing_entropy_production_rate`."""
    cases = []

    # Integration test 1: the benchmark configuration, called with no
    # arguments, which must reproduce the reported answer.
    cases.append({
        "setup": "import numpy as np",
        "call": "sensing_entropy_production_rate()",
        "gold_call": "_oracle_sensing_entropy_production_rate()",
    })

    # Integration test 2: a different configuration entirely -- sharper sensor,
    # stronger steering, higher Peclet number and a wider annulus.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.9\n"
            "sigma_phi_sq = 0.12\n"
            "peclet = 25.0\n"
            "r_inner = 0.05\n"
            "r_outer = 0.9\n"
            "gamma_mid = 0.3"
        ),
        "call": (
            "sensing_entropy_production_rate(kappa_tilde, sigma_phi_sq, "
            "peclet, r_inner, r_outer, gamma_mid)"
        ),
        "gold_call": (
            "_oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)"
        ),
    })

    # Integration test 3: a weakly steered, poorly sensing agent at low Peclet
    # number, where the localisation length is long and the annulus is narrow.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.25\n"
            "sigma_phi_sq = 0.8\n"
            "peclet = 3.0\n"
            "r_inner = 0.15\n"
            "r_outer = 0.4\n"
            "gamma_mid = 0.75"
        ),
        "call": (
            "sensing_entropy_production_rate(kappa_tilde, sigma_phi_sq, "
            "peclet, r_inner, r_outer, gamma_mid)"
        ),
        "gold_call": (
            "_oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)"
        ),
    })

    # Boundary case: full modulation on the annulus, so the profile collapses
    # onto a single interface at r_inner.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.48\n"
            "sigma_phi_sq = 0.30\n"
            "peclet = 2.5\n"
            "r_inner = 0.06\n"
            "r_outer = 0.55\n"
            "gamma_mid = 1.0"
        ),
        "call": (
            "sensing_entropy_production_rate(kappa_tilde, sigma_phi_sq, "
            "peclet, r_inner, r_outer, gamma_mid)"
        ),
        "gold_call": (
            "_oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)"
        ),
    })

    # Edge case: zero modulation on the annulus, so the profile collapses onto
    # a single interface at r_outer instead, and the annulus contributes
    # nothing to the rate.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.48\n"
            "sigma_phi_sq = 0.30\n"
            "peclet = 2.5\n"
            "r_inner = 0.06\n"
            "r_outer = 0.55\n"
            "gamma_mid = 0.0"
        ),
        "call": (
            "sensing_entropy_production_rate(kappa_tilde, sigma_phi_sq, "
            "peclet, r_inner, r_outer, gamma_mid)"
        ),
        "gold_call": (
            "_oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)"
        ),
    })

    # Edge case: an annulus of vanishing width, where the two interfaces almost
    # coincide and the annulus amplitude must stop mattering.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.48\n"
            "sigma_phi_sq = 0.30\n"
            "peclet = 2.5\n"
            "r_inner = 0.2\n"
            "r_outer = 0.2000001\n"
            "gamma_mid = 0.4"
        ),
        "call": (
            "sensing_entropy_production_rate(kappa_tilde, sigma_phi_sq, "
            "peclet, r_inner, r_outer, gamma_mid)"
        ),
        "gold_call": (
            "_oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)"
        ),
    })

    # Invalid input: an outer radius that does not exceed the inner one.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.48\n"
            "sigma_phi_sq = 0.30\n"
            "peclet = 2.5\n"
            "r_inner = 0.55\n"
            "r_outer = 0.06\n"
            "gamma_mid = 0.6\n"
            "def run_model():\n"
            "    try:\n"
            "        sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a modulation amplitude above one.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.48\n"
            "sigma_phi_sq = 0.30\n"
            "peclet = 2.5\n"
            "r_inner = 0.06\n"
            "r_outer = 0.55\n"
            "gamma_mid = 1.5\n"
            "def run_model():\n"
            "    try:\n"
            "        sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a non-positive sensor accuracy.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.48\n"
            "sigma_phi_sq = 0.0\n"
            "peclet = 2.5\n"
            "r_inner = 0.06\n"
            "r_outer = 0.55\n"
            "gamma_mid = 0.6\n"
            "def run_model():\n"
            "    try:\n"
            "        sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_sensing_entropy_production_rate(kappa_tilde, "
            "sigma_phi_sq, peclet, r_inner, r_outer, gamma_mid)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Regression: resolve the profile integral near a small inner radius.
    cases.append({
        "setup": "import numpy as np",
        "call": "sensing_entropy_production_rate(r_inner=0.001)",
        "gold_call": "_oracle_sensing_entropy_production_rate(r_inner=0.001)",
    })

    # Regression: an annulus modulation just above zero, where the annulus
    # carries almost none of the sensing weight.
    cases.append({
        "setup": "import numpy as np",
        "call": "sensing_entropy_production_rate(gamma_mid=1e-8)",
        "gold_call": "_oracle_sensing_entropy_production_rate(gamma_mid=1e-8)",
    })

    return cases
