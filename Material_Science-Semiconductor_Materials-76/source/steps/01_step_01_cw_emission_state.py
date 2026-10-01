"""
Step 01 - Continuous-wave state of the ring laser equations.

Single-frequency (continuous-wave) emission of a unidirectional ring quantum cascade laser.

The laser is described by effective semiconductor Maxwell-Bloch equations for the
slowly varying field envelope F(eta, t), the polarisation envelope P(eta, t) and
the carrier density D(eta, t), all dimensionless. Time is measured in units of the
polarisation dephasing time tau_d and the propagation coordinate eta in units of
v_g * tau_d, so that the group velocity is 1:

    dF/deta + dF/dt = -sigma (F + P)
    dP/dt           = -Gamma (1 + i alpha) [ P + (1 + i alpha) D F ]
    dD/dt           = b [ mu - D + (F* P + F P*) / 2 ]

sigma is the field loss rate, b the carrier recovery rate, Gamma the scaled gain
bandwidth, alpha the linewidth enhancement factor and mu the pump. The reference
frequency (zero detuning) is the maximum of the unsaturated gain.

A continuous-wave solution has uniform intensity and a single wavenumber k:
F = F0 exp(-i k eta + i omega t), P = P0 exp(-i k eta + i omega t), D = D0
constant, with F0 chosen real and positive. For a given k the stationary
equations fix omega, D0, the intensity X = F0^2 and the complex P0. Where the
stationary equations admit several frequencies, the physical one is the root
continuously connected to omega = k as sigma -> 0, i.e. the root closest to k;
for the parameters used here it is unique within the gain bandwidth. A solution
exists only above that wavenumber's lasing threshold, where X > 0.

Values are returned in the scaled units above (omega in units of 1/tau_d).

Returns
-------
numpy.ndarray of shape (5,): [omega, D0, X, Re(P0), Im(P0)] of the continuous wave, scaled units (omega in 1/tau_d)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cw_emission_state(k: float, Gamma: float, alpha: float, sigma: float, mu: float) -> "np.ndarray":
    '''Continuous-wave solution of the ring equations at scaled wavenumber k.

    Parameters
    ----------
    k : float
        Scaled wavenumber of the continuous wave (k = 0 is the reference
        frequency, the gain maximum).
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    sigma : float
        Scaled field loss rate, > 0.
    mu : float
        Pump parameter, > 0.

    Returns
    -------
    state : np.ndarray
        Shape (5,): [omega, D0, X, Re(P0), Im(P0)] with F0 = sqrt(X) real
        and positive; omega and the envelopes in the scaled units of the
        equations.

    Raises
    ------
    ValueError
        If any argument is not a finite real number, if Gamma, sigma or mu is
        not positive, if the wavenumber lies outside the gain band (no
        positive-gain root of the dispersion relation), or if mu does not
        exceed the lasing threshold of this continuous wave (X <= 0).
    '''
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cw_check_float(name, value, positive):
    """Return value as a finite float, optionally requiring it to be positive."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s must be finite%s" % (name, " and positive" if positive else ""))
    return value


def _cw_gain_dispersion(omega, Gamma, alpha):
    """Real and imaginary parts of the scaled medium response at detuning omega."""
    shift = omega + alpha * Gamma
    den = Gamma ** 2 + shift ** 2
    h1 = (Gamma ** 2 * (1.0 - alpha ** 2) + 2.0 * alpha * Gamma * shift) / den
    h2 = (-2.0 * alpha * Gamma ** 2 + Gamma * (1.0 - alpha ** 2) * shift) / den
    return h1, h2


def _cw_frequency(k, Gamma, alpha, sigma):
    """Root of the continuous-wave dispersion relation closest to omega = k."""
    omega = k
    for _ in range(200):
        h1, h2 = _cw_gain_dispersion(omega, Gamma, alpha)
        if h1 <= 0.0:
            raise ValueError("the wavenumber lies outside the gain band, no lasing solution")
        new = k - sigma * h2 / h1
        if abs(new - omega) <= 1e-16 * max(1.0, abs(new)):
            omega = new
            break
        omega = new
    else:
        raise ValueError("the dispersion relation did not converge")
    # polish with Newton steps on f(w) = w - k + sigma*h2/h1
    for _ in range(5):
        h1, h2 = _cw_gain_dispersion(omega, Gamma, alpha)
        d = 1e-7 * max(1.0, abs(omega))
        h1p, h2p = _cw_gain_dispersion(omega + d, Gamma, alpha)
        h1m, h2m = _cw_gain_dispersion(omega - d, Gamma, alpha)
        f = omega - k + sigma * h2 / h1
        fp = 1.0 + sigma * (h2p / h1p - h2m / h1m) / (2.0 * d)
        step = f / fp
        omega -= step
        if abs(step) <= 1e-17 * max(1.0, abs(omega)):
            break
    return omega


def _oracle_cw_emission_state(k: float, Gamma: float, alpha: float, sigma: float, mu: float) -> "np.ndarray":
    k = _cw_check_float("k", k, False)
    Gamma = _cw_check_float("Gamma", Gamma, True)
    alpha = _cw_check_float("alpha", alpha, False)
    sigma = _cw_check_float("sigma", sigma, True)
    mu = _cw_check_float("mu", mu, True)
    omega = _cw_frequency(k, Gamma, alpha, sigma)
    h1, h2 = _cw_gain_dispersion(omega, Gamma, alpha)
    d0 = 1.0 / h1
    x = mu - 1.0 / h1
    if not x > 0.0:
        raise ValueError("mu does not exceed the lasing threshold of this continuous wave")
    f0 = np.sqrt(x)
    p0 = (-h1 + 1j * h2) * d0 * f0
    return np.array([omega, d0, x, p0.real, p0.imag], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: wave at the gain maximum, terahertz ring parameters ---
        {
            "setup": "import numpy as np\n",
            "call": "cw_emission_state(0.0, 0.06, 0.95, 1.6e-3, 9.2)",
            "gold_call": "_oracle_cw_emission_state(0.0, 0.06, 0.95, 1.6e-3, 9.2)",
            "tol": 1e-9,
        },
        # --- Boundary: detuned wave, strong loss, just above its own threshold ---
        {
            "setup": "import numpy as np\n",
            "call": "cw_emission_state(0.021, 0.05, 1.3, 0.012, 1.35)",
            "gold_call": "_oracle_cw_emission_state(0.021, 0.05, 1.3, 0.012, 1.35)",
            "tol": 1e-9,
        },
        # --- Edge: negative wavenumber and negative alpha ---
        {
            "setup": "import numpy as np\n",
            "call": "cw_emission_state(-0.015, 0.08, -0.7, 0.006, 3.0)",
            "gold_call": "_oracle_cw_emission_state(-0.015, 0.08, -0.7, 0.006, 3.0)",
            "tol": 1e-9,
        },
        # --- Edge: two-level limit alpha = 0 at the gain maximum ---
        {
            "setup": "import numpy as np\n",
            "call": "cw_emission_state(0.0, 0.06, 0.0, 1.6e-3, 4.0)",
            "gold_call": "_oracle_cw_emission_state(0.0, 0.06, 0.0, 1.6e-3, 4.0)",
            "tol": 1e-9,
        },
        # --- Invalid: pump below the lasing threshold of the detuned wave ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        cw_emission_state(0.03, 0.06, 0.95, 1.6e-3, 1.05)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_cw_emission_state(0.03, 0.06, 0.95, 1.6e-3, 1.05)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # a strongly detuned wave, where the pulled frequency departs from k
            "setup": 'import numpy as np',
            "call": "cw_emission_state(0.03, 0.06, 0.95, 1.6e-3, 9.2)",
            "gold_call": "_oracle_cw_emission_state(0.03, 0.06, 0.95, 1.6e-3, 9.2)",
            "tol": 1e-9,
        },
        {   # detuning of the other sign on a lossier ring
            "setup": 'import numpy as np',
            "call": "cw_emission_state(-0.025, 0.06, 0.9, 2.5e-3, 7.0)",
            "gold_call": "_oracle_cw_emission_state(-0.025, 0.06, 0.9, 2.5e-3, 7.0)",
            "tol": 1e-9,
        },
        {   # a shorter, lossier wave at alpha = 1.3
            "setup": 'import numpy as np',
            "call": "cw_emission_state(0.012, 0.05, 1.3, 6.0e-3, 3.2)",
            "gold_call": "_oracle_cw_emission_state(0.012, 0.05, 1.3, 6.0e-3, 3.2)",
            "tol": 1e-9,
        },
        {   # the other sign of detuning at alpha = 0.4
            "setup": 'import numpy as np',
            "call": "cw_emission_state(-0.04, 0.09, 0.4, 5.0e-3, 4.5)",
            "gold_call": "_oracle_cw_emission_state(-0.04, 0.09, 0.4, 5.0e-3, 4.5)",
            "tol": 1e-9,
        },
        # --- Invalid: arguments outside their documented ranges ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        cw_emission_state(0.0, 0.06, 0.95, 0.0, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    try:\n"
                     "        cw_emission_state(0.0, -1.0, 0.95, 1.6e-3, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 16\n"
                     "    try:\n"
                     "        cw_emission_state(0.0, 0.06, 0.95, 1.6e-3, float('inf'))\n"
                     "    except ValueError:\n"
                     "        mask += 4\n"
                     "    except Exception:\n"
                     "        mask += 32\n"
                     "    return mask\n"
                     "def run_gold():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        _oracle_cw_emission_state(0.0, 0.06, 0.95, 0.0, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    try:\n"
                     "        _oracle_cw_emission_state(0.0, -1.0, 0.95, 1.6e-3, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 16\n"
                     "    try:\n"
                     "        _oracle_cw_emission_state(0.0, 0.06, 0.95, 1.6e-3, float('inf'))\n"
                     "    except ValueError:\n"
                     "        mask += 4\n"
                     "    except Exception:\n"
                     "        mask += 32\n"
                     "    return mask\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
]
