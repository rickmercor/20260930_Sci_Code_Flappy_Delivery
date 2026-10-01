"""
Evaluate the thermal phonon response on its analytic frequency strip.

At positive temperature the causal acoustic response admits a local analytic continuation, allowing derivatives of the paper’s joint rate operator to be defined consistently.

Returns
-------
result : np.ndarray     Complex array with frequencies.shape, in ps^-1. On the real axis,     F(w) is the causal Abel limit of integral_0^infinity exp(i*w*t)C(t)dt,     where C is the equilibrium harmonic-bath correlation for     J(v)=alpha*v^3*exp(-(v/cutoff)^2), v>=0. For complex w, return the     analytic continuation of this real-axis response across the stated     strip. Both dispersive and dissipative contributions are included.     Required error is at most 3e-9*max(1 ps^-1, abs(F(w))).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bath_response(frequencies: "np.ndarray", alpha: float, cutoff: float, temperature: float) -> "np.ndarray":
    """Evaluate the causal thermal phonon response and its analytic continuation.

    Parameters
    ----------
    frequencies : np.ndarray
        Complex array of any shape, angular frequencies w in ps^-1,
        with abs(real(w))<=6*cutoff and abs(imag(w))<=pi*theta,
        where theta=(k_B/hbar)*temperature.
    alpha : float
        Nonnegative spectral-density strength in ps^2.
    cutoff : float
        Cutoff in ps^-1, 0.8<=cutoff<=3.
    temperature : float
        Temperature in K, 1<=temperature<=10, using
        k_B/hbar=0.1309203391 ps^-1 K^-1.

    Returns
    -------
    result : np.ndarray
        Complex array with frequencies.shape, in ps^-1. On the real axis,
        F(w) is the causal Abel limit of integral_0^infinity exp(i*w*t)C(t)dt,
        where C is the equilibrium harmonic-bath correlation for
        J(v)=alpha*v^3*exp(-(v/cutoff)^2), v>=0. For complex w, return the
        analytic continuation of this real-axis response across the stated
        strip. Both dispersive and dissipative contributions are included.
        Required error is at most 3e-9*max(1 ps^-1, abs(F(w))).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from functools import lru_cache
from scipy.special import roots_legendre

@lru_cache(maxsize=None)
def _gauss_nodes(count):
    return roots_legendre(count)

def _oracle_bath_response(frequencies: "np.ndarray", alpha: float, cutoff: float, temperature: float) -> "np.ndarray":
    z = np.asarray(frequencies, complex)
    flat = z.ravel()
    theta = .1309203391 * temperature
    bound = 12 * cutoff
    x, weights = _gauss_nodes(240)
    # Split at zero to resolve the thermal frequency scale near the endpoint.
    v = np.concatenate(((x + 1) * bound / 2, -(x + 1) * bound / 2))
    weights = np.tile(weights * bound / 2, 2)

    def _thermal_weight(w):
        w = np.asarray(w, complex)
        out = np.zeros_like(w)
        nz = np.abs(w) > 1e-14
        out[nz] = alpha * w[nz]**3 * np.exp(-(w[nz]/cutoff)**2) / (-np.expm1(-w[nz]/theta))
        return out

    hz = _thermal_weight(flat)
    denom = flat[:, None] - v[None, :]
    diff = _thermal_weight(v)[None, :] - hz[:, None]
    integrand = np.divide(diff, denom, out=np.zeros_like(diff), where=np.abs(denom) > 1e-7)
    close = np.abs(denom) <= 1e-7
    if np.any(close):
        mid = (flat[:, None] + v[None, :]) / 2
        w = mid[close]
        derivative = _thermal_weight(w) * (3/w - 2*w/cutoff**2 - np.exp(-w/theta)/(theta*(-np.expm1(-w/theta))))
        integrand[close] = -derivative
    value = np.pi*hz + 1j*(integrand @ weights + hz*np.log((bound+flat)/(bound-flat)))
    return value.reshape(z.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized scientific cases."""
    return [{'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 1\n'
               'frequencies = np.array([(-0.22+0j), 0j, (0.22+0j)], dtype=complex)\n'
               'alpha = 0.027\n'
               'cutoff = 2.2\n'
               'temperature = 4.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 2\n'
               'frequencies = np.array([0j], dtype=complex)\n'
               'alpha = 0.027\n'
               'cutoff = 2.2\n'
               'temperature = 1.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 3\n'
               'frequencies = np.array([(0.3+0.12j), (0.3-0.12j)], dtype=complex)\n'
               'alpha = 0.02\n'
               'cutoff = 1.8\n'
               'temperature = 2.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 4\n'
               'frequencies = np.array([(1e-05+0j), (-1e-05+0j), 1e-05j], dtype=complex)\n'
               'alpha = 0.015\n'
               'cutoff = 1.8\n'
               'temperature = 6.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 5\n'
               'frequencies = np.array([(-0.3+0.1j), (0.8-0.2j)], dtype=complex)\n'
               'alpha = 0.0\n'
               'cutoff = 2.2\n'
               'temperature = 4.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 6\n'
               'frequencies = np.array([(2.2+0j), (-2.2+0j), 0.25j], dtype=complex)\n'
               'alpha = 0.027\n'
               'cutoff = 2.2\n'
               'temperature = 4.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 7\n'
               'frequencies = np.array([(5-0.2j), (-5+0.2j)], dtype=complex)\n'
               'alpha = 0.018\n'
               'cutoff = 2.0\n'
               'temperature = 3.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 8\n'
               'frequencies = np.array([(-0.4-0.15j), (0.4+0.15j)], dtype=complex)\n'
               'alpha = 0.02\n'
               'cutoff = 1.2\n'
               'temperature = 1.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 9\n'
               'frequencies = np.array([[(-0.1+0j), (0.2+0.4j)], [(1.1+0j), (-0.2-0.4j)]], dtype=complex)\n'
               'alpha = 0.025\n'
               'cutoff = 2.6\n'
               'temperature = 8.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09},
     {'setup': 'import numpy as np\n'
               '## Thermal analytic response regime 10\n'
               'frequencies = np.array([(-0.03+0j), 0j, (0.07-0.1j)], dtype=complex)\n'
               'alpha = 0.01\n'
               'cutoff = 0.8\n'
               'temperature = 2.0\n',
      'call': 'bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'gold_call': '_oracle_bath_response(frequencies.copy(), alpha, cutoff, temperature)',
      'tol': 3e-09}]
