"""
Return the real, non-negative profile the medium relaxes to under the prescribed number of imaginary-distance steps, on the grid the earlier steps use. Start from the unit-width Gaussian rescaled to carry the given power, and take each step as half of the diffraction operator, then the full nonlinear and confinement factor, then the remaining half, rescaling to the given power after every step. Return the pointwise magnitude of the field after the final rescaling, so a sample whose real part is negative contributes its magnitude and not zero. Raise ValueError if the step size is not strictly positive, if the step count is not a positive integer, or if the power is not strictly positive.

Replacing the propagation distance by an imaginary one turns the evolution into a relaxation: every mode decays at a rate set by its own energy, so the lowest one survives and the profile settles onto the medium's self-trapped state. Rescaling to fixed power after each step keeps the relaxation at the intended power instead of letting the field decay away. The result is the state the beam would sit in if it were left alone in the medium, and it is the reference an engineered protocol is judged against. It is not the Gaussian of the reduced description: that is an approximation to this profile, and the difference between them is what a single collective coordinate cannot represent.

Returns
-------
ndarray of shape (npts,), float64: the relaxed profile, real and non-negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stationary_soliton(npts: int, half_width: float, power: float, gamma: float, alpha: float, sigma: float, dtau: float, nsteps: int) -> 'np.ndarray':
    """Return the real, non-negative profile the medium relaxes to under the prescribed number of imaginary-distance steps, on the grid the earlier steps use. Start from the unit-width Gaussian rescaled to carry the given power, and take each step as half of the diffraction operator, then the full nonlinear and confinement factor, then the remaining half, rescaling to the given power after every step. Return the pointwise magnitude of the field after the final rescaling, so a sample whose real part is negative contributes its magnitude and not zero. Raise ValueError if the step size is not strictly positive, if the step count is not a positive integer, or if the power is not strictly positive.

    Returns
    -------
    ndarray of shape (npts,), float64: the relaxed profile, real and non-negative.

    Raises
    ------
    ValueError
        If dtau or power is not strictly positive, or nsteps is not a positive integer.
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

def _oracle_stationary_soliton(npts: int, half_width: float, power: float, gamma: float,
                               alpha: float, sigma: float, dtau: float, nsteps: int) -> "np.ndarray":
    """Imaginary-time relaxation at fixed power; returns the real, positive profile."""
    dtau = float(dtau); nsteps = int(nsteps)
    if dtau <= 0.0 or nsteps < 1:
        raise ValueError("dtau must be positive and nsteps a positive integer")
    if power <= 0.0:
        raise ValueError("power must be positive")
    g = _grid(npts, half_width)
    x, k = g[0], g[1]
    dx = x[1] - x[0]
    rhat = _oracle_response_spectrum(k, sigma)
    u = np.exp(-0.5 * x ** 2).astype(complex)
    u *= np.sqrt(power / (np.sum(np.abs(u) ** 2) * dx))
    half = np.exp(-0.25 * dtau * k ** 2)
    for _ in range(nsteps):
        u = np.fft.ifft(half * np.fft.fft(u))
        a2 = np.abs(u) ** 2
        phase = np.real(np.fft.ifft(rhat * np.fft.fft(a2))) + gamma * a2 - alpha ** 2 * x ** 2
        u = u * np.exp(dtau * phase)
        u = np.fft.ifft(half * np.fft.fft(u))
        u *= np.sqrt(power / (np.sum(np.abs(u) ** 2) * dx))
    return np.abs(u)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "stationary_soliton(256, 20.0, 12.0, 0.2, 0.15, 0.8, 0.01, 400)",
                    "gold_call": "_oracle_stationary_soliton(256, 20.0, 12.0, 0.2, 0.15, 0.8, 0.01, 400)",
                    "tol": 1e-07
            },
            {
                    "setup": "import numpy as np",
                    "call": "stationary_soliton(256, 20.0, 12.0, 0.2, 0.15, 4.0, 0.02, 300)",
                    "gold_call": "_oracle_stationary_soliton(256, 20.0, 12.0, 0.2, 0.15, 4.0, 0.02, 300)",
                    "tol": 1e-07
            },
            {
                    "setup": "import numpy as np",
                    "call": "stationary_soliton(128, 15.0, 20.0, 0.1, 0.1, 1.5, 0.005, 500)",
                    "gold_call": "_oracle_stationary_soliton(128, 15.0, 20.0, 0.1, 0.1, 1.5, 0.005, 500)",
                    "tol": 1e-07
            },
            {
                    "setup": "import numpy as np\ndef holds_power(fn):\n    g = _grid(256, 20.0)\n    dx = g[0][1] - g[0][0]\n    v = fn(256, 20.0, 12.0, 0.2, 0.15, 0.8, 0.01, 400)\n    return float(abs(float(np.sum(v ** 2) * dx) - 12.0) < 1e-9)",
                    "call": "holds_power(stationary_soliton)",
                    "gold_call": "holds_power(_oracle_stationary_soliton)"
            },
            {
                    "setup": "import numpy as np\ndef is_symmetric(fn):\n    v = fn(256, 20.0, 12.0, 0.2, 0.15, 0.8, 0.01, 400)\n    return float(float(np.max(np.abs(v[1:] - v[1:][::-1]))) < 1e-8 * float(np.max(v)))",
                    "call": "is_symmetric(stationary_soliton)",
                    "gold_call": "is_symmetric(_oracle_stationary_soliton)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        stationary_soliton(256, 20.0, 12.0, 0.2, 0.15, 0.8, 0.0, 400)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_stationary_soliton(256, 20.0, 12.0, 0.2, 0.15, 0.8, 0.0, 400)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
