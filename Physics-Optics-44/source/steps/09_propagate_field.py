"""
March the complex field through the medium and return it after the last step. Use the symmetric split-step scheme in which each step applies half of the diffraction operator in the frequency domain, then the whole nonlinear and confinement phase in the coordinate domain, then the remaining half of diffraction. Element j of controls is the value the named knob takes on step j, with the other two parameters held at their supplied values. Raise ValueError if dz is not strictly positive, if controls is not a non-empty one-dimensional array, if u0 does not match the grid length, or if the knob name is not one of the three.

Splitting the evolution this way is exact for each operator separately and second-order accurate in the step for their combination, because the two do not commute. The nonlinear phase contains both the local term and the nonlocal average, the latter evaluated as a multiplication in the frequency domain, so a step costs a small fixed number of transforms regardless of how long the response is.

Returns
-------
ndarray of shape (npts,), complex128: the field after the final step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_field(u0: 'np.ndarray', npts: int, half_width: float, controls: 'np.ndarray', dz: float, gamma: float, alpha: float, sigma: float, knob: str) -> 'np.ndarray':
    """March the complex field through the medium and return it after the last step. Use the symmetric split-step scheme in which each step applies half of the diffraction operator in the frequency domain, then the whole nonlinear and confinement phase in the coordinate domain, then the remaining half of diffraction. Element j of controls is the value the named knob takes on step j, with the other two parameters held at their supplied values. Raise ValueError if dz is not strictly positive, if controls is not a non-empty one-dimensional array, if u0 does not match the grid length, or if the knob name is not one of the three.

    Returns
    -------
    ndarray of shape (npts,), complex128: the field after the final step.

    Raises
    ------
    ValueError
        If dz is not strictly positive, controls is not a non-empty one-dimensional array, u0 does not match the grid length, or knob is not 'sigma', 'gamma' or 'alpha2'.
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

def _oracle_propagate_field(u0: "np.ndarray", npts: int, half_width: float, controls: "np.ndarray",
                      dz: float, gamma: float, alpha: float, sigma: float, knob: str) -> "np.ndarray":
    """Symmetric split-step march; controls[j] is the knob value on step j."""
    controls = np.asarray(controls, dtype=float)
    dz = float(dz)
    if dz <= 0.0 or controls.ndim != 1 or controls.size < 1:
        raise ValueError("dz must be positive and controls a non-empty 1-D array")
    g = _grid(npts, half_width)
    x, k = g[0], g[1]
    u = np.asarray(u0, dtype=complex).copy()
    if u.shape != x.shape:
        raise ValueError("u0 must have the same length as the grid")
    half = np.exp(-0.25j * dz * k ** 2)
    for c in controls:
        s, gm, al = sigma, gamma, alpha
        if knob == "sigma":
            s = c
        elif knob == "gamma":
            gm = c
        elif knob == "alpha2":
            al = np.sqrt(c) if c >= 0.0 else 0.0
        else:
            raise ValueError("knob must be 'sigma', 'gamma' or 'alpha2'")
        rhat = _oracle_response_spectrum(k, s)
        u = np.fft.ifft(half * np.fft.fft(u))
        a2 = np.abs(u) ** 2
        phase = np.real(np.fft.ifft(rhat * np.fft.fft(a2))) + gm * a2 - (al ** 2 if knob != "alpha2" else c) * x ** 2
        u = u * np.exp(1j * dz * phase)
        u = np.fft.ifft(half * np.fft.fft(u))
    return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np\ng = _grid(256, 20.0)\nu0 = (np.exp(-g[0]**2/2.0)).astype(complex)\nc = np.full(50, 4.0)",
                    "call": "propagate_field(u0.copy(), 256, 20.0, c.copy(), 0.01, 0.2, 0.15, 4.0, 'sigma')",
                    "gold_call": "_oracle_propagate_field(u0, 256, 20.0, c, 0.01, 0.2, 0.15, 4.0, 'sigma')"
            },
            {
                    "setup": "import numpy as np\ng = _grid(256, 20.0)\nu0 = (np.exp(-g[0]**2/2.0)).astype(complex)\nc = np.linspace(0.2, 0.4, 40)",
                    "call": "propagate_field(u0.copy(), 256, 20.0, c.copy(), 0.005, 0.2, 0.15, 4.0, 'gamma')",
                    "gold_call": "_oracle_propagate_field(u0, 256, 20.0, c, 0.005, 0.2, 0.15, 4.0, 'gamma')"
            },
            {
                    "setup": "import numpy as np\ng = _grid(256, 20.0)\nu0 = (np.exp(-g[0]**2/2.0)).astype(complex)\nc = np.array([4.0])",
                    "call": "propagate_field(u0.copy(), 256, 20.0, c.copy(), 1e-6, 0.2, 0.15, 4.0, 'sigma')",
                    "gold_call": "_oracle_propagate_field(u0, 256, 20.0, c, 1e-6, 0.2, 0.15, 4.0, 'sigma')"
            },
            {
                    "setup": "import numpy as np\ng = _grid(256, 20.0)\nu0 = (np.exp(-g[0]**2/2.0)).astype(complex)\nc = np.linspace(0.0225, 0.09, 40)",
                    "call": "propagate_field(u0.copy(), 256, 20.0, c.copy(), 0.005, 0.2, 0.15, 4.0, 'alpha2')",
                    "gold_call": "_oracle_propagate_field(u0, 256, 20.0, c, 0.005, 0.2, 0.15, 4.0, 'alpha2')"
            },
            {
                    "setup": "import numpy as np\ndef power_drift(fn):\n    g = _grid(256, 20.0)\n    x = g[0]; dx = x[1]-x[0]\n    u0 = (np.exp(-x**2/2.0)).astype(complex)\n    p0 = float(np.sum(np.abs(u0)**2)*dx)\n    u = fn(u0, 256, 20.0, np.full(200, 3.0), 0.01, 0.2, 0.15, 3.0, 'sigma')\n    return float(abs((np.sum(np.abs(u)**2)*dx - p0)/p0) < 1e-10)",
                    "call": "power_drift(propagate_field)",
                    "gold_call": "power_drift(_oracle_propagate_field)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        propagate_field(np.zeros(256, dtype=complex), 256, 20.0, np.full(5, 4.0), 0.0, 0.2, 0.15, 4.0, 'sigma')\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_propagate_field(np.zeros(256, dtype=complex), 256, 20.0, np.full(5, 4.0), 0.0, 0.2, 0.15, 4.0, 'sigma')\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
