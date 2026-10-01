"""
Local wave number of a two-dimensional plasma wave under an ideal metal plane, from its nonlocal screened dispersion.

A two-dimensional electron layer with equilibrium density N and effective mass m* supports plasma waves. For a wave of in-plane wave number q the dispersion is

  omega^2 = 2 pi N e^2 q / (m* eps(q))

where eps(q) is the effective permittivity seen by the charge oscillation. When the layer lies on a semiconductor of permittivity eps_s and is covered by a spacer of permittivity eps with an ideal metal plane at distance h, the image charges in the plane give

  eps(q) = [eps_s + eps coth(q h)] / 2.

Far from the plane (q h >> 1) this reduces to the unscreened average (eps_s + eps)/2 and the square-root law omega proportional to sqrt(q); close to the plane (q h << 1) it grows like eps/(2 q h) and the wave becomes the linear screened plasmon. Neither limit is assumed here: the full expression is used, so the wave number belonging to a given frequency has to be found numerically. The right-hand side of the dispersion increases monotonically with q, so every positive frequency has exactly one positive wave number.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s. The frequency is given in GHz (omega = 2 pi times the frequency) and the wave number is returned in cm^-1.

Returns
-------
float: the local plasma wave number q in cm^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def local_plasmon_wavenumber(frequency_ghz: float, density: float, gate_nm: float, eps_substrate: float,
                             eps_spacer: float, mass_ratio: float) -> float:
    '''Wave number of the screened two-dimensional plasma wave at a given frequency.

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency omega/(2 pi) in GHz.
    density : float
        Equilibrium electron density N in cm^-2.
    gate_nm : float
        Distance h between the electron layer and the ideal metal plane, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the layer.
    eps_spacer : float
        Permittivity eps of the spacer between the layer and the metal plane.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    wavenumber : float
        The unique positive q, in cm^-1, satisfying the screened dispersion.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number.
    '''
    return wavenumber

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _sm67_constants():
    return 4.80320471e-10, 9.1093837015e-28, 1.054571817e-27


def _sm67_require_positive(values):
    import numpy as np
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")


def _sm67_screened_eps(q, h, eps_sub, eps_spacer):
    import numpy as np
    return 0.5 * (eps_sub + eps_spacer / np.tanh(q * h))


def _sm67_q_solve(omega, density, h, eps_sub, eps_spacer, m_rel):
    import numpy as np
    e, m0, _ = _sm67_constants()
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    c = 2.0 * np.pi * density * e ** 2 / (m_rel * m0)
    target = 2.0 * np.log(omega) - np.log(c)
    x = np.log(0.5 * (eps_sub + eps_spacer)) + target
    for _ in range(200):
        q = np.exp(x)
        qh = q * h
        ep = _sm67_screened_eps(q, h, eps_sub, eps_spacer)
        with np.errstate(over="ignore"):
            dep = np.where(qh > 300.0, 0.0, -0.5 * eps_spacer * h / np.sinh(np.minimum(qh, 300.0)) ** 2)
        step = (x - np.log(ep) - target) / (1.0 - q * dep / ep)
        x = x - step
        if np.all(np.abs(step) < 1e-14):
            break
    q = np.exp(x)
    resid = np.log(c * q / _sm67_screened_eps(q, h, eps_sub, eps_spacer)) - 2.0 * np.log(omega)
    if not np.all(np.abs(resid) < 1e-10):
        raise ValueError("the local plasma wave number did not converge")
    return q


def _oracle_local_plasmon_wavenumber(frequency_ghz: float, density: float, gate_nm: float, eps_substrate: float,
                                     eps_spacer: float, mass_ratio: float) -> float:
    import numpy as np
    _sm67_require_positive(dict(frequency_ghz=frequency_ghz, density=density, gate_nm=gate_nm,
                                eps_substrate=eps_substrate, eps_spacer=eps_spacer, mass_ratio=mass_ratio))
    omega = 2.0 * np.pi * frequency_ghz * 1e9
    return float(_sm67_q_solve(omega, density, gate_nm * 1e-7, eps_substrate, eps_spacer, mass_ratio)[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: GaAs layer under a close gate at 500 GHz, q h of order 0.25 ---
        {
            "setup": """import numpy as np
""",
            "call": "local_plasmon_wavenumber(500.0, 8.0e11, 150.0, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_local_plasmon_wavenumber(500.0, 8.0e11, 150.0, 12.8, 12.8, 0.067)",
            "tol": 1e-08,
        },
        # --- Normal: the same frequency under a distant gate, q h of order 0.5 ---
        {
            "setup": """import numpy as np
""",
            "call": "local_plasmon_wavenumber(500.0, 1.2e12, 600.0, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_local_plasmon_wavenumber(500.0, 1.2e12, 600.0, 12.8, 12.8, 0.067)",
            "tol": 1e-08,
        },
        # --- Boundary: very low frequency, deep in the screened regime ---
        {
            "setup": """import numpy as np
""",
            "call": "local_plasmon_wavenumber(0.5, 2.0e12, 50.0, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_local_plasmon_wavenumber(0.5, 2.0e12, 50.0, 12.8, 12.8, 0.067)",
            "tol": 1e-08,
        },
        # --- Edge: dilute layer and a far plane with unequal permittivities, deep in the unscreened regime ---
        {
            "setup": """import numpy as np
""",
            "call": "local_plasmon_wavenumber(2000.0, 1.0e10, 50000.0, 12.9, 3.9, 0.067)",
            "gold_call": "_oracle_local_plasmon_wavenumber(2000.0, 1.0e10, 50000.0, 12.9, 3.9, 0.067)",
            "tol": 1e-08,
        },
        # --- Edge: GaN-like heavy electrons at high density under a very close gate ---
        {
            "setup": """import numpy as np
""",
            "call": "local_plasmon_wavenumber(300.0, 6.0e12, 25.0, 9.5, 9.5, 0.22)",
            "gold_call": "_oracle_local_plasmon_wavenumber(300.0, 6.0e12, 25.0, 9.5, 9.5, 0.22)",
            "tol": 1e-08,
        },
        # --- Invalid: negative gate distance ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        local_plasmon_wavenumber(500.0, 8.0e11, -150.0, 12.8, 12.8, 0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_local_plasmon_wavenumber(500.0, 8.0e11, -150.0, 12.8, 12.8, 0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
