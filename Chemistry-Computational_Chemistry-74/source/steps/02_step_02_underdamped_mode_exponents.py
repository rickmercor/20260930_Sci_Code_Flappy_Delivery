"""
Exponential decomposition of the correlation function of an underdamped intramolecular vibration treated as a Brownian oscillator: the oscillating pair of terms and the first Matsubara terms, with complex decay rates.

A high-frequency intramolecular vibration, damped by its own surroundings, is described as a separate harmonic bath with the Brownian-oscillator spectral density J(omega) = 2 (lambda / hbar) Gamma omega_0^2 omega / ((omega_0^2 - omega^2)^2 + Gamma^2 omega^2). Here omega_0 is the mode frequency, Gamma = 1 / tau_damp its damping rate and lambda its reorganization energy, again with lambda / hbar = (1 / pi) times the integral of J(omega) / omega over positive frequencies. The mode is underdamped, omega_0 > Gamma / 2.

Its correlation function C(t) = <Q(t) Q(0)> / hbar^2, defined by the same integral over J(omega) as in the previous step, is again a sum of exponentials: a pair of oscillating terms with complex rates and a series of real Matsubara terms. Work out the rate and the complex amplitude of every term.

Energies are given in cm^-1 and converted to angular frequencies in rad/ps by the factor 2 pi c with c = 2.99792458e-2 cm/ps; k_B = 0.6950348 cm^-1/K.

Returns
-------
numpy.ndarray of shape (n_matsubara + 2, 4): rows (Re c_k, Im c_k, Re gamma_k, Im gamma_k) of the oscillating pair and the Matsubara terms of the mode correlation function, in ps^-2 and ps^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def underdamped_mode_exponents(reorg_cm: float, mode_cm: float, tau_damp_fs: float, temperature_k: float, n_matsubara: int) -> "np.ndarray":
    '''Brownian-oscillator (underdamped mode) correlation function as a finite sum of exponentials.

    Parameters
    ----------
    reorg_cm : float
        Reorganization energy lambda of the mode bath in cm^-1, non-negative.
    mode_cm : float
        Mode frequency hbar omega_0 in cm^-1, positive.
    tau_damp_fs : float
        Damping time in fs; the damping rate is Gamma = 1 / tau_damp.
    temperature_k : float
        Temperature in K.
    n_matsubara : int
        Number of Matsubara terms kept after the oscillating pair.

    Returns
    -------
    exponents : numpy.ndarray
        Real array of shape (n_matsubara + 2, 4). Row k holds (Re c_k, Im c_k, Re gamma_k,
        Im gamma_k) of the term c_k exp(-gamma_k t) of C(t) = <Q(t)Q(0)>/hbar^2,
        with c_k in ps^-2 and gamma_k in ps^-1.
        Rows 0 and 1 are the oscillating pair, row 0 the term whose rate has a
        negative imaginary part and row 1 the term whose rate has a positive
        imaginary part; rows 2..n_matsubara + 1 are the Matsubara terms in
        increasing order of frequency.

    Raises
    ------
    ValueError
        If reorg_cm is negative or not finite, mode_cm, tau_damp_fs or
        temperature_k is not positive and finite, n_matsubara is not a
        non-negative integer, or the mode is not underdamped (omega_0 <= Gamma / 2).
    '''
    return exponents

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _angular_per_wavenumber() -> float:
    """Angular frequency in rad/ps carried by one cm^-1 (2 pi c with c in cm/ps)."""
    import numpy as np
    return 2.0 * np.pi * 2.99792458e-2


def _oracle_underdamped_mode_exponents(reorg_cm: float, mode_cm: float, tau_damp_fs: float,
                                       temperature_k: float, n_matsubara: int) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(reorg_cm) or reorg_cm < 0.0:
        raise ValueError("reorganization energy must be finite and non-negative")
    if not np.isfinite(mode_cm) or mode_cm <= 0.0:
        raise ValueError("mode frequency must be positive")
    if not np.isfinite(tau_damp_fs) or tau_damp_fs <= 0.0:
        raise ValueError("damping time must be positive")
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if int(n_matsubara) != n_matsubara or n_matsubara < 0:
        raise ValueError("number of Matsubara terms must be a non-negative integer")
    w2c = _angular_per_wavenumber()
    lam = reorg_cm * w2c
    w0 = mode_cm * w2c
    gam = 1000.0 / tau_damp_fs
    if w0 <= 0.5 * gam:
        raise ValueError("the mode must be underdamped: omega_0 > gamma / 2")
    beta = 1.0 / (0.6950348 * temperature_k * w2c)
    om = np.sqrt(w0 * w0 - 0.25 * gam * gam)
    out = np.zeros((int(n_matsubara) + 2, 4))
    for row, s in ((0, -1.0), (1, 1.0)):
        z = s * om - 0.5j * gam
        amp = lam * w0 * w0 / (2.0 * s * om) * (1.0 / np.tanh(0.5 * beta * z) + 1.0)
        rate = 1j * z
        out[row] = [amp.real, amp.imag, rate.real, rate.imag]
    for k in range(1, int(n_matsubara) + 1):
        nu = 2.0 * np.pi * k / beta
        amp = -4.0 * lam * gam * w0 * w0 / beta * nu / ((w0 * w0 + nu * nu) ** 2 - gam * gam * nu * nu)
        out[k + 1] = [amp, 0.0, nu, 0.0]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: a 600 cm^-1 mode at room temperature with one Matsubara term ---
        {
            "setup": """import numpy as np
""",
            "call": "underdamped_mode_exponents(3.0, 600.0, 80.0, 300.0, 1)",
            "gold_call": "_oracle_underdamped_mode_exponents(3.0, 600.0, 80.0, 300.0, 1)",
            "tol": 1e-08,
        },
        # --- Normal: a softer, more strongly damped mode at 150 K with two Matsubara terms ---
        {
            "setup": """import numpy as np
""",
            "call": "underdamped_mode_exponents(10.0, 250.0, 40.0, 150.0, 2)",
            "gold_call": "_oracle_underdamped_mode_exponents(10.0, 250.0, 40.0, 150.0, 2)",
            "tol": 1e-08,
        },
        # --- Boundary: no Matsubara term, the oscillating pair alone ---
        {
            "setup": """import numpy as np
""",
            "call": "underdamped_mode_exponents(6.0, 900.0, 200.0, 400.0, 0)",
            "gold_call": "_oracle_underdamped_mode_exponents(6.0, 900.0, 200.0, 400.0, 0)",
            "tol": 1e-08,
        },
        # --- Edge: zero reorganization energy, all amplitudes vanish but the rates remain ---
        {
            "setup": """import numpy as np
""",
            "call": "underdamped_mode_exponents(0.0, 500.0, 100.0, 300.0, 1)",
            "gold_call": "_oracle_underdamped_mode_exponents(0.0, 500.0, 100.0, 300.0, 1)",
            "tol": 1e-08,
        },
        # --- Edge: a mode close to critical damping, omega_0 about 1.13 Gamma / 2, so the oscillation frequency is about half of Gamma / 2 ---
        {
            "setup": """import numpy as np
""",
            "call": "underdamped_mode_exponents(5.0, 30.0, 100.0, 300.0, 1)",
            "gold_call": "_oracle_underdamped_mode_exponents(5.0, 30.0, 100.0, 300.0, 1)",
            "tol": 1e-08,
        },
        # --- Edge: a cold bath in which the amplitude on the rate with negative imaginary part is Boltzmann suppressed and the Matsubara terms are comparable to it ---
        {
            "setup": """import numpy as np
""",
            "call": "underdamped_mode_exponents(2.0, 87.0, 150.0, 20.0, 3)",
            "gold_call": "_oracle_underdamped_mode_exponents(2.0, 87.0, 150.0, 20.0, 3)",
            "tol": 1e-08,
        },
        # --- Invalid: an overdamped mode ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        underdamped_mode_exponents(4.0, 20.0, 50.0, 300.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_underdamped_mode_exponents(4.0, 20.0, 50.0, 300.0, 1)
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
