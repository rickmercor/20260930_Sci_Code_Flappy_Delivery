"""
Exponential decomposition of the correlation function of an overdamped Drude-Lorentz bath: the Drude term and the first Matsubara terms, as complex amplitudes and decay rates in picosecond units.

The low-frequency environment of the donor-acceptor pair is a harmonic bath that couples to the molecule through one collective coordinate Q = sum_j lambda_j (b_j + b_j^dagger), with reorganization energy lambda_0 = sum_j lambda_j^2 / (hbar nu_j). Its spectral density is of Drude-Lorentz form, J(omega) = 2 (lambda_0 / hbar) W omega / (omega^2 + W^2), with cutoff W equal to the inverse bath relaxation time, so that lambda_0 / hbar = (1 / pi) times the integral of J(omega) / omega over positive frequencies.

The hierarchical equations of motion need the equilibrium correlation function C(t) = <Q(t) Q(0)> / hbar^2 = (1 / pi) times the integral over positive omega of J(omega) [coth(beta hbar omega / 2) cos(omega t) - i sin(omega t)], for t >= 0, written as a sum of decaying exponentials. One term comes from the pole of the Drude-Lorentz form and decays at the rate W; an infinite series comes from the poles of the hyperbolic cotangent and decays at the bosonic Matsubara frequencies. Work out the complex amplitude of each term.

Energies are given in cm^-1 and converted to angular frequencies in rad/ps by the factor 2 pi c with c = 2.99792458e-2 cm/ps; k_B = 0.6950348 cm^-1/K.

Returns
-------
numpy.ndarray of shape (n_matsubara + 1, 4): rows (Re c_k, Im c_k, Re gamma_k, Im gamma_k) of the Drude and Matsubara terms of the bath correlation function, in ps^-2 and ps^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drude_bath_exponents(reorg_cm: float, tau_bath_fs: float, temperature_k: float, n_matsubara: int) -> "np.ndarray":
    '''Drude-Lorentz bath correlation function as a finite sum of exponentials.

    Parameters
    ----------
    reorg_cm : float
        Reorganization energy lambda_0 of the bath in cm^-1, non-negative.
    tau_bath_fs : float
        Bath relaxation time in fs; the Drude cutoff is W = 1 / tau_bath.
    temperature_k : float
        Temperature in K.
    n_matsubara : int
        Number of Matsubara terms kept after the Drude term.

    Returns
    -------
    exponents : numpy.ndarray
        Real array of shape (n_matsubara + 1, 4). Row k holds (Re c_k, Im c_k, Re gamma_k,
        Im gamma_k) of the term c_k exp(-gamma_k t) of C(t) = <Q(t)Q(0)>/hbar^2,
        with c_k in ps^-2 and gamma_k in ps^-1.
        Row 0 is the Drude term (gamma_0 = W); rows 1..n_matsubara are the
        Matsubara terms in increasing order of frequency.

    Raises
    ------
    ValueError
        If reorg_cm is negative or not finite, tau_bath_fs or temperature_k is not
        positive and finite, n_matsubara is not a non-negative integer, or a kept
        Matsubara frequency coincides with W to relative precision 1e-12.
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


def _oracle_drude_bath_exponents(reorg_cm: float, tau_bath_fs: float, temperature_k: float,
                                 n_matsubara: int) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(reorg_cm) or reorg_cm < 0.0:
        raise ValueError("reorganization energy must be finite and non-negative")
    if not np.isfinite(tau_bath_fs) or tau_bath_fs <= 0.0:
        raise ValueError("bath relaxation time must be positive")
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if int(n_matsubara) != n_matsubara or n_matsubara < 0:
        raise ValueError("number of Matsubara terms must be a non-negative integer")
    w2c = _angular_per_wavenumber()
    lam = reorg_cm * w2c
    gam = 1000.0 / tau_bath_fs
    beta = 1.0 / (0.6950348 * temperature_k * w2c)
    out = np.zeros((int(n_matsubara) + 1, 4))
    out[0] = [lam * gam / np.tan(0.5 * beta * gam), -lam * gam, gam, 0.0]
    for k in range(1, int(n_matsubara) + 1):
        nu = 2.0 * np.pi * k / beta
        if abs(nu - gam) <= 1e-12 * gam:
            raise ValueError("a Matsubara frequency coincides with the Drude cutoff")
        out[k] = [4.0 * lam * gam / beta * nu / (nu * nu - gam * gam), 0.0, nu, 0.0]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: a room-temperature bath, Drude term plus one Matsubara term ---
        {
            "setup": """import numpy as np
""",
            "call": "drude_bath_exponents(50.0, 100.0, 300.0, 1)",
            "gold_call": "_oracle_drude_bath_exponents(50.0, 100.0, 300.0, 1)",
            "tol": 1e-08,
        },
        # --- Normal: a colder, slower bath with three Matsubara terms ---
        {
            "setup": """import numpy as np
""",
            "call": "drude_bath_exponents(35.0, 60.0, 120.0, 3)",
            "gold_call": "_oracle_drude_bath_exponents(35.0, 60.0, 120.0, 3)",
            "tol": 1e-08,
        },
        # --- Boundary: no Matsubara term, the Drude term alone ---
        {
            "setup": """import numpy as np
""",
            "call": "drude_bath_exponents(80.0, 200.0, 250.0, 0)",
            "gold_call": "_oracle_drude_bath_exponents(80.0, 200.0, 250.0, 0)",
            "tol": 1e-08,
        },
        # --- Edge: zero reorganization energy, all amplitudes vanish but the rates remain ---
        {
            "setup": """import numpy as np
""",
            "call": "drude_bath_exponents(0.0, 100.0, 300.0, 2)",
            "gold_call": "_oracle_drude_bath_exponents(0.0, 100.0, 300.0, 2)",
            "tol": 1e-08,
        },
        # --- Edge: a cold, fast bath with beta hbar W / 2 above pi / 2, where the real part of the Drude amplitude turns negative ---
        {
            "setup": """import numpy as np
""",
            "call": "drude_bath_exponents(40.0, 25.0, 60.0, 2)",
            "gold_call": "_oracle_drude_bath_exponents(40.0, 25.0, 60.0, 2)",
            "tol": 1e-08,
        },
        # --- Edge: a cutoff above the first Matsubara frequency, which flips the sign of that term's amplitude ---
        {
            "setup": """import numpy as np
""",
            "call": "drude_bath_exponents(50.0, 10.0, 100.0, 2)",
            "gold_call": "_oracle_drude_bath_exponents(50.0, 10.0, 100.0, 2)",
            "tol": 1e-08,
        },
        # --- Invalid: zero temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        drude_bath_exponents(50.0, 100.0, 0.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_drude_bath_exponents(50.0, 100.0, 0.0, 1)
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
