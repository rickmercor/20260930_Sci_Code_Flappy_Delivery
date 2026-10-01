"""
Expand the correlation function of an overdamped (Drude-Lorentz) harmonic bath into decaying exponentials.

A pigment embedded in a protein feels a fluctuating environment whose influence on the electronic dynamics is fixed
entirely by the spectral density of the linear coupling and by the temperature, because the bath is harmonic and starts
in thermal equilibrium. An overdamped Brownian oscillator gives the Drude-Lorentz form, fixed by a reorganization
energy and a cutoff frequency that sets the bath relaxation time. Numerically exact reduced-dynamics methods need the
bath correlation function in a form that can be propagated with the system, and a sum of exponentials in time with
complex amplitudes is the standard choice. The poles of the spectral density give the Drude contribution, and the
poles of the thermal occupation factor give the Matsubara contributions, whose frequencies grow with the index and
matter most at low temperature. Keeping a finite number of Matsubara terms defines the bath model used downstream.

Energies are given in wavenumbers and converted to angular frequencies (rad/ps) with the speed of light in cm/ps, and
thermal energies use the Boltzmann constant in wavenumbers per kelvin, so that hbar = 1 throughout.

Returns
-------
numpy.ndarray of shape (n_matsubara + 1, 3): rows [Re c_k (ps^-2), Im c_k (ps^-2), nu_k (ps^-1)], Drude term first
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drude_lorentz_exponents(reorg_cm: float, cutoff_cm: float, temperature_K: float,
                            n_matsubara: int) -> "np.ndarray":
    '''Exponential expansion of the Drude-Lorentz bath correlation function, truncated after n_matsubara terms.

    Parameters
    ----------
    reorg_cm : float
        Reorganization energy E_R in cm^-1, non-negative. The spectral density is
        S(omega) = 2 E_R gamma_c omega / (omega^2 + gamma_c^2), with E_R and gamma_c converted to rad/ps.
    cutoff_cm : float
        Cutoff gamma_c in cm^-1, positive.
    temperature_K : float
        Temperature in kelvin, positive.
    n_matsubara : int
        Number of Matsubara terms kept after the Drude term, n_matsubara >= 0.

    Returns
    -------
    exponents : np.ndarray
        Shape (n_matsubara + 1, 3). Row k holds [Re c_k, Im c_k, nu_k] such that the bath correlation function
        C(t) = (1/pi) int_0^inf S(omega) [coth(beta omega / 2) cos(omega t) - i sin(omega t)] d omega, t >= 0,
        equals sum_k c_k exp(-nu_k t) when all Matsubara terms are kept. Row 0 is the Drude term (nu_0 = gamma_c) and
        row k >= 1 is the k-th Matsubara term. Amplitudes c_k are in ps^-2 and rates nu_k in ps^-1, using
        omega[rad/ps] = 2 pi c nu[cm^-1] with c = 2.99792458e-2 cm/ps and beta = 1 / (k_B T) with
        k_B = 0.6950348 cm^-1/K, so that hbar = 1.

    Raises
    ------
    ValueError
        If cutoff_cm or temperature_K is not positive, reorg_cm is negative, n_matsubara is negative, or a kept
        Matsubara frequency equals gamma_c to within a relative 1e-9.
    '''
    return exponents

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_drude_lorentz_exponents(reorg_cm: float, cutoff_cm: float, temperature_K: float,
                                    n_matsubara: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if cutoff_cm <= 0 or temperature_K <= 0 or reorg_cm < 0 or n_matsubara < 0:
        raise ValueError("cutoff and temperature must be positive, reorganization energy and n_matsubara non-negative")
    to_radps = 2.0 * np.pi * 2.99792458e-2
    lam = reorg_cm * to_radps
    gam = cutoff_cm * to_radps
    beta = 1.0 / (0.6950348 * temperature_K * to_radps)
    rows = [[lam * gam / np.tan(beta * gam / 2.0), -lam * gam, gam]]
    for k in range(1, int(n_matsubara) + 1):
        nu = 2.0 * np.pi * k / beta
        if abs(nu - gam) <= 1e-9 * gam:
            raise ValueError("a Matsubara frequency coincides with the cutoff")
        rows.append([4.0 * lam * gam / beta * nu / (nu * nu - gam * gam), 0.0, nu])
    return np.array(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: room-temperature pigment bath with one Matsubara term ---
        {
            "setup": "",
            "call": "drude_lorentz_exponents(27.5, 60.0, 295.0, 1)",
            "gold_call": "_oracle_drude_lorentz_exponents(27.5, 60.0, 295.0, 1)",
            "tol": 1e-9,
        },
        # --- Boundary: no Matsubara term, the Drude term alone ---
        {
            "setup": "",
            "call": "drude_lorentz_exponents(100.0, 106.2, 300.0, 0)",
            "gold_call": "_oracle_drude_lorentz_exponents(100.0, 106.2, 300.0, 0)",
            "tol": 1e-9,
        },
        # --- Edge: cryogenic temperature and a fast bath, where Matsubara terms are large and change sign order ---
        {
            "setup": "",
            "call": "drude_lorentz_exponents(35.0, 400.0, 77.0, 4)",
            "gold_call": "_oracle_drude_lorentz_exponents(35.0, 400.0, 77.0, 4)",
            "tol": 1e-9,
        },
        # --- Edge: slow bath at high temperature, cot argument small ---
        {
            "setup": "",
            "call": "drude_lorentz_exponents(8.0, 15.0, 400.0, 3)",
            "gold_call": "_oracle_drude_lorentz_exponents(8.0, 15.0, 400.0, 3)",
            "tol": 1e-9,
        },
        # --- Invalid: a Matsubara frequency equal to the cutoff must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "T = 300.0\n"
                     "gc = 2.0 * np.pi * 0.6950348 * T\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(20.0, gc, T, 2)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(drude_lorentz_exponents)",
            "gold_call": "run(_oracle_drude_lorentz_exponents)",
        },
    ]
