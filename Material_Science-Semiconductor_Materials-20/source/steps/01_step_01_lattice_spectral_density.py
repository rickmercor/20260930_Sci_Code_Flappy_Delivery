"""
Step 01 - Deformation-potential spectral density of the dot-lattice coupling.

Spectral density of the longitudinal-acoustic lattice coupling of a quantum dot exciton.

An electron-hole pair confined in a self-assembled GaAs quantum dot couples to bulk
longitudinal-acoustic (LA) phonons through the deformation potential. Writing the
coupling of one exciton as sum_k (g_k b_k^dag + g_k^* b_k), with g_k an angular
frequency, the whole effect of the lattice on the dot is carried by the spectral density
J(omega) = sum_k |g_k|^2 delta(omega - omega_k).

For a linear acoustic dispersion omega = c_s q, electron and hole ground states taken as
isotropic Gaussians psi(r) proportional to exp(-r^2 / (2 a^2)) of widths a_e and a_h, and
deformation potentials D_e (electron) and D_h (hole), the continuum limit gives

    J(omega) = omega^3 / (4 pi^2 rho hbar c_s^5)
               * [ D_e exp(-omega^2 a_e^2 / (4 c_s^2)) - D_h exp(-omega^2 a_h^2 / (4 c_s^2)) ]^2

in SI units (omega in rad/s, D in J, rho the mass density in kg/m^3, c_s in m/s, a in m,
hbar = 1.054571817e-34 J s), a rate in rad/s. Every step of this task works in
picosecond units: omega is given in rad/ps and J is returned in rad/ps (1 rad/ps is
1e12 rad/s), with D_e and D_h in eV (1 eV = 1.602176634e-19 J) and the Gaussian widths
in nm.

The lattice is described by one array used unchanged by every later step,
lattice = [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

J vanishes as omega^3 at low frequency and is cut off where the phonon wavelength
becomes comparable with the confinement length.

Returns
-------
numpy.ndarray of the same shape as omega: the deformation-potential spectral density J(omega) in rad/ps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lattice_spectral_density(omega: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    '''Deformation-potential spectral density J(omega) of the dot-lattice coupling.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of angular frequencies in rad/ps, every entry >= 0.
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    J : np.ndarray
        Array of the same shape as omega, J(omega) in rad/ps.

    Raises
    ------
    ValueError
        If omega is not a one-dimensional array of finite values >= 0, or if lattice is
        not a numpy array of six finite real numbers with the mass density, c_s, a_e and a_h
        positive.
    '''
    return J

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_lattice(lattice):
    """Validate the lattice array; return (De, Dh, density, cs, ae, ah) as floats."""
    arr = np.asarray(lattice)
    if not isinstance(lattice, np.ndarray) or arr.shape != (6,) or arr.dtype.kind not in "iuf":
        raise ValueError("lattice must be a numpy array of six real numbers")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("lattice entries must be finite")
    De, Dh, density, cs, ae, ah = (float(x) for x in arr)
    if density <= 0.0 or cs <= 0.0 or ae <= 0.0 or ah <= 0.0:
        raise ValueError("density, sound velocity and Gaussian widths must be positive")
    return De, Dh, density, cs, ae, ah


def _oracle_lattice_spectral_density(omega: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("omega must be a one-dimensional array of finite values >= 0")
    De, Dh, density, cs, ae, ah = _check_lattice(lattice)
    hbar_si = 1.054571817e-34
    ev = 1.602176634e-19
    wsi = w * 1.0e12
    ae_m, ah_m = ae * 1.0e-9, ah * 1.0e-9
    form = (De * ev * np.exp(-wsi ** 2 * ae_m ** 2 / (4.0 * cs ** 2))
            - Dh * ev * np.exp(-wsi ** 2 * ah_m ** 2 / (4.0 * cs ** 2)))
    return wsi ** 3 / (4.0 * np.pi ** 2 * density * hbar_si * cs ** 5) * form ** 2 * 1.0e-12

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: GaAs dot, frequency grid across the phonon band ---
        {
            "setup": "import numpy as np\n"
                     "w = np.linspace(0.0, 8.0, 41)\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "lattice_spectral_density(w.copy(), lat.copy())",
            "gold_call": "_oracle_lattice_spectral_density(w.copy(), lat.copy())",
            "tol": 1e-10,
        },
        # --- Boundary: zero frequency and far above the cut-off (both vanish) ---
        {
            "setup": "import numpy as np\n"
                     "w = np.array([0.0, 1.0e-4, 25.0, 40.0])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.0, 3.0 / 1.15])\n",
            "call": "lattice_spectral_density(w.copy(), lat.copy())",
            "gold_call": "_oracle_lattice_spectral_density(w.copy(), lat.copy())",
            "tol": 1e-12,
        },
        # --- Edge: small dot, equal electron and hole widths, deformation potentials of the same sign ---
        {
            "setup": "import numpy as np\n"
                     "w = np.array([0.3, 1.1, 2.7, 4.4, 6.0, 9.5])\n"
                     "lat = np.array([8.5, 2.0, 4800.0, 4700.0, 2.0, 2.0])\n",
            "call": "lattice_spectral_density(w.copy(), lat.copy())",
            "gold_call": "_oracle_lattice_spectral_density(w.copy(), lat.copy())",
            "tol": 1e-10,
        },
        # --- Invalid: negative frequency ---
        {
            "setup": "import numpy as np\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.0])\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        lattice_spectral_density(np.array([-0.5, 1.0]), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_lattice_spectral_density(np.array([-0.5, 1.0]), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
