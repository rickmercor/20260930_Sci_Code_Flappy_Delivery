"""
Step 03 - Response functions of the residual polaron-frame coupling.

Response functions of the residual dot-lattice coupling in the polaron frame.

After the polaron transformation of step 02, a dot-cavity coupling term that changes the
exciton number by one acquires a displacement factor B_+ or B_-. Separating the thermal
average <B> from the fluctuations leaves two Hermitian bath operators,

    B_x = (B_+ + B_- - 2 <B>) / 2,        B_y = (B_+ - B_-) / (2 i),

which carry the residual coupling that a perturbative master equation treats to second
order. Their thermal correlation functions are C_u(tau) = < B_u(tau) B_u(0) > for u = x, y
(free phonon evolution as in step 02, lattice in equilibrium at temperature T). A
Born-Markov master equation needs their one-sided Fourier transforms

    kappa_u(omega) = integral_0^infinity  C_u(tau) exp(i omega tau) d tau ,

evaluated at real angular frequencies omega of either sign (rad/ps); kappa_u is in ps.

Every entry must be converged to at least eight significant figures. Units and constants
are those of step 02; the lattice array is the one of step 01.

Returns
-------
numpy.ndarray of shape (4, len(omega)): [Re kappa_x, Im kappa_x, Re kappa_y, Im kappa_y] in ps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def polaron_response_functions(omega: np.ndarray, T: float, lattice: np.ndarray) -> np.ndarray:
    '''One-sided Fourier transforms kappa_x(omega) and kappa_y(omega) of the polaron bath.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of real angular frequencies in rad/ps (any sign).
    T : float
        Lattice temperature in K, > 0.
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    kappa : np.ndarray
        Shape (4, len(omega)): rows are Re kappa_x, Im kappa_x, Re kappa_y, Im kappa_y,
        in ps.

    Raises
    ------
    ValueError
        If omega is not a one-dimensional array of finite values, if T is not finite
        and positive, or if lattice is invalid (as in step 01).
    '''
    return kappa

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _polaron_times(T, lattice):
    """Uniform Simpson time grid (ps) long enough for C_u to have decayed."""
    De, Dh, density, cs, ae, ah = _check_lattice(lattice)
    hbar, kb = 0.6582119569, 0.08617333262
    tmax = max(20.0, 10.0 * hbar / (kb * T), 40.0 * max(ae, ah) * 1.0e3 / cs)
    n = int(np.ceil(tmax / 0.005))
    n += n % 2
    t = np.linspace(0.0, tmax, n + 1)
    wts = np.full(t.size, 2.0)
    wts[1:-1:2] = 4.0
    wts[0] = wts[-1] = 1.0
    return t, wts * (t[1] - t[0]) / 3.0


def _oracle_polaron_response_functions(omega: np.ndarray, T: float, lattice: np.ndarray) -> np.ndarray:
    om = np.asarray(omega, dtype=float)
    if om.ndim != 1 or not np.all(np.isfinite(om)):
        raise ValueError("omega must be a one-dimensional array of finite values")
    T = _check_temperature(T)
    t, wts = _polaron_times(T, lattice)
    parts = [_oracle_phonon_propagator(t[i:i + 2000], T, lattice) for i in range(0, t.size, 2000)]
    ph = np.concatenate(parts, axis=1)
    phi = ph[0] + 1j * ph[1]
    b2 = np.exp(-phi[0].real)
    cx = 0.5 * b2 * (np.exp(phi) + np.exp(-phi) - 2.0)
    cy = 0.5 * b2 * (np.exp(phi) - np.exp(-phi))
    kern = np.exp(1j * np.outer(om, t))
    kx = kern @ (cx * wts)
    ky = kern @ (cy * wts)
    return np.array([kx.real, kx.imag, ky.real, ky.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: GaAs dot at 20 K, frequencies of either sign ---
        {
            "setup": "import numpy as np\n"
                     "om = np.array([-2.0, -1.2, -0.45, 0.0, 0.3, 0.9, 1.6, 2.4])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "polaron_response_functions(om.copy(), 20.0, lat.copy())",
            "gold_call": "_oracle_polaron_response_functions(om.copy(), 20.0, lat.copy())",
            "tol": 1e-8,
        },
        # --- Boundary: cold lattice (2 K) ---
        {
            "setup": "import numpy as np\n"
                     "om = np.array([-1.0, -0.2, 0.2, 1.0])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.0, 3.0 / 1.15])\n",
            "call": "polaron_response_functions(om.copy(), 2.0, lat.copy())",
            "gold_call": "_oracle_polaron_response_functions(om.copy(), 2.0, lat.copy())",
            "tol": 1e-8,
        },
        # --- Edge: hot lattice (60 K) and a large dot ---
        {
            "setup": "import numpy as np\n"
                     "om = np.array([-3.0, -0.7, 0.05, 0.7, 3.0])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 5.0, 4.2])\n",
            "call": "polaron_response_functions(om.copy(), 60.0, lat.copy())",
            "gold_call": "_oracle_polaron_response_functions(om.copy(), 60.0, lat.copy())",
            "tol": 1e-8,
        },
        # --- Invalid: non-finite frequency ---
        {
            "setup": "import numpy as np\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.0])\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        polaron_response_functions(np.array([0.1, np.nan]), 10.0, lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_polaron_response_functions(np.array([0.1, np.nan]), 10.0, lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
