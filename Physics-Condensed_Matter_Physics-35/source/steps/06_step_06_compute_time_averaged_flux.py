"""
Evaluate the time-averaged flux density the branch passes under one assignment of the two terminal potentials.

The instantaneous flux density is minus the conductivity times the gradient of the potential, and both factors oscillate at the modulation frequency, so the physically meaningful transport quantity is the average of their product over one modulation period. Carrying that average through the Bloch superposition produces a decisive simplification. At a fixed position the plane-wave factor of each state is a constant while its envelope sweeps a full period, so the contribution of a state reduces to its amplitude times its plane-wave factor times the cell average of the conductivity acting on the derivative of that state. In a steady state the time-averaged flux cannot depend on position, since any dependence would accumulate stored quantity somewhere; a contribution whose plane-wave factor varies with position therefore has to vanish identically, and only the state whose decay constant is zero survives the average. The whole net transport of the branch is carried by that one state.




What remains is a single cell average: the conductivity multiplied by the spatial derivative of the envelope of the non-decaying state. In Fourier language that average is a convolution of the conductivity coefficients with the envelope coefficients weighted by the harmonic wavenumber, so it pairs each conductivity coefficient with the envelope coefficient of the opposite index. The derivative brings the harmonic index down as a factor, which kills the term in which both indices vanish, so an unmodulated envelope carries no flux at all under this mechanism. The amplitude of the non-decaying state is linear in the two imposed potentials through the two boundary coefficients, so the flux is linear in them as well. Exchanging the two potentials therefore does not simply reverse the flux, because the two coefficients are not equal and opposite; the difference between the two exchanged values is precisely the nonreciprocity, and their sum, which would vanish for any reciprocal conductor, is proportional to the effective advective coefficient of the medium. The result is real despite being assembled from complex quantities, since the conjugate symmetry of the coefficients of a real profile pairs each term with its conjugate.

Returns
-------
float: the time-averaged flux density passed by the branch under the given terminal potentials, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_time_averaged_flux(sigma_modes: "np.ndarray",
                               star_envelope: "np.ndarray",
                               beta: float,
                               boundary_coefficients: "np.ndarray",
                               phi_high: float,
                               phi_low: float) -> float:
    """Evaluate the time-averaged flux density carried by the branch.

    Parameters
    ----------
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order.
    star_envelope : np.ndarray
        Complex envelope coefficients of the non-decaying state, same shape
        and ordering.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    boundary_coefficients : np.ndarray
        Complex array of shape (2) as returned by the boundary step, the first
        entry belonging to the terminal held at phi_high.
    phi_high : float
        Potential imposed at the terminal belonging to the first coefficient.
    phi_low : float
        Potential imposed at the terminal belonging to the second coefficient.

    Returns
    -------
    flux : float
        Time-averaged flux density in the direction of increasing position, as
        a native Python float.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        star_envelope does not have the same shape as sigma_modes, if
        boundary_coefficients does not hold exactly two entries, if any array
        entry is not finite, if beta is not a finite nonzero number, or if
        either terminal potential is not a finite number.
    """
    return flux  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_time_averaged_flux(sigma_modes: "np.ndarray",
                                       star_envelope: "np.ndarray",
                                       beta: float,
                                       boundary_coefficients: "np.ndarray",
                                       phi_high: float,
                                       phi_low: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    envelope = np.asarray(star_envelope, dtype=complex)
    coefficients = np.asarray(boundary_coefficients, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if envelope.shape != sigma.shape:
        raise ValueError("star_envelope must have the same shape as sigma_modes")
    if coefficients.shape != (2,):
        raise ValueError("boundary_coefficients must hold exactly two entries")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(envelope))
            and np.all(np.isfinite(coefficients))):
        raise ValueError("all coefficient arrays must be finite")
    if not (isinstance(beta, (int, float, np.floating, np.integer))
            and np.isfinite(beta) and float(beta) != 0.0):
        raise ValueError("beta must be a finite nonzero number")
    for name, value in (("phi_high", phi_high), ("phi_low", phi_low)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")

    order = (sigma.size - 1) // 2
    harmonics = np.arange(-order, order + 1)
    # Cell average of the conductivity acting on the derivative of the envelope.
    kernel = np.sum(1j * float(beta) * harmonics * sigma[order - harmonics] * envelope)
    amplitude = coefficients[0] * float(phi_high) + coefficients[1] * float(phi_low)
    return float(np.real(-amplitude * kernel))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark medium and terminal potentials ---
        {
            "setup": """import numpy as np
order = 4
sigma_modes = np.zeros(2 * order + 1, dtype=complex)
sigma_modes[order] = 1.0
sigma_modes[order + 1] = sigma_modes[order - 1] = 0.35
sigma_modes[order + 2] = 0.1 * np.exp(1j * np.pi / 3.0)
sigma_modes[order - 2] = 0.1 * np.exp(-1j * np.pi / 3.0)
star_envelope = np.array([4.421153768720e-04 + 1.078113171e-03j,
                          -2.548503287670e-03 - 1.96855798e-03j,
                          1.253901818431e-02 - 3.10896999e-04j,
                          -5.638429085302e-02 + 1.783868894e-03j,
                          1.0 + 0.0j,
                          -5.638429085302e-02 - 1.783868894e-03j,
                          1.253901818431e-02 + 3.10896999e-04j,
                          -2.548503287670e-03 + 1.96855798e-03j,
                          4.421153768720e-04 - 1.078113171e-03j])
beta = 10.0 * np.pi
boundary_coefficients = np.array([-3.928572598098789 + 0.0j, 4.983980113954025 + 0.0j])
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
        },
        # --- Valid: same medium with the terminal potentials exchanged ---
        {
            "setup": """import numpy as np
order = 4
sigma_modes = np.zeros(2 * order + 1, dtype=complex)
sigma_modes[order] = 1.0
sigma_modes[order + 1] = sigma_modes[order - 1] = 0.35
sigma_modes[order + 2] = 0.1 * np.exp(1j * np.pi / 3.0)
sigma_modes[order - 2] = 0.1 * np.exp(-1j * np.pi / 3.0)
star_envelope = np.array([4.421153768720e-04 + 1.078113171e-03j,
                          -2.548503287670e-03 - 1.96855798e-03j,
                          1.253901818431e-02 - 3.10896999e-04j,
                          -5.638429085302e-02 + 1.783868894e-03j,
                          1.0 + 0.0j,
                          -5.638429085302e-02 - 1.783868894e-03j,
                          1.253901818431e-02 + 3.10896999e-04j,
                          -2.548503287670e-03 + 1.96855798e-03j,
                          4.421153768720e-04 - 1.078113171e-03j])
beta = 10.0 * np.pi
boundary_coefficients = np.array([-3.928572598098789 + 0.0j, 4.983980113954025 + 0.0j])
phi_high = 10.0
phi_low = 30.0
""",
            "call": "float(1.0 + 1.0e6 * compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
        },
        # --- Boundary: structureless envelope, which passes no net flux ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.05, 0.35, 1.0, 0.35, 0.05], dtype=complex)
star_envelope = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=complex)
beta = 10.0 * np.pi
boundary_coefficients = np.array([-2.0 + 0.5j, 3.0 - 0.5j])
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
        },
        # --- Edge: smallest truncation with negative terminal potentials ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([-0.04 - 0.017j, 1.0 + 0.0j, -0.04 + 0.017j], dtype=complex)
beta = 2.0 * np.pi
boundary_coefficients = np.array([-1.25 + 0.03j, 1.9 - 0.03j])
phi_high = -5.0
phi_low = -20.0
""",
            "call": "float(1.0 + 1.0e6 * compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low))",
        },
        # --- Invalid: envelope shape inconsistent with the conductivity array ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex)
beta = 2.0
boundary_coefficients = np.array([-1.0 + 0.0j, 1.0 + 0.0j])
phi_high = 1.0
phi_low = 0.0
def run_model():
    try:
        compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong number of boundary coefficients ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([0.1, 1.0, 0.1], dtype=complex)
beta = 2.0
boundary_coefficients = np.array([-1.0 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j])
phi_high = 1.0
phi_low = 0.0
def run_model():
    try:
        compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, boundary_coefficients, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
