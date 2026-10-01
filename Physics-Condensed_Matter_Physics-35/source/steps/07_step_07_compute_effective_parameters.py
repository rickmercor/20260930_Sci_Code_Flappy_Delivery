"""
Map the modulated branch onto an equivalent advection-diffusion medium and extract its effective conductivity, advective coefficient and their ratio.

Seen from its terminals the modulated branch behaves like an ordinary medium in which diffusion competes with a steady drift. The steady state of such a medium under two fixed potentials is a single exponential profile controlled by the ratio of the advective coefficient to the conductivity, and the flux it passes is a fixed linear combination of the two terminal potentials whose coefficients are set by that ratio, by the advective coefficient and by the separation of the terminals. The exact flux of the modulated branch is also a linear combination of the same two potentials, with coefficients read from the boundary step and from the cell average that appeared in the flux step. Matching the two combinations term by term defines the equivalence, and because there are two independent coefficients to match there is exactly enough information to determine both effective parameters, with no fitting and no free constant left over.




Matching the ratio of the two coefficients removes the overall scale and yields the exponential of the drift-to-diffusion ratio across the branch directly as minus the ratio of the two boundary coefficients; taking a logarithm and dividing by the branch length gives that ratio itself. The sign is the physically loaded part: a positive value means the effective drift runs from the higher-potential terminal towards the lower one, a negative value means it runs against the imposed gradient, and a vanishing value means the medium is reciprocal however strongly it is modulated. Matching either coefficient separately then fixes the advective coefficient as minus the sum of the two boundary coefficients times the same cell average, and the effective conductivity follows as the advective coefficient divided by the ratio. The terminal potentials do not appear anywhere in the extraction, so the ratio characterises the intrinsic nonreciprocity of the material rather than the experiment. Two consistency checks are worth applying: the three numbers must satisfy the defining relation among themselves, and the effective conductivity must come out positive, since a negative value signals that the wrong branch of the logarithm or the wrong pairing of coefficients has been used.

Returns
-------
np.ndarray: real array of shape (3) holding the effective conductivity, the effective advective coefficient and the advection-diffusion ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_effective_parameters(sigma_modes: "np.ndarray",
                                 star_envelope: "np.ndarray",
                                 beta: float,
                                 boundary_coefficients: "np.ndarray",
                                 length: float) -> "np.ndarray":
    """Extract the effective parameters of the equivalent medium.

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
        entry belonging to the higher-potential terminal.
    length : float
        Separation of the two terminals in metres, strictly positive.

    Returns
    -------
    parameters : np.ndarray
        Real array of shape (3) holding the effective conductivity, the
        effective advective coefficient and the advection-diffusion ratio, in
        that order.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        star_envelope does not have the same shape as sigma_modes, if
        boundary_coefficients does not hold exactly two entries, if any array
        entry is not finite, if beta is not a finite nonzero number, if length
        is not a finite number > 0, if the second boundary coefficient is
        zero, if the two boundary coefficients do not give a positive real
        exponential factor to numerical precision, or if they cancel so that
        the matching leaves the effective conductivity undetermined.
    """
    return parameters  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_effective_parameters(sigma_modes: "np.ndarray",
                                         star_envelope: "np.ndarray",
                                         beta: float,
                                         boundary_coefficients: "np.ndarray",
                                         length: float) -> "np.ndarray":
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
    if not (isinstance(length, (int, float, np.floating, np.integer))
            and np.isfinite(length) and float(length) > 0.0):
        raise ValueError("length must be a finite number > 0")
    if coefficients[1] == 0.0:
        raise ValueError("the second boundary coefficient must be nonzero")

    quotient = -coefficients[0] / coefficients[1]
    imaginary_tolerance = 1.0e-10 * max(1.0, abs(float(quotient.real)))
    if quotient.real <= 0.0 or abs(float(quotient.imag)) > imaginary_tolerance:
        raise ValueError("the boundary coefficients must give a positive real "
                         "exponential factor")
    quotient = float(quotient.real)

    order = (sigma.size - 1) // 2
    harmonics = np.arange(-order, order + 1)
    kernel = np.sum(1j * float(beta) * harmonics * sigma[order - harmonics] * envelope)

    logarithm = np.log(quotient)
    if logarithm == 0.0:
        raise ValueError("boundary coefficients that cancel leave the effective "
                         "conductivity undetermined by this matching")
    advection_ratio = float(np.real(logarithm) / float(length))
    advective = float(np.real(-(coefficients[0] + coefficients[1]) * kernel))
    conductivity = float(np.real(-float(length) * (coefficients[0] + coefficients[1])
                                 * kernel / logarithm))
    return np.array([conductivity, advective, advection_ratio], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark branch of five modulation cells ---
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
length = 1.0

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
            "gold_call": "digest(_oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
        },
        # --- Valid: same medium over a longer branch ---
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
length = 2.5

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
            "gold_call": "digest(_oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
        },
        # --- Boundary: nearly cancelling coefficients, a weakly biased branch ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.05, 0.35, 1.0, 0.35, 0.05], dtype=complex)
star_envelope = np.array([0.001 - 0.002j, -0.05 + 0.001j, 1.0 + 0.0j,
                          -0.05 - 0.001j, 0.001 + 0.002j], dtype=complex)
beta = 10.0 * np.pi
boundary_coefficients = np.array([-4.0 + 0.0j, 4.02 + 0.0j])
length = 1.0

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
            "gold_call": "digest(_oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
        },
        # --- Edge: smallest truncation and a strongly biased branch ---
        # The envelope and the boundary coefficients are those of a realisable
        # state: the envelope is conjugate-symmetric with the handedness a real
        # profile produces, so the conductivity-weighted moment comes out real
        # and positive, and both boundary coefficients are real. The three
        # returned numbers therefore satisfy the defining relation among
        # themselves exactly and the effective conductivity comes out positive,
        # which is what a solver enforcing that relation also returns.
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([-0.04 + 0.017j, 1.0 + 0.0j, -0.04 - 0.017j], dtype=complex)
beta = 2.0 * np.pi
boundary_coefficients = np.array([-9.0 + 0.0j, 1.2 + 0.0j])
length = 0.4

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
            "gold_call": "digest(_oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length))",
        },
        # --- Invalid: non-positive branch length ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([-0.04, 1.0, -0.04], dtype=complex)
beta = 2.0
boundary_coefficients = np.array([-2.0 + 0.0j, 3.0 + 0.0j])
length = 0.0
def run_model():
    try:
        compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: boundary coefficients that cancel exactly ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([-0.04, 1.0, -0.04], dtype=complex)
beta = 2.0
boundary_coefficients = np.array([-4.0 + 0.0j, 4.0 + 0.0j])
length = 1.0
def run_model():
    try:
        compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: same-sign coefficients, which admit no real exponential factor ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([-0.04, 1.0, -0.04], dtype=complex)
beta = 2.0
boundary_coefficients = np.array([2.0 + 0.0j, 3.0 + 0.0j])
length = 1.0
def run_model():
    try:
        compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the coefficient quotient is positive but genuinely complex ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
star_envelope = np.array([-0.04, 1.0, -0.04], dtype=complex)
beta = 2.0
boundary_coefficients = np.array([-1.0 - 1.0j, 1.0 + 0.0j])
length = 1.0
def run_model():
    try:
        compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, boundary_coefficients, length)
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
