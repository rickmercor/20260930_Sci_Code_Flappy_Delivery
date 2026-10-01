"""
Implement compute_shielding_fourier_components which returns the rotor-modulated Fourier coefficients of a single-spin shielding Hamiltonian.

Under sample rotation the secular shielding term of one spin is a linear operator whose
coefficient is periodic in the rotor phase. Writing that coefficient as a finite Fourier series
in the rotor frequency turns a continuously modulated interaction into five complex numbers that
depend only on the interaction parameters and on the orientation of the crystallite.

The isotropic part is rotation invariant and contributes only to the stationary coefficient,
while the anisotropic part is carried from the principal axis frame of the shielding tensor into
the rotor frame by a second-rank Wigner rotation and from the rotor frame into the laboratory
frame by the rotor inclination. The asymmetry parameter mixes the two outer principal-frame
components into the same coefficients.

Second-rank Wigner rotation matrix elements follow the convention
D^(2)_{m',m}(alpha, beta, gamma) = exp(-i*m'*alpha) * d^(2)_{m',m}(beta) * exp(-i*m*gamma),
with d^(2) the real reduced Wigner matrix.

Returns
-------
np.ndarray, complex array of shape (5,) holding the rotor-frame shielding Fourier coefficients for m = -2, -1, 0, 1, 2 in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_shielding_fourier_components(omega_iso: float, omega_aniso: float, eta: float,
                                         alpha_pr: float, beta_pr: float, gamma_pr: float,
                                         beta_rl: float) -> "np.ndarray":
    '''Compute the five rotor-frame Fourier coefficients of a shielding interaction.

    The coefficients are defined so that the instantaneous secular shielding
    frequency of the spin is sum over m from -2 to 2 of
    components[m + 2] * exp(i * m * omega_r * t).

    Parameters
    ----------
    omega_iso : float
        Isotropic shielding frequency in rad/s, measured relative to the
        radio-frequency carrier.
    omega_aniso : float
        Anisotropic shielding frequency in rad/s.
    eta : float
        Shielding asymmetry parameter, 0 <= eta <= 1.
    alpha_pr : float
        First Euler angle in radians carrying the shielding principal axis
        frame into the rotor frame.
    beta_pr : float
        Second Euler angle in radians carrying the shielding principal axis
        frame into the rotor frame.
    gamma_pr : float
        Third Euler angle in radians carrying the shielding principal axis
        frame into the rotor frame.
    beta_rl : float
        Angle in radians between the rotor axis and the static field.

    Returns
    -------
    components : np.ndarray
        Complex array of shape (5,) holding the coefficients for
        m = -2, -1, 0, 1, 2 in that order.

    Raises
    ------
    ValueError
        If eta is outside the interval [0, 1], or if any argument is not
        finite.
    '''
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import factorial


def _reduced_wigner_d2(beta: float) -> "np.ndarray":
    """Return the real reduced Wigner matrix d^(2)(beta) indexed from m = 2 down to m = -2."""
    rank = 2
    cos_half = np.cos(beta / 2.0)
    sin_half = np.sin(beta / 2.0)
    matrix = np.zeros((5, 5), dtype=float)
    orders = list(range(2, -3, -1))
    for row, m_out in enumerate(orders):
        for col, m_in in enumerate(orders):
            total = 0.0
            for k in range(2 * rank + 1):
                exponents = (rank + m_in - k, k, m_out - m_in + k, rank - m_out - k)
                if min(exponents) < 0:
                    continue
                numerator = ((-1.0) ** (m_out - m_in + k)) * np.sqrt(
                    factorial(rank + m_out) * factorial(rank - m_out)
                    * factorial(rank + m_in) * factorial(rank - m_in))
                denominator = 1.0
                for value in exponents:
                    denominator *= factorial(value)
                total += (numerator / denominator
                          * cos_half ** (2 * rank + m_in - m_out - 2 * k)
                          * sin_half ** (m_out - m_in + 2 * k))
            matrix[row, col] = total
    return matrix


def _wigner_index(order: int) -> int:
    """Map a second-rank order m to its row or column index in the reduced Wigner matrix."""
    return 2 - order


def _wigner_d2_element(matrix: "np.ndarray", m_out: int, m_in: int) -> float:
    """Return the reduced Wigner element d^(2)_{m_out, m_in} from a precomputed matrix."""
    return float(matrix[_wigner_index(m_out), _wigner_index(m_in)])


def _oracle_compute_shielding_fourier_components(omega_iso: float, omega_aniso: float, eta: float,
                                                 alpha_pr: float, beta_pr: float, gamma_pr: float,
                                                 beta_rl: float) -> "np.ndarray":
    values = (omega_iso, omega_aniso, eta, alpha_pr, beta_pr, gamma_pr, beta_rl)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("all shielding parameters must be finite")
    eta = float(eta)
    if eta < 0.0 or eta > 1.0:
        raise ValueError("eta must satisfy 0 <= eta <= 1")

    d_pr = _reduced_wigner_d2(float(beta_pr))
    d_rl = _reduced_wigner_d2(float(beta_rl))

    def _wigner_rotation(m_out, m_in):
        return (np.exp(-1j * m_out * float(alpha_pr))
                * _wigner_d2_element(d_pr, m_out, m_in)
                * np.exp(-1j * m_in * float(gamma_pr)))

    components = np.zeros(5, dtype=complex)
    for index, m in enumerate(range(-2, 3)):
        principal = (_wigner_rotation(0, -m)
                     - (eta / np.sqrt(6.0)) * (_wigner_rotation(-2, -m) + _wigner_rotation(2, -m)))
        components[index] = (float(omega_aniso) * principal
                             * _wigner_d2_element(d_rl, -m, 0))
        if m == 0:
            components[index] += float(omega_iso)
    return components

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    magic = "beta_rl = float(np.arccos(1.0 / np.sqrt(3.0)))\n"
    common = "import numpy as np\n" + magic

    def _flat(call):
        return ("[round(float(v), 6) for v in np.concatenate([np.real(" + call
                + "), np.imag(" + call + ")])]")

    return [
        # Normal: the graded shielding, a general crystallite and magic-angle
        # rotation.
        {
            "setup": common,
            "call": _flat("compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.3, 0.9, 1.7, beta_rl)"),
            "gold_call": _flat("_oracle_compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.3, 0.9, 1.7, beta_rl)"),
        },
        # Boundary: an axially symmetric tensor removes the outer principal-frame
        # contributions entirely.
        {
            "setup": common,
            "call": _flat("compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.0, 0.3, 0.9, 1.7, beta_rl)"),
            "gold_call": _flat("_oracle_compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.0, 0.3, 0.9, 1.7, beta_rl)"),
        },
        # Edge: at the magic angle the stationary coefficient carries only the
        # isotropic offset, for any crystallite and any asymmetry.
        {
            "setup": common,
            "call": "[round(float(np.real(compute_shielding_fourier_components(2.0 * np.pi * 12.0e3, 2.0 * np.pi * 150.0e3, 1.0, a, b, g, beta_rl)[2])), 6) for a, b, g in [(0.0, 0.0, 0.0), (0.4, 1.1, 2.2), (5.0, 2.7, 0.1)]]",
            "gold_call": "[round(float(np.real(_oracle_compute_shielding_fourier_components(2.0 * np.pi * 12.0e3, 2.0 * np.pi * 150.0e3, 1.0, a, b, g, beta_rl)[2])), 6) for a, b, g in [(0.0, 0.0, 0.0), (0.4, 1.1, 2.2), (5.0, 2.7, 0.1)]]",
        },
        # Edge: the coefficients must obey the conjugate symmetry that makes the
        # reconstructed shielding frequency real.
        {
            "setup": common,
            "call": "(lambda c: round(float(np.max(np.abs(c[::-1] - np.conj(c)))), 10))(compute_shielding_fourier_components(2.0 * np.pi * 5.0e3, 2.0 * np.pi * 80.0e3, 0.73, 1.3, 2.1, 0.55, beta_rl))",
            "gold_call": "(lambda c: round(float(np.max(np.abs(c[::-1] - np.conj(c)))), 10))(_oracle_compute_shielding_fourier_components(2.0 * np.pi * 5.0e3, 2.0 * np.pi * 80.0e3, 0.73, 1.3, 2.1, 0.55, beta_rl))",
        },
        # Boundary: the maximum asymmetry, which weights the outer components
        # most strongly.
        {
            "setup": common,
            "call": _flat("compute_shielding_fourier_components(2.0 * np.pi * 3.0e3, 2.0 * np.pi * 80.0e3, 1.0, 0.0, 1.2, 0.0, beta_rl)"),
            "gold_call": _flat("_oracle_compute_shielding_fourier_components(2.0 * np.pi * 3.0e3, 2.0 * np.pi * 80.0e3, 1.0, 0.0, 1.2, 0.0, beta_rl)"),
        },
        # Edge: a crystallite whose principal z axis is along the rotor axis
        # leaves only the stationary rotor-frame component of the tensor.
        {
            "setup": common,
            "call": _flat("compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.0, 0.0, 0.0, beta_rl)"),
            "gold_call": _flat("_oracle_compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.0, 0.0, 0.0, beta_rl)"),
        },
        # Edge: a non-spinning rotor axis along the field, where the shielding
        # becomes purely stationary and the anisotropy survives in full.
        {
            "setup": common,
            "call": _flat("compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.3, 0.9, 1.7, 0.0)"),
            "gold_call": _flat("_oracle_compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.3, 0.9, 1.7, 0.0)"),
        },
        # Invalid input must raise ValueError rather than return coefficients.
        {
            "setup": "\n\nimport numpy as np\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "_exception_code(compute_shielding_fourier_components, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.9553166181245093)",
            "gold_call": "_exception_code(_oracle_compute_shielding_fourier_components, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.9553166181245093)",
        },
    ]
