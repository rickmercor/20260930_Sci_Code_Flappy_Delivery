"""
Implement ao_eri_tensor, which returns the exact four-index electron repulsion integral tensor over a basis of normalized s-type primitive Gaussians.

The electron repulsion integral (mu nu | lambda sigma) in chemists' notation is the Coulomb interaction between the charge distribution that primitives mu and nu form for one electron and the distribution that lambda and sigma form for the other. For s-type primitives each such distribution is itself a spherical Gaussian, so the six-dimensional integral collapses to a closed form built from a standard one-dimensional auxiliary function of a single argument. That argument vanishes for concentric distributions, whose value is the limit of the auxiliary function and has to be taken separately in floating-point arithmetic.



The tensor carries the full eightfold permutational symmetry of real orbitals, and it is to be evaluated exactly rather than by quadrature. Storing it explicitly costs the fourth power of the basis size, which is the bottleneck that separable factorizations of the tensor are designed to remove.

Returns
-------
np.ndarray of shape (n_ao, n_ao, n_ao, n_ao), float: the exact electron repulsion integrals in chemists' notation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ao_eri_tensor(ao_centers: "np.ndarray", ao_exponents: "np.ndarray") -> "np.ndarray":
    '''Exact analytic electron repulsion integrals over normalized s-type primitives.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.

    Returns
    -------
    eri : np.ndarray
        Array of shape (n_ao, n_ao, n_ao, n_ao) of dtype float whose entry
        (mu, nu, lambda, sigma) is the exact integral (mu nu | lambda sigma) in chemists'
        notation, so that the first two indices label the charge distribution of electron
        one and the last two that of electron two.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, or if any
        exponent is not strictly positive.
    '''
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def _boys0(argument):
    """Zeroth-order Boys function, taking the removable limit one at the origin."""
    argument = np.asarray(argument, dtype=float)
    value = np.ones_like(argument)
    large = argument > 1e-12
    scaled = argument[large]
    value[large] = 0.5 * np.sqrt(np.pi / scaled) * erf(np.sqrt(scaled))
    return value


def _oracle_ao_eri_tensor(ao_centers: "np.ndarray",
                          ao_exponents: "np.ndarray") -> "np.ndarray":
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    n_ao = centers.shape[0]
    norm = (2.0 * exponents / np.pi) ** 0.75

    total = exponents[:, None] + exponents[None, :]
    separation2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    prefactor = np.exp(-(exponents[:, None] * exponents[None, :]) / total * separation2)
    midpoint = (exponents[:, None, None] * centers[:, None, :]
                + exponents[None, :, None] * centers[None, :, :]) / total[:, :, None]

    pair_exponent = total.reshape(-1)
    pair_prefactor = (prefactor * norm[:, None] * norm[None, :]).reshape(-1)
    pair_center = midpoint.reshape(-1, 3)

    separation_pq2 = np.sum((pair_center[:, None, :] - pair_center[None, :, :]) ** 2, axis=-1)
    exponent_sum = pair_exponent[:, None] + pair_exponent[None, :]
    rho = (pair_exponent[:, None] * pair_exponent[None, :]) / exponent_sum
    eri = (2.0 * np.pi ** 2.5
           / (pair_exponent[:, None] * pair_exponent[None, :] * np.sqrt(exponent_sum))
           * pair_prefactor[:, None] * pair_prefactor[None, :]
           * _boys0(rho * separation_pq2))
    return eri.reshape(n_ao, n_ao, n_ao, n_ao)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    symmetry_setup = """import numpy as np
def symmetry_and_values(fn, ao_centers, ao_exponents):
    tensor = np.asarray(fn(ao_centers.copy(), ao_exponents.copy()), dtype=float)
    n = ao_centers.shape[0]
    if tensor.shape != (n, n, n, n):
        raise AssertionError("electron repulsion tensor must have shape (n_ao,)*4")
    deviations = [
        np.abs(tensor - tensor.transpose(1, 0, 2, 3)).max(),
        np.abs(tensor - tensor.transpose(0, 1, 3, 2)).max(),
        np.abs(tensor - tensor.transpose(2, 3, 0, 1)).max(),
    ]
    return np.concatenate([tensor.reshape(-1), np.asarray(deviations, dtype=float)])
"""
    return [
        # --- Typical: three primitives spanning three centers, with permutational symmetry ---
        {
            "setup": symmetry_setup + """
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [1.1, 0.3, 0.0],
                       [0.2, 1.4, 0.5]], dtype=float)
ao_exponents = np.array([0.5, 1.3, 2.2], dtype=float)
""",
            "call": "symmetry_and_values(ao_eri_tensor, ao_centers, ao_exponents)",
            "gold_call": "symmetry_and_values(_oracle_ao_eri_tensor, ao_centers, ao_exponents)",
        },
        # --- Edge: one primitive, whose single entry is the analytic self-repulsion ---
        {
            "setup": """import numpy as np
ao_centers = np.array([[0.0, 0.0, 0.0]], dtype=float)
ao_exponents = np.array([1.0], dtype=float)
""",
            "call": (
                "float(np.asarray(ao_eri_tensor(ao_centers.copy(), ao_exponents.copy()))"
                ".reshape(-1)[0])"
            ),
            "gold_call": (
                "float(np.asarray(_oracle_ao_eri_tensor(ao_centers.copy(), ao_exponents.copy()))"
                ".reshape(-1)[0])"
            ),
        },
        # --- Boundary: coincident centers, where every Gaussian prefactor is exactly one ---
        {
            "setup": """import numpy as np
ao_centers = np.zeros((3, 3), dtype=float)
ao_exponents = np.array([0.18, 0.45, 2.8125], dtype=float)
""",
            "call": "ao_eri_tensor(ao_centers.copy(), ao_exponents.copy())",
            "gold_call": "_oracle_ao_eri_tensor(ao_centers.copy(), ao_exponents.copy())",
        },
        # --- Edge: a widely separated pair approaching the point-charge limit ---
        {
            "setup": """import numpy as np
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [12.0, 0.0, 0.0]], dtype=float)
ao_exponents = np.array([1.2, 0.9], dtype=float)
""",
            "call": "ao_eri_tensor(ao_centers.copy(), ao_exponents.copy())",
            "gold_call": "_oracle_ao_eri_tensor(ao_centers.copy(), ao_exponents.copy())",
        },
        # --- Invalid: a negative exponent ---
        {
            "setup": """import numpy as np
ao_centers = np.zeros((2, 3), dtype=float)
ao_exponents = np.array([1.0, -0.5], dtype=float)
def run_model():
    try:
        ao_eri_tensor(ao_centers.copy(), ao_exponents.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_ao_eri_tensor(ao_centers.copy(), ao_exponents.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
