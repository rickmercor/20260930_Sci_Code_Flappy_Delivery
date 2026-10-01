"""
The cell operator of periodic conductivity homogenisation maps a periodic potential v to A v = -div(C grad v). On the odd sample grid it is discretised in the standard Fourier fashion: the gradient is the exact derivative of the trigonometric interpolant, that is multiplication by 2 pi i k_d / L_d for every frequency k_d in -m, ..., m, the conductivity multiplies the gradient pointwise at the sample points, and the divergence is again spectral. Because the frequency set is symmetric and the conductivity is real, the discrete operator is real and symmetric with respect to the pixel-average inner product, and its bilinear form is the average of grad v_1 . C grad v_2 over the sample points, exactly as in the continuous weak form. The result of the divergence has zero frequency component equal to zero by construction, so the operator maps into mean-free fields and its only null vector on mean-free periodic fields is absent; on all periodic fields the constants are its kernel. The function accepts a batch of fields so that the operator can later be applied to a whole basis at once. Conductivity values may be zero, which is what allows the same routine to serve for an indicator function later on.

Returns
-------
np.ndarray of the same shape as field holding the real values of -div(C grad v) at the sample points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_periodic_conductivity_operator(
    field: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
) -> np.ndarray:
    """Apply the Fourier-discretised conductivity operator to one or more fields.

    Raises
    ------
    ValueError
        If the last two axes of field are not square with an odd size of at least three, if conductivity does not have that shape or contains a negative or non-finite value, or if a cell length is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def _oracle_apply_periodic_conductivity_operator(
    field: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
) -> np.ndarray:
    """Reference implementation."""
    field = np.asarray(field, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    if field.ndim < 2 or field.shape[-1] != field.shape[-2] or field.shape[-1] < 3 or field.shape[-1] % 2 == 0:
        raise ValueError("field must end with two square axes of odd size at least three")
    n_pixels = field.shape[-1]
    if conductivity.shape != (n_pixels, n_pixels) or not np.all(np.isfinite(conductivity)) or np.any(conductivity < 0.0):
        raise ValueError("conductivity must be a finite non-negative array matching the grid")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    multipliers = _frequency_multipliers(n_pixels, lengths)
    gradient_1, gradient_2 = _spectral_gradient(field, multipliers)
    return -_spectral_divergence(conductivity * gradient_1, conductivity * gradient_2, multipliers)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
N, L = 15, (1.0, 2.0)
x = np.arange(N) * L[0] / N
y = np.arange(N) * L[1] / N
X, Y = np.meshgrid(x, y, indexing="ij")
MODE = np.cos(2 * np.pi * (3 * X / L[0] - 2 * Y / L[1]) + 0.4)
EXACT = 5.5 * (2 * np.pi) ** 2 * ((3 / L[0]) ** 2 + (2 / L[1]) ** 2) * MODE
def summarize(out):
    return (out.shape, round(float(np.abs(out - EXACT).max()), 9), round(float(out.mean()), 12))
""",
            "call": "summarize(apply_periodic_conductivity_operator(MODE, np.full((N, N), 5.5), L))",
            "gold_call": "summarize(_oracle_apply_periodic_conductivity_operator(MODE, np.full((N, N), 5.5), L))",
        },
        {
            "setup": """import numpy as np
N, L = 21, (1.0, 1.0)
rng = np.random.default_rng(7)
C = np.where(rng.random((N, N)) < 0.3, 10.0, 1.0)
V = rng.standard_normal((3, N, N))
def summarize(fn):
    out = fn(V, C, L)
    W = rng.standard_normal((N, N))
    lhs = np.mean(W * out[1])
    rhs = np.mean(V[1] * fn(W, C, L))
    return (out.shape, round(float(abs(lhs - rhs)), 9), round(float(np.abs(out.mean(axis=(1, 2))).max()), 12),
            round(float(np.mean(V[0] * out[0])), 9), round(float(out[2, 4, 7]), 9))
""",
            "call": "summarize(apply_periodic_conductivity_operator)",
            "gold_call": "summarize(_oracle_apply_periodic_conductivity_operator)",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(np.ones((16, 16)), np.ones((16, 16)), (1.0, 1.0))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(apply_periodic_conductivity_operator)",
            "gold_call": "run(_oracle_apply_periodic_conductivity_operator)",
        },
    ]
