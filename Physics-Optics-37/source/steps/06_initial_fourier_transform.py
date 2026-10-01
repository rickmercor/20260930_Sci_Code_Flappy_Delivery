"""
Transform a square centered physical field into its unnormalized centered two-dimensional Fourier field and the matching angular-wavenumber axes. Use fftshift(fft2(ifftshift(U0))), with spacings dx and dy assigned to array axes zero and one. Return (U,kx,ky), preserving complex amplitudes and the center-origin convention.

The spatial origin is at index (size//2,size//2), and centered FFT frequencies are 2*pi*fftshift(fftfreq(size,d=spacing)). Origin shifts fix Fourier phases, while the same frequency ordering is needed by the derivative diagonal and material shifts. This step fixes a transform convention; it does not assert that performing an additional mathematically consistent transform changes the approximation order.

Returns
-------
return np.zeros_like(U0, dtype=complex), np.zeros(size), np.zeros(size)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def initial_fourier_transform(U0, dx, dy):
    """Return (U, kx, ky) for a square centered physical field U0.

    U = fftshift(fft2(ifftshift(U0))) is an unnormalized centered FFT.
    With size = U0.shape[0], kx = 2*pi*fftshift(fftfreq(size, d=dx))
    and ky = 2*pi*fftshift(fftfreq(size, d=dy)). Positive dx and dy
    measure spacings on axes 0 and 1. The physical origin is at
    (size//2, size//2). Preserve complex amplitudes without additional
    normalization or spatial-cell-area factors.
    """
    size = np.shape(U0)[0]
    return np.zeros_like(U0, dtype=complex), np.zeros(size), np.zeros(size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_initial_fourier_transform(U0, dx, dy):
    """Return the centered Fourier field and angular-wavenumber axes.

    The input origin is at (size//2, size//2). The forward transform
    is unnormalized, with dx and dy assigned to axes 0 and 1.
    """
    field = np.asarray(U0, dtype=complex)

    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("U0 must be a square 2D array.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")

    size = field.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(size, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(size, d=dy))
    return spectrum, kx, ky

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

ny = 8
dx = 0.5
dy = 0.75
rng = np.random.default_rng(10)
U0 = rng.random((ny, ny)) + 1j * rng.random((ny, ny))


def run_model():
    return initial_fourier_transform(U0=U0, dx=dx, dy=dy)


def run_gold():
    return _oracle_initial_fourier_transform(U0=U0, dx=dx, dy=dy)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 16
dx = 0.25
dy = 0.5
U0 = np.zeros((ny, ny), dtype=complex)
U0[ny // 2, ny // 2] = 1.0


def run_model():
    return initial_fourier_transform(U0=U0, dx=dx, dy=dy)


def run_gold():
    return _oracle_initial_fourier_transform(U0=U0, dx=dx, dy=dy)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 12
dx = 0.75
dy = 0.25
first_positions = (np.arange(ny) - ny // 2) * dx
second_positions = (np.arange(ny) - ny // 2) * dy
first_grid, second_grid = np.meshgrid(
    first_positions, second_positions, indexing="ij"
)
U0 = np.exp(-(first_grid**2 + second_grid**2)).astype(complex)


def run_model():
    return initial_fourier_transform(U0=U0, dx=dx, dy=dy)


def run_gold():
    return _oracle_initial_fourier_transform(U0=U0, dx=dx, dy=dy)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 5
dx = 0.4
dy = 0.8
amplitude = 0.7 - 0.2j
positions = np.arange(ny) - ny // 2
phase = 2.0 * np.pi * (
    positions[:, None] + 2 * positions[None, :]
) / ny
U0 = amplitude * np.exp(1j * phase)
expected_spectrum = np.zeros((ny, ny), dtype=complex)
expected_spectrum[ny // 2 + 1, ny // 2 + 2] = ny**2 * amplitude
modes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
expected = (
    expected_spectrum,
    2.0 * np.pi * modes / (ny * dx),
    2.0 * np.pi * modes / (ny * dy),
)
""",
            "call": "initial_fourier_transform(U0=U0, dx=dx, dy=dy)",
            "gold_call": "expected",
        },
    ]
