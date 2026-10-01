"""
Solve for the gravitational potential with a single 3D FFT convolution against the Fourier-space kernel and crop to the physical cells.

With the analytic Fourier-space kernel in hand, the gravitational potential is a single three-dimensional convolution evaluated spectrally. Writing the in-plane directions as Fourier series on the periodic box and the vertical direction as a Fourier integral discretized on the padded domain, all volume and normalization factors cancel exactly, leaving

    Phi = 4 * pi * G * IFFT3( Ghat_L * FFT3(rho_padded) ),

with unnormalized forward FFT and 1/N-normalized inverse FFT (the NumPy convention). Because the truncated kernel is even in z, its transform is real and the result is real up to roundoff; the physical potential is the real part restricted to the first Nz_out vertical slices (the padded region carries no physical meaning). This one-step 3D transform is what makes the method compatible with pencil-decomposed parallel FFTs.

Returns
-------
np.ndarray of shape (Nx, Ny, Nz_out), float: the real gravitational potential 4*pi*G * IFFT3(Ghat * FFT3(rho_padded)) on the physical cells.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def spectral_convolution_potential(rho_padded: np.ndarray, greens_hat_grid: np.ndarray,
                                   Nz_out: int, G: float = 1.0) -> np.ndarray:
    """Solve for the potential by one 3D spectral convolution.

    Parameters
    ----------
    rho_padded : np.ndarray
        Zero-padded density of shape (Nx, Ny, Nz_pad) with finite values.
    greens_hat_grid : np.ndarray
        Fourier-space Green's function sampled on the FFT wavenumber grid of
        the padded domain, real array of the same shape as rho_padded.
    Nz_out : int
        Number of physical vertical cells to return (1 <= Nz_out <= Nz_pad).
    G : float
        Gravitational constant (G > 0).

    Returns
    -------
    phi : np.ndarray
        Real potential of shape (Nx, Ny, Nz_out) on the physical cells.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(rho_padded).shape[:2] + (int(Nz_out),), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_spectral_convolution_potential(rho_padded: np.ndarray, greens_hat_grid: np.ndarray,
                                           Nz_out: int, G: float = 1.0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    rho_padded = np.asarray(rho_padded, dtype=float)
    greens_hat_grid = np.asarray(greens_hat_grid, dtype=float)
    if rho_padded.ndim != 3 or min(rho_padded.shape) < 1:
        raise ValueError("rho_padded must be a 3D array of shape (Nx, Ny, Nz_pad)")
    if greens_hat_grid.shape != rho_padded.shape:
        raise ValueError("greens_hat_grid must have the same shape as rho_padded")
    if not np.all(np.isfinite(rho_padded)) or not np.all(np.isfinite(greens_hat_grid)):
        raise ValueError("rho_padded and greens_hat_grid must contain only finite values")
    if not (isinstance(Nz_out, (int, np.integer)) and not isinstance(Nz_out, bool)
            and 1 <= Nz_out <= rho_padded.shape[2]):
        raise ValueError("Nz_out must be an integer with 1 <= Nz_out <= Nz_pad")
    if not (isinstance(G, (int, float)) and np.isfinite(G) and float(G) > 0.0):
        raise ValueError("G must be a finite number > 0")

    phi_padded = np.fft.ifftn(greens_hat_grid * np.fft.fftn(rho_padded)).real
    phi = 4.0 * np.pi * float(G) * phi_padded[:, :, :int(Nz_out)]
    return phi.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: generic deterministic density and smooth kernel (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
Nx, Ny, Nzp = 6, 5, 12
rho_padded = np.zeros((Nx, Ny, Nzp))
rho_padded[:, :, :3] = rng.normal(size=(Nx, Ny, 3))
kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=1.0)
ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=1.0)
kz = 2.0 * np.pi * np.fft.fftfreq(Nzp, d=0.5)
k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
greens_hat_grid = -1.0 / (1.0 + k2)
Nz_out = 3
G = 1.0
""",
            "call": "spectral_convolution_potential(rho_padded, greens_hat_grid, Nz_out, G=G)",
            "gold_call": "_oracle_spectral_convolution_potential(rho_padded, greens_hat_grid, Nz_out, G=G)",
        },
        # --- Boundary: identity kernel, potential must equal 4*pi*G*rho ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
rho_padded = rng.normal(size=(4, 4, 8))
greens_hat_grid = np.ones((4, 4, 8))
Nz_out = 8
G = 2.5
""",
            "call": "spectral_convolution_potential(rho_padded, greens_hat_grid, Nz_out, G=G)",
            "gold_call": "_oracle_spectral_convolution_potential(rho_padded, greens_hat_grid, Nz_out, G=G)",
        },
        # --- Edge: Nz_out = 1, only the bottom physical slice returned ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(13)
rho_padded = rng.normal(size=(3, 3, 4))
greens_hat_grid = rng.normal(size=(3, 3, 4))
Nz_out = 1
""",
            "call": "spectral_convolution_potential(rho_padded, greens_hat_grid, Nz_out)",
            "gold_call": "_oracle_spectral_convolution_potential(rho_padded, greens_hat_grid, Nz_out)",
        },
        # --- Valid: quadruple-padded input cropped back to the physical slices ---
        {
            "setup": """import numpy as np
Nx = Ny = 6
Nz = 5
rho_padded = np.zeros((Nx, Ny, 4 * Nz))
zc = -3.0 + (np.arange(Nz) + 0.5) * (6.0 / Nz)
rho_padded[:, :, :Nz] = np.exp(-0.5 * zc ** 2)[None, None, :]
kz = 2.0 * np.pi * np.fft.fftfreq(4 * Nz, d=6.0 / Nz)
g_hat = -1.0 / (1.0 + kz[None, None, :] ** 2) * np.ones((Nx, Ny, 1))
""",
            "call": "spectral_convolution_potential(rho_padded, g_hat, Nz, G=1.0)",
            "gold_call": "_oracle_spectral_convolution_potential(rho_padded, g_hat, Nz, G=1.0)",
        },
    ]
