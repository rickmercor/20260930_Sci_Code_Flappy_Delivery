"""
Evaluate the analytical Fourier-space Green's function of the truncated free-space kernel elementwise on wavenumber grids.

Fourier transforming the Poisson equation in the two periodic in-plane directions turns it into a one-dimensional screened Poisson (Helmholtz) equation in z for each in-plane mode, (d^2/dz^2 - k^2) Phi_tilde = 4*pi*G* rho_tilde with k^2 = kx^2 + ky^2. Its free-space Green's function is G_k(z) = |z|/2 for k = 0 and G_k(z) = -exp(-k|z|)/(2k) for k != 0, which encodes vacuum boundary conditions in z. Truncating this kernel with a rectangular window of half-width L = alpha*Lz (alpha > 1) leaves the convolution over the finite slab unchanged while making the kernel compactly supported, and its continuous Fourier transform in z is analytic:

    Ghat_L(k, kz) = L^2 / 2                                  if k = 0, kz = 0,

    Ghat_L(k, kz) = [kz*L*sin(kz*L) + cos(kz*L) - 1] / kz^2  if k = 0, kz != 0,

    Ghat_L(k, kz) = -[exp(-k*L)*((kz/k)*sin(kz*L) - cos(kz*L)) + 1]

                     / (k^2 + kz^2)                           otherwise.

The kernel is regular at (k, kz) = (0, 0), so no ad hoc zero-mode fix is needed. This step evaluates Ghat_L elementwise on wavenumber grids.

Returns
-------
np.ndarray, float, with the broadcast shape of (k_perp, kz): the three-branch Fourier-space Green's function values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def greens_function_hat(k_perp: np.ndarray, kz: np.ndarray, L: float) -> np.ndarray:
    """Evaluate the truncated free-space Green's function in Fourier space.

    Parameters
    ----------
    k_perp : np.ndarray
        In-plane wavenumber magnitudes sqrt(kx^2 + ky^2), all >= 0. May be
        any shape broadcastable against kz.
    kz : np.ndarray
        Vertical wavenumbers, broadcastable against k_perp.
    L : float
        Truncation half-width of the kernel, L = alpha * Lz with alpha > 1
        (L > 0).

    Returns
    -------
    g_hat : np.ndarray
        The analytic Fourier-space Green's function evaluated elementwise on
        the broadcast shape of (k_perp, kz), as a float array.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.broadcast(np.asarray(k_perp), np.asarray(kz)).shape, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_greens_function_hat(k_perp: np.ndarray, kz: np.ndarray, L: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(L, (int, float)) and np.isfinite(L) and float(L) > 0.0):
        raise ValueError("L must be a finite number > 0")
    k_perp = np.asarray(k_perp, dtype=float)
    kz = np.asarray(kz, dtype=float)
    if not np.all(np.isfinite(k_perp)) or not np.all(np.isfinite(kz)):
        raise ValueError("k_perp and kz must contain only finite values")
    if np.any(k_perp < 0.0):
        raise ValueError("k_perp must be non-negative everywhere")
    try:
        k_perp, kz = np.broadcast_arrays(k_perp, kz)
    except ValueError:
        raise ValueError("k_perp and kz must have broadcast-compatible shapes")

    L = float(L)
    g_hat = np.empty(k_perp.shape, dtype=float)

    zero_zero = (k_perp == 0.0) & (kz == 0.0)
    zero_kz = (k_perp == 0.0) & (kz != 0.0)
    nonzero = k_perp != 0.0

    g_hat[zero_zero] = 0.5 * L * L

    # The k = 0 branch is [u sin u + cos u - 1] * L^2 / u^2 with u = kz * L.
    # Both u sin u and cos u - 1 are O(u^2), so the closed form loses all
    # significance as u -> 0 (catastrophic cancellation: cos u rounds to 1).
    # Below a cutoff use the Taylor series L^2 * (1/2 - u^2/8 + u^4/144),
    # which is exact to double precision there and tends to L^2/2 at u = 0.
    kz_v = kz[zero_kz]
    u = kz_v * L
    small = np.abs(u) < 1.0e-3
    out = np.empty(u.shape, dtype=float)
    us = u[small]
    out[small] = L * L * (0.5 - us**2 / 8.0 + us**4 / 144.0)
    ub = u[~small]
    out[~small] = (ub * np.sin(ub) + np.cos(ub) - 1.0) * (L * L) / ub**2
    g_hat[zero_kz] = out

    kp_v = k_perp[nonzero]
    kz_v = kz[nonzero]
    g_hat[nonzero] = -(
        np.exp(-kp_v * L) * ((kz_v / kp_v) * np.sin(kz_v * L) - np.cos(kz_v * L)) + 1.0
    ) / (kp_v**2 + kz_v**2)

    return g_hat

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the three branches at known analytic points ---
        # Expected values derived independently from the closed form, not from
        # the oracle. With L = 6.6: the zero mode is L^2/2 = 21.78; the k = 0
        # branch must tend to that same limit as kz -> 0 (a stable evaluation
        # is required, since the closed form cancels catastrophically there);
        # and the k != 0 branch at k = sqrt(2)*2*pi/6 is -0.4559194.
        {
            "setup": """import numpy as np
k_bench = np.sqrt(2.0) * 2.0 * np.pi / 6.0
EXPECTED = np.array([21.78, 21.78, 21.78, -0.4559194])
""",
            "call": ("np.round(np.array([greens_function_hat(np.array(0.0), np.array(0.0), 6.6),"
                     " greens_function_hat(np.array(0.0), np.array(1e-8), 6.6),"
                     " greens_function_hat(np.array(0.0), np.array(1e-12), 6.6),"
                     " greens_function_hat(np.array(k_bench), np.array(0.0), 6.6)],"
                     " dtype=float), 7)"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: mixed grid containing all three branches (normal scenario) ---
        {
            "setup": """import numpy as np
k_perp = np.array([[0.0], [0.0], [1.2], [2.5]])
kz = np.array([[0.0, 0.9, 3.1]])
L = 6.6
""",
            "call": "greens_function_hat(k_perp, kz, L)",
            "gold_call": "_oracle_greens_function_hat(k_perp, kz, L)",
        },
        # --- Valid: full FFT-ordered wavenumber grids as used in the solver ---
        {
            "setup": """import numpy as np
Nx, Nz = 8, 16
kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=6.0 / Nx)
ky = 2.0 * np.pi * np.fft.fftfreq(Nx, d=6.0 / Nx)
k_perp = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)[:, :, None]
kz = (2.0 * np.pi * np.fft.fftfreq(Nz, d=6.0 / Nz))[None, None, :]
L = 1.1 * 6.0
""",
            "call": "greens_function_hat(k_perp, kz, L)",
            "gold_call": "_oracle_greens_function_hat(k_perp, kz, L)",
        },
        # --- Boundary: single scalar zero mode, Ghat = L^2 / 2 ---
        {
            "setup": """import numpy as np
k_perp = np.array(0.0)
kz = np.array(0.0)
L = 3.3
""",
            "call": "greens_function_hat(k_perp, kz, L)",
            "gold_call": "_oracle_greens_function_hat(k_perp, kz, L)",
        },
        # --- Edge: negative and very small kz on the k_perp = 0 line ---
        {
            "setup": """import numpy as np
k_perp = np.zeros(4)
kz = np.array([-2.0, -1.0e-8, 1.0e-8, 2.0])
L = 6.6
""",
            "call": "greens_function_hat(k_perp, kz, L)",
            "gold_call": "_oracle_greens_function_hat(k_perp, kz, L)",
        },
    ]
