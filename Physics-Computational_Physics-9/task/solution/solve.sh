#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_density_grid(N: int, L: float, H: float, m_x: int = 1,
                               m_y: int = 1) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool) and N >= 4):
        raise ValueError("N must be an integer >= 4")
    if not (isinstance(L, (int, float)) and np.isfinite(L) and float(L) > 0.0):
        raise ValueError("L must be a finite number > 0")
    if not (isinstance(H, (int, float)) and np.isfinite(H) and float(H) > 0.0):
        raise ValueError("H must be a finite number > 0")
    for name, val in (("m_x", m_x), ("m_y", m_y)):
        if not (isinstance(val, (int, np.integer)) and not isinstance(val, bool)):
            raise ValueError(f"{name} must be an integer")
    if int(m_x) == 0 and int(m_y) == 0:
        raise ValueError("m_x and m_y must not both be zero")

    N = int(N)
    L = float(L)
    H = float(H)

    # -- Cell-centered coordinates: the first centre sits half a cell inside
    #    the lower face, so no sample lands on a box face.
    d = L / N
    coords = -L / 2.0 + (np.arange(N) + 0.5) * d
    x_3d, y_3d, z_3d = np.meshgrid(coords, coords, coords, indexing="ij")

    # -- Single in-plane Fourier mode with a truncated Gaussian vertical
    #    profile; vacuum outside the box is implied by the box-confined support.
    kx_mode = 2.0 * np.pi * int(m_x) / L
    ky_mode = 2.0 * np.pi * int(m_y) / L
    rho = np.cos(kx_mode * x_3d) * np.cos(ky_mode * y_3d) * np.exp(-0.5 * (z_3d / H) ** 2)

    return np.stack([x_3d, y_3d, z_3d, rho]).astype(float)

def compute_shear_wavevectors(Nx: int, Ny: int, Lx: float, Ly: float,
                                      Omega: float, t: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(Nx, (int, np.integer)) and not isinstance(Nx, bool) and Nx >= 1):
        raise ValueError("Nx must be an integer >= 1")
    if not (isinstance(Ny, (int, np.integer)) and not isinstance(Ny, bool) and Ny >= 1):
        raise ValueError("Ny must be an integer >= 1")
    if not (isinstance(Lx, (int, float)) and np.isfinite(Lx) and float(Lx) > 0.0):
        raise ValueError("Lx must be a finite number > 0")
    if not (isinstance(Ly, (int, float)) and np.isfinite(Ly) and float(Ly) > 0.0):
        raise ValueError("Ly must be a finite number > 0")
    if not (isinstance(Omega, (int, float)) and np.isfinite(Omega)):
        raise ValueError("Omega must be a finite number")
    if not (isinstance(t, (int, float)) and np.isfinite(t)):
        raise ValueError("t must be a finite number")

    Lx = float(Lx)
    Ly = float(Ly)
    kx_base = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky_base = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)

    delta_y0 = np.mod(1.5 * float(Omega) * Lx * float(t), Ly)

    kx_2d = kx_base[:, None] + (delta_y0 / Lx) * ky_base[None, :]
    ky_2d = np.broadcast_to(ky_base[None, :], (Nx, Ny)).copy()

    return np.stack([kx_2d, ky_2d]).astype(float)

def greens_function_hat(k_perp: np.ndarray, kz: np.ndarray, L: float) -> np.ndarray:
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

def zero_pad_density(rho: np.ndarray, pad_factor: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 3 or min(rho.shape) < 1:
        raise ValueError("rho must be a 3D array of shape (Nx, Ny, Nz)")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho must contain only finite values")
    if not (isinstance(pad_factor, (int, np.integer)) and not isinstance(pad_factor, bool)
            and pad_factor >= 2):
        raise ValueError("pad_factor must be an integer >= 2")

    Nx, Ny, Nz = rho.shape
    rho_padded = np.zeros((Nx, Ny, int(pad_factor) * Nz), dtype=float)
    rho_padded[:, :, :Nz] = rho
    return rho_padded

def spectral_convolution_potential(rho_padded: np.ndarray, greens_hat_grid: np.ndarray,
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

def analytic_layer_potential(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                     kx: float, ky: float, H: float, Lz: float,
                                     G: float = 1.0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import math

    import numpy as np

    # erf is not provided by numpy. The stdlib scalar erf is the platform C
    # library routine, accurate to about one ulp, so vectorising it matches a
    # dedicated special-function library to ~1e-16. A closed-form polynomial
    # approximation would not do: the usual Abramowitz & Stegun 7.1.26 form
    # errs by ~1.4e-07, which is comparable to the ~9e-06 solver error this
    # benchmark measures and would corrupt the reference field.
    _erf_u = np.frompyfunc(math.erf, 1, 1)

    def erf(v):
        return np.asarray(_erf_u(np.asarray(v, dtype=float)), dtype=float)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        raise ValueError("x, y and z must contain only finite values")
    for name, val in (("kx", kx), ("ky", ky)):
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be a finite number")
    if float(kx) == 0.0 and float(ky) == 0.0:
        raise ValueError("kx and ky must not both be zero")
    if not (isinstance(H, (int, float)) and np.isfinite(H) and float(H) > 0.0):
        raise ValueError("H must be a finite number > 0")
    if not (isinstance(Lz, (int, float)) and np.isfinite(Lz) and float(Lz) > 0.0):
        raise ValueError("Lz must be a finite number > 0")
    if not (isinstance(G, (int, float)) and np.isfinite(G) and float(G) > 0.0):
        raise ValueError("G must be a finite number > 0")

    kx = float(kx)
    ky = float(ky)
    H = float(H)
    Lz = float(Lz)

    # This expression is the interior solution: the 2a*cosh(kz) term grows
    # like exp(k|z|) and only cancels against F_+ + F_- for |z| <= Lz/2.
    # Outside the slab the potential decays instead, so evaluating there
    # would return a spuriously growing value rather than a small one.
    if np.any(np.abs(z) > Lz / 2.0 + 1e-12):
        raise ValueError("z must lie inside the source slab, |z| <= Lz/2")

    k = np.hypot(kx, ky)

    prefactor = np.sqrt(np.pi / 2.0) * H / (2.0 * k)
    f_plus = prefactor * np.exp((k * H / 2.0) * (k * H + 2.0 * z / H)) \
        * erf((k * H + z / H) / np.sqrt(2.0))
    f_minus = prefactor * np.exp((k * H / 2.0) * (k * H - 2.0 * z / H)) \
        * erf((k * H - z / H) / np.sqrt(2.0))
    a = -prefactor * np.exp((k * H) ** 2 / 2.0) \
        * erf((k * H + Lz / (2.0 * H)) / np.sqrt(2.0))

    phi = 4.0 * np.pi * float(G) * np.cos(kx * x) * np.cos(ky * y) \
        * (2.0 * a * np.cosh(k * z) + f_plus + f_minus)
    return np.asarray(phi, dtype=float)

def relative_error_rms(phi_num: np.ndarray, phi_ref: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    phi_num = np.asarray(phi_num, dtype=float)
    phi_ref = np.asarray(phi_ref, dtype=float)
    if phi_num.size < 1:
        raise ValueError("phi_num and phi_ref must contain at least one element")
    if phi_num.shape != phi_ref.shape:
        raise ValueError("phi_num and phi_ref must have identical shapes")
    if not np.all(np.isfinite(phi_num)) or not np.all(np.isfinite(phi_ref)):
        raise ValueError("phi_num and phi_ref must contain only finite values")

    num_shifted = phi_num - phi_num.min() + 1.0
    ref_shifted = phi_ref - phi_ref.min() + 1.0
    eps = np.abs((num_shifted - ref_shifted) / ref_shifted)
    return float(np.sqrt(np.mean(eps**2)))

def run_benchmark_pipeline(N: int, L: float, H: float, alpha: float = 1.1,
                                   pad_factor: int = 4, G: float = 1.0,
                                   Omega: float = 1.0, t: float = 0.0,
                                   m_x: int = 1, m_y: int = 1) -> float:
    import numpy as np

    # -- Validate the orchestrator inputs.
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool) and N >= 4):
        raise ValueError("N must be an integer >= 4")
    if not (isinstance(L, (int, float)) and np.isfinite(L) and float(L) > 0.0):
        raise ValueError("L must be a finite number > 0")
    if not (isinstance(H, (int, float)) and np.isfinite(H) and float(H) > 0.0):
        raise ValueError("H must be a finite number > 0")
    if not (isinstance(alpha, (int, float)) and np.isfinite(alpha) and float(alpha) > 1.0):
        raise ValueError("alpha must be a finite number > 1")
    if not (isinstance(pad_factor, (int, np.integer)) and not isinstance(pad_factor, bool)
            and pad_factor >= 2.0 * float(alpha)):
        raise ValueError("pad_factor must be an integer >= 2 * alpha")
    if not (isinstance(G, (int, float)) and np.isfinite(G) and float(G) > 0.0):
        raise ValueError("G must be a finite number > 0")
    if not (isinstance(Omega, (int, float)) and np.isfinite(Omega)):
        raise ValueError("Omega must be a finite number")
    if not (isinstance(t, (int, float)) and np.isfinite(t)):
        raise ValueError("t must be a finite number")
    for name, val in (("m_x", m_x), ("m_y", m_y)):
        if not (isinstance(val, (int, np.integer)) and not isinstance(val, bool)):
            raise ValueError(f"{name} must be an integer")
    if int(m_x) == 0 and int(m_y) == 0:
        raise ValueError("m_x and m_y must not both be zero")

    N = int(N)
    L = float(L)
    H = float(H)
    alpha = float(alpha)
    pad_factor = int(pad_factor)
    G = float(G)
    d = L / N

    # -- Sub-problem 01: cell-centered grid and truncated Gaussian density.
    grid = build_density_grid(N, L, H, m_x=int(m_x), m_y=int(m_y))
    x_3d, y_3d, z_3d, rho = grid[0], grid[1], grid[2], grid[3]
    kx_mode = 2.0 * np.pi * int(m_x) / L
    ky_mode = 2.0 * np.pi * int(m_y) / L

    # -- Sub-problem 02: shear-corrected in-plane wave-vectors.
    k_grids = compute_shear_wavevectors(N, N, L, L, float(Omega), float(t))
    k_perp = np.sqrt(k_grids[0] ** 2 + k_grids[1] ** 2)[:, :, None]

    # -- Sub-problem 03: analytic Fourier-space kernel on the padded kz grid.
    kz = (2.0 * np.pi * np.fft.fftfreq(pad_factor * N, d=d))[None, None, :]
    greens_hat_grid = greens_function_hat(k_perp, kz, alpha * L)

    # -- Sub-problem 04: vertical zero-padding of the density.
    rho_padded = zero_pad_density(rho, pad_factor)

    # -- Sub-problem 05: one-step 3D spectral convolution on physical cells.
    phi_num = spectral_convolution_potential(rho_padded, greens_hat_grid, N, G=G)

    # -- Sub-problem 06: closed-form reference potential.
    phi_ref = analytic_layer_potential(x_3d, y_3d, z_3d, kx_mode, ky_mode, H, L, G=G)

    # -- Sub-problem 07: min-to-1 shifted RMS relative-error norm.
    return relative_error_rms(phi_num, phi_ref)
SCICODE_GOLD_EOF
