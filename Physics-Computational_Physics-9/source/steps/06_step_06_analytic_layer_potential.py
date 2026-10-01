"""
Evaluate the closed-form error-function reference potential of a truncated Gaussian single-mode layer with vacuum vertical boundaries.

The solver is benchmarked against a closed-form solution for a single in-plane mode with a truncated Gaussian vertical profile, rho = cos(kx*x) * cos(ky*y) * exp(-z^2 / (2*H^2)) on z in [-Lz/2, Lz/2] and vacuum outside. For this separable source the screened equation (d^2/dz^2 - k^2) Phi_tilde = 4*pi*G*rho_tilde, k = sqrt(kx^2 + ky^2) > 0, is solved by convolving with -exp(-k|z - z'|)/(2k). Completing the square in the Gaussian-times-exponential integrals yields error functions:

    Phi = 4*pi*G * cos(kx*x) * cos(ky*y) * [ 2*a*cosh(k*z) + F_plus + F_minus ],

    F_pm(z) = sqrt(pi/2) * (H / (2k)) * exp((k*H/2) * (k*H pm 2*z/H))

              * erf((k*H pm z/H) / sqrt(2)),

    a = -sqrt(pi/2) * (H / (2k)) * exp((k*H)^2 / 2)

        * erf((k*H + Lz/(2*H)) / sqrt(2)),

which is continuous with a continuous derivative at the slab boundaries and vanishes as |z| -> infinity, i.e. it satisfies the vacuum boundary conditions the spectral kernel is designed to reproduce.

Returns
-------
np.ndarray, float, with the broadcast shape of (x, y, z): the analytic potential values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np

def analytic_layer_potential(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             kx: float, ky: float, H: float, Lz: float,
                             G: float = 1.0) -> np.ndarray:
    """Closed-form potential of a truncated Gaussian single-mode layer.

    Parameters
    ----------
    x : np.ndarray
        x coordinates, broadcastable against y and z.
    y : np.ndarray
        y coordinates, broadcastable against x and z.
    z : np.ndarray
        Vertical coordinates, broadcastable against x and y. Every value
        must lie inside the source slab, |z| <= Lz/2; this closed form is
        the interior solution and is not valid above or below the slab,
        where the potential instead decays exponentially.
    kx : float
        In-plane wavenumber of the density mode in x (finite).
    ky : float
        In-plane wavenumber of the density mode in y (finite; kx and ky
        must not both be zero).
    H : float
        Gaussian scale height of the vertical density profile (H > 0).
    Lz : float
        Vertical extent of the slab hosting the density (Lz > 0).
    G : float
        Gravitational constant (G > 0).

    Returns
    -------
    phi : np.ndarray
        The analytic potential evaluated elementwise on the broadcast shape
        of (x, y, z), as a float array.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_analytic_layer_potential(x: np.ndarray, y: np.ndarray, z: np.ndarray,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: cell-centered 3D benchmark grid (normal scenario) ---
        {
            "setup": """import numpy as np
N = 16
L = 6.0
d = L / N
coords = -L / 2.0 + (np.arange(N) + 0.5) * d
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
kx = ky = 2.0 * np.pi / L
H = 1.0
""",
            "call": "analytic_layer_potential(X, Y, Z, kx, ky, H, L)",
            "gold_call": "_oracle_analytic_layer_potential(X, Y, Z, kx, ky, H, L)",
        },
        # --- Valid: thick layer with H comparable to the box (H = 3) ---
        {
            "setup": """import numpy as np
z = np.linspace(-3.0, 3.0, 41)
kx = ky = 2.0 * np.pi / 6.0
H = 3.0
""",
            "call": "analytic_layer_potential(0.0, 0.0, z, kx, ky, H, 6.0, G=1.0)",
            "gold_call": "_oracle_analytic_layer_potential(0.0, 0.0, z, kx, ky, H, 6.0, G=1.0)",
        },
        # --- Boundary: evaluation exactly at the slab edges z = +-Lz/2 ---
        {
            "setup": """import numpy as np
z = np.array([-3.0, 0.0, 3.0])
kx = 2.0 * np.pi / 6.0
ky = 0.0
H = 1.0
""",
            "call": "analytic_layer_potential(1.0, 2.0, z, kx, ky, H, 6.0)",
            "gold_call": "_oracle_analytic_layer_potential(1.0, 2.0, z, kx, ky, H, 6.0)",
        },
        # --- Pinned value: midplane potential of the benchmark mode ---
        # Expected value derived independently from the closed form, not from
        # the oracle: 4*pi*[2a + 2F(0)] with k = sqrt(2)*2*pi/6, H = 1, Lz = 6.
        {
            "setup": """import numpy as np
kx = ky = 2.0 * np.pi / 6.0
EXPECTED = -4.4134327
""",
            "call": "float(np.round(analytic_layer_potential(0.0, 0.0, 0.0, kx, ky, 1.0, 6.0), 7))",
            "gold_call": "EXPECTED",
        },
    ]
