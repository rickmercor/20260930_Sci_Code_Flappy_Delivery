"""
Differentiable spectral objective for the three-layer design.

The objective compares the scattering-efficiency spectrum of a layered sphere

with a target spectrum and penalizes absorption. Its derivatives with respect

to the unconstrained design variables follow the chain rule through the

bounded sigmoid parameterization, the cumulative interface radii, the complex

layer indices, every radial recurrence, the exterior coefficient extraction and

the geometric-area normalization of the efficiencies.

Returns
-------
ndarray, shape (10,), float64: The loss L followed by its nine derivatives dL/du_i in input order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_loss_gradient(
    u: "np.ndarray",
) -> "np.ndarray":
    """Return the spectral objective and all nine design derivatives.

    Use three nonmagnetic concentric layers in vacuum with e^(-i omega t)
    time convention. Set p=lo+(hi-lo)/(1+exp(-u)) elementwise, with
    lo=(35,12,15,2.8,1.35,2.1,0,0.015,0.005) and
    hi=(85,48,65,4.2,2.15,3.3,0.07,0.16,0.08).
    The first three entries are core radius and two shell thicknesses in nm;
    radii are their cumulative sums. Entries 3:6 are real refractive indices
    and entries 6:9 are positive extinction coefficients, so m=p[3:6]+i*p[6:9].
    All materials are nondispersive. For lambda_j=430+17*j nm, j=0,...,24, set
    x_l=2*pi*r_l/lambda_j. Use multipoles n=1,...,12 and D1_80=0 as the fixed
    downward seed, differentiating every recurrence with respect to all nine
    u entries. Initialize both composite impedances from the core's D1;
    propagate across each shell using its material at both bounding radii;
    apply the outer coefficient formulas and host radial functions. Define
    Qext, Qsca and Qabs by geometric-area normalization, 2/x_outer**2.
    The target is v_j=0.15+3*exp(-0.5*((lambda_j-620)/70)**2). The loss is
    L=mean((Qsca-v)**2+0.25*Qabs**2). Include the derivatives of the area
    normalization and of the sigmoid, cumulative radii and complex materials.
    This is a fixed finite numerical experiment; no convergence stopping rule
    or adaptive multipole selection changes it.

    Parameters
    ----------
    u : ndarray, shape (9,), float64
        Finite real unconstrained variables with abs(u_i) <= 6.

    Returns
    -------
    ndarray, shape (10,), float64
        The loss L followed by its nine derivatives dL/du_i in input order.

    Raises
    ------
    ValueError
        If u has the wrong shape, nonfinite entries, or abs(u_i) > 6.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _material_design(u):
    lower = np.array([35, 12, 15, 2.8, 1.35, 2.1, 0, 0.015, 0.005])
    upper = np.array([85, 48, 65, 4.2, 2.15, 3.3, 0.07, 0.16, 0.08])
    s = 1 / (1 + np.exp(-u))
    params = lower + (upper - lower) * s
    jac = np.diag((upper - lower) * s * (1 - s))
    radii = np.cumsum(params[:3])
    dr = np.cumsum(jac[:3], axis=0)
    index = params[3:6] + 1j * params[6:9]
    dm = jac[3:6] + 1j * jac[6:9]
    return params, radii, index, dr, dm


def _oracle_spectral_loss_gradient(
    u: "np.ndarray",
) -> "np.ndarray":
    u = np.asarray(u, dtype=float)
    if u.shape != (9,) or not np.all(np.isfinite(u)) or np.any(abs(u) > 6):
        raise ValueError("u must have nine finite entries in [-6, 6]")
    nmax, depth = 12, 80
    _, radii, index, dr, dm = _material_design(np.asarray(u))
    wavelengths = 430 + 17 * np.arange(25)
    k = 2 * np.pi / wavelengths
    x, dx = radii[:, None] * k, dr[:, None, :] * k[None, :, None]
    ms = index[:, None] * np.ones((3, len(k)))
    dms = dm[:, None, :] * np.ones((3, len(k), 9))
    z = ms[0] * x[0]
    dz = dms[0] * x[0, :, None] + ms[0, :, None] * dx[0]
    core = _oracle_logarithmic_wave_jets(z, dz, nmax, depth)
    composite = np.stack((core[:, 0], core[:, 0]), axis=1)
    for layer in range(1, 3):
        z1 = ms[layer] * x[layer - 1]
        z2 = ms[layer] * x[layer]
        dz1 = (
            dms[layer] * x[layer - 1, :, None]
            + ms[layer, :, None] * dx[layer - 1]
        )
        dz2 = dms[layer] * x[layer, :, None] + ms[layer, :, None] * dx[layer]
        inner = _oracle_logarithmic_wave_jets(z1, dz1, nmax, depth)
        outer = _oracle_logarithmic_wave_jets(z2, dz2, nmax, depth)
        q = _oracle_shell_ratio_jets(z1, dz1, z2, dz2, inner, outer)
        composite = _oracle_composite_impedance_jets(
            composite,
            ms[layer - 1],
            dms[layer - 1],
            ms[layer],
            dms[layer],
            inner,
            outer,
            q,
        )
    host = _oracle_logarithmic_wave_jets(x[-1], dx[-1], nmax, depth)
    radial = _oracle_host_radial_jets(x[-1], dx[-1], host)
    coeff = _oracle_layered_coefficient_jets(
        x[-1], dx[-1], ms[-1], dms[-1], composite, radial
    )
    spectra = _oracle_mie_efficiency_jets(x[-1], dx[-1], coeff).transpose(
        1, 0, 2
    )
    target = 0.15 + 3 * np.exp(-0.5 * ((wavelengths - 620) / 70) ** 2)
    residual = spectra[:, 1, 0] - target
    loss = np.mean(residual**2 + 0.25 * spectra[:, 2, 0] ** 2)
    grad = np.mean(
        2 * residual[:, None] * spectra[:, 1, 1:]
        + 0.5 * spectra[:, 2, :1] * spectra[:, 2, 1:],
        axis=0,
    )
    return np.r_[loss, grad]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "u = np.array([0.15, -0.4, 0.35, -0.25, 0.3, -0.1, -0"
                ".8, 0.2, -0.5])\n"
            ),
            "call": ("spectral_loss_gradient(u.copy())\n"),
            "gold_call": ("_oracle_spectral_loss_gradient(u.copy())\n"),
            "tol": 1e-09,
        },
        {
            "setup": ("import numpy as np\n" "\n" "u = np.zeros(9)\n"),
            "call": ("spectral_loss_gradient(u.copy())\n"),
            "gold_call": ("_oracle_spectral_loss_gradient(u.copy())\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n" "\n" "u = np.linspace(-1.0, 1.0, 9)\n"
            ),
            "call": ("spectral_loss_gradient(u.copy())\n"),
            "gold_call": ("_oracle_spectral_loss_gradient(u.copy())\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def raises(fn):\n"
                "    try:\n"
                "        fn(np.zeros(8))\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(spectral_loss_gradient)\n"),
            "gold_call": ("raises(_oracle_spectral_loss_gradient)\n"),
            "tol": 0.0,
        },
    ]
