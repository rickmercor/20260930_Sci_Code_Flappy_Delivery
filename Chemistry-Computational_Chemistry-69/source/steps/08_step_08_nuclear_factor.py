"""
Step 08 - Exact classical nuclear factor with unequal metal-ligand curvatures.

The golden-rule rate needs the thermal probability density, per unit energy, of finding the reactants in a configuration where the reactant and product states are degenerate. Two sets of coordinates contribute. The equal-curvature coordinates of the previous steps produce an energy gap that is linear in their collective displacement, so averaging over them turns the resonance condition into a normalised Gaussian in whatever part of the gap they have to supply, centred one symmetric reorganisation energy away.

The metal-ligand breathing coordinate does not behave that way. With different force constants in the two oxidation states, the part of the gap it contributes is quadratic in the displacement, and the reactant's thermal distribution in that coordinate is the Boltzmann distribution of the oxidised complex, a Gaussian whose width is set by the oxidised force constant and the temperature. The nuclear factor is therefore the average of the solvent Gaussian over that breathing distribution, with the breathing contribution shifting the resonance configuration by configuration. It has no closed form and must be integrated; because the breathing distribution is Gaussian, a Gauss-Hermite rule converges very quickly.

Replacing this average by a single Gaussian in the total gap requires a single effective curvature for the breathing mode. That is exact for equal force constants and not otherwise, and for a cobalt ammine the error it introduces in where the resonance sits is several kcal/mol.

Returns
-------
numpy.ndarray, nuclear factor in (kcal/mol)^-1 at each dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def nuclear_factor(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Classical Franck-Condon weighted density of states at each dehydration level.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor, V vs SHE.
    lam_acceptor_outer : float
        Outer-sphere self-exchange reorganisation energy of the acceptor couple,
        kcal/mol, positive.
    breathing : array_like
        Length-5 sequence [nu_ox, nu_red, d_ox, d_red, m_ligand] describing the
        acceptor's octahedral breathing mode, as for breathing_mode_surfaces.
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal, angstrom.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    fc : numpy.ndarray
        Thermal average, over the reactant state, of the delta function of the
        product-minus-reactant energy gap, in (kcal/mol)^-1, at 298.15 K and
        with all nuclear motion classical, same shape as theta. The value must
        be accurate to a relative 1e-10 or better.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if lam_acceptor_outer is not
        positive, if breathing is not a valid breathing-mode description, or if
        any element of theta does not lie strictly inside (0, 1).
    '''
    return fc

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_nuclear_factor(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Solvent Gaussian averaged over the oxidised complex's breathing distribution."""
    _R_KCAL_N = 0.001987204259
    _T_N = 298.15
    _J_PER_KCAL_N = 6.9476954571e-21
    _DG_SOL_ANION_N = -106.4
    _DG_SOL_RADICAL_N = -3.9
    _GH_NODES = 128
    import numpy as np
    E, lam_a = float(E0_acceptor), float(lam_acceptor_outer)
    Za, zz = float(Z_acceptor), float(charge_product)
    if not all(np.isfinite(v) for v in (E, lam_a, Za, zz)):
        raise ValueError("scalar arguments must be finite")
    if lam_a <= 0.0:
        raise ValueError("lam_acceptor_outer must be positive")
    th = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    surf = _oracle_breathing_mode_surfaces(breathing)
    f_ox, f_red = float(surf[0]), float(surf[1])
    b = np.asarray(breathing, dtype=float).ravel()
    shift = (b[3] - b[2]) * 1e-10
    kT = _R_KCAL_N * _T_N
    kT_J = kT * _J_PER_KCAL_N
    lam_s = np.atleast_1d(_oracle_solvent_reorganisation_energy(lam_a, th))
    dG = np.atleast_1d(_oracle_interfacial_driving_force(E, _DG_SOL_ANION_N,
                                                         _DG_SOL_RADICAL_N, th, Za, zz))
    sigma = np.sqrt(kT_J / (6.0 * f_ox))
    x, w = np.polynomial.hermite.hermgauss(_GH_NODES)
    q = np.sqrt(2.0) * sigma * x
    gap_breath = (3.0 * f_red * (q - shift) ** 2 - 3.0 * f_ox * q ** 2) / _J_PER_KCAL_N
    arg = lam_s[:, None] + dG[:, None] + gap_breath[None, :]
    avg = (np.exp(-arg ** 2 / (4.0 * lam_s[:, None] * kT)) * w[None, :]).sum(axis=1) / np.sqrt(np.pi)
    out = avg / np.sqrt(4.0 * np.pi * lam_s * kT)
    return out.reshape(th.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target system sampled through the resonance ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.50, 0.70, 0.78, 0.86, 0.93])\n"),
            "call": "nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
            "tol": 1e-08,
        },
        # --- Normal: hexaaquairon(III) acceptor, different resonance position ---
        {
            "setup": ("import numpy as np\nE0 = 0.77\nlam_out = 25.0\n"
                      "br = np.array([490.0, 389.0, 1.990, 2.128, 18.015])\n"
                      "Z_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.30, 0.45, 0.60, 0.75])\n"),
            "call": "nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
        },
        # --- Boundary: equal wavenumbers, reduces to a single Gaussian ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([420.0, 420.0, 1.936, 2.114, 17.031])\n"
                      "Z_acc = -6.50\nzz = -3.0\ntheta = np.array([0.72, 0.80])\n"),
            "call": "nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
        },
        # --- Edge: bulk-like donor far below resonance, factor vanishingly small ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_acc = -6.50\nzz = -3.0\ntheta = 0.05\n"),
            "call": "nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_nuclear_factor(E0, lam_out, br, theta, Z_acc, zz)",
        },
    ]
