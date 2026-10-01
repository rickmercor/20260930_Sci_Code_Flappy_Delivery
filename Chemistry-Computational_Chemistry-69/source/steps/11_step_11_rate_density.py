"""
Step 11 - Golden-rule rate constant at a fixed dehydration level.

With the nuclear factor and the coupling both resolved in dehydration level, the rate at a fixed level follows from Fermi's golden rule: two pi over hbar, times the squared electronic coupling, times the density per unit energy of configurations in which reactant and product are degenerate. Nothing else enters, because the transfer is non-adiabatic and the nuclei are classical.

The only care needed is with units. The coupling is squared and expressed in joules squared, while the nuclear factor from the previous step is per kcal/mol, so it has to be converted to per joule before the product is formed. A slip there leaves the shape of the rate against dehydration level untouched and moves its magnitude by a factor of about 1e20, which is easy to miss because no single number in the problem looks obviously wrong afterwards.

The rate at a fixed dehydration level is not yet observable, because no experiment prepares a single hydration environment. It is the integrand of the observable, and its shape is what makes the interface special: a sharp maximum where the donor's loss of hydration has brought the two states into resonance, cut off on the dry side by the loss of electronic coupling and on the wet side by the return of a barrier.

Returns
-------
numpy.ndarray, electron transfer rate constant in s^-1 at each dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def rate_density(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Non-adiabatic electron transfer rate constant at fixed dehydration level.

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
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, on the liquid side of the donor.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    rate : numpy.ndarray
        Electron transfer rate constant in inverse seconds for a donor held at
        each dehydration level, same shape as theta, accurate to a relative
        1e-10 or better.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if lam_acceptor_outer is not
        positive, if breathing is not a valid breathing-mode description, or if
        any element of theta does not lie strictly inside (0, 1).
    '''
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_rate_density(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Golden rule: 2 pi / hbar times squared coupling times nuclear factor per joule."""
    _HBAR_JS = 1.054571817e-34
    _J_PER_KCAL_R = 6.9476954571e-21
    import numpy as np
    th = np.asarray(theta, dtype=float)
    fc_per_kcal = _oracle_nuclear_factor(E0_acceptor, lam_acceptor_outer, breathing, th,
                                         Z_acceptor, charge_product)
    coupling_sq = _oracle_tunnelling_coupling_squared(th, Z_acceptor)
    return (2.0 * np.pi / _HBAR_JS) * coupling_sq * (fc_per_kcal / _J_PER_KCAL_R)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target system sampled through the reactive region ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.60, 0.72, 0.78, 0.85, 0.92])\n"),
            "call": "rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "tol": 0.01,
        },
        # --- Normal: hexaaquairon(III) acceptor with a deeper acceptor plane ---
        {
            "setup": ("import numpy as np\nE0 = 0.77\nlam_out = 25.0\n"
                      "br = np.array([490.0, 389.0, 1.990, 2.128, 18.015])\n"
                      "Z_acc = -8.0\nzz = -3.0\n"
                      "theta = np.array([0.35, 0.50, 0.65])\n"),
            "call": "rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "tol": 0.01,
        },
        # --- Boundary: neutral acceptor, work term switched off ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_acc = -6.50\nzz = 0.0\ntheta = 0.80\n"),
            "call": "rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "tol": 0.01,
        },
        # --- Edge: nearly dry donor, coupling almost extinguished ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.96, 0.985])\n"),
            "call": "rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "gold_call": "_oracle_rate_density(E0, lam_out, br, theta, Z_acc, zz)",
            "tol": 1e-07,
        },
    ]
