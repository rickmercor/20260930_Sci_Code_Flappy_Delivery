"""
Step 12 - Orchestrator: population-averaged interfacial rate constant.

The observable is not the rate in any single hydration environment but the average over the whole interfacial population. Assembling it means running the dehydration coordinate across the slab, weighting the rate at each level by the probability density of finding a donor there, and integrating.

Every piece is already in place. The slab faces fix the dehydration interval, the population density supplies the weight, and the rate at fixed dehydration level supplies the integrand. The integral is one dimensional and smooth but strongly structured: the weight diverges towards both ends of the dehydration range while the rate is sharply peaked where the two states come into resonance, so a rule that is uniform in dehydration level wastes its nodes. Placing the panel edges at equal steps in depth rather than in dehydration level removes the Jacobian's end-point growth from each panel, and a modest Gauss-Legendre rule on each panel then converges to many more digits than the answer needs. The result must be accurate to a relative 1e-10.

The number has a clear physical reading. It is the pseudo-first-order rate constant a donor-acceptor pair experiences at the interface, and its size relative to the bulk value is the whole content of the claim that the air-water interface, rather than any applied field or exotic intermediate, is what makes microdroplet redox chemistry go.

Returns
-------
float, the population-averaged interfacial electron transfer rate constant in s^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def interfacial_rate_constant(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, Z_lo: float, Z_hi: float, Z_acceptor: float, charge_product: float) -> float:
    '''Population-averaged interfacial electron transfer rate constant.

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
    Z_lo : float
        Depth of the liquid-side face of the interfacial slab, angstrom.
    Z_hi : float
        Depth of the vapour-side face of the interfacial slab, angstrom,
        strictly greater than Z_lo.
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, strictly below the liquid-side face of the slab.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    rate_constant : float
        Population-averaged electron transfer rate constant in inverse seconds,
        as a native Python float, accurate to a relative 1e-10 or better.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if lam_acceptor_outer is not
        positive, if breathing is not a valid breathing-mode description, if
        Z_hi is not strictly greater than Z_lo, or if Z_acceptor is not strictly
        below Z_lo.
    '''
    return rate_constant

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_interfacial_rate_constant(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, Z_lo: float, Z_hi: float, Z_acceptor: float, charge_product: float) -> float:
    """Average the dehydration-resolved rate over the interfacial population."""
    _ZG_O = 0.84
    _DEL_O = 1.5
    _PANELS = 48
    _NODES = 48
    import numpy as np
    E, lam_a = float(E0_acceptor), float(lam_acceptor_outer)
    zlo, zhi, za = float(Z_lo), float(Z_hi), float(Z_acceptor)
    zz = float(charge_product)
    if not all(np.isfinite(v) for v in (E, lam_a, zlo, zhi, za, zz)):
        raise ValueError("all scalar arguments must be finite")
    if lam_a <= 0.0:
        raise ValueError("lam_acceptor_outer must be positive")
    if not zhi > zlo:
        raise ValueError("Z_hi must be strictly greater than Z_lo")
    if not za < zlo:
        raise ValueError("Z_acceptor must lie strictly below Z_lo")
    _oracle_breathing_mode_surfaces(breathing)

    limits = _oracle_slab_dehydration_limits(zlo, zhi, 0.5)
    theta_lo, theta_hi = float(limits[0]), float(limits[1])
    z_edges = np.linspace(zlo, zhi, _PANELS + 1)
    t_edges = 0.5 * (1.0 + np.tanh((z_edges - _ZG_O) / _DEL_O))
    t_edges[0], t_edges[-1] = theta_lo, theta_hi

    x, w = np.polynomial.legendre.leggauss(_NODES)
    total = 0.0
    for a, b in zip(t_edges[:-1], t_edges[1:]):
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        weight = _oracle_dehydration_density(t, zlo, zhi)
        rate = _oracle_rate_density(E, lam_a, breathing, t, za, zz)
        total += float(np.dot(w, weight * rate)) * 0.5 * (b - a)
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the target system, hexaamminecobalt(III) on a six-angstrom slab ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_lo = -3.0\nZ_hi = 3.0\nZ_acc = -6.50\nzz = -3.0\n"),
            "call": "interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "tol": 0.001,
        },
        # --- Normal: hexaaquairon(III) acceptor on the same slab ---
        {
            "setup": ("import numpy as np\nE0 = 0.77\nlam_out = 25.0\n"
                      "br = np.array([490.0, 389.0, 1.990, 2.128, 18.015])\n"
                      "Z_lo = -3.0\nZ_hi = 3.0\nZ_acc = -6.50\nzz = -3.0\n"),
            "call": "interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "tol": 0.01,
        },
        # --- Boundary: a thicker slab reaching further into both phases ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n"
                      "Z_lo = -4.0\nZ_hi = 4.0\nZ_acc = -7.50\nzz = -3.0\n"),
            "call": "interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "tol": 0.0001,
        },
        # --- Edge: equal breathing wavenumbers and a neutral pair, no work term ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nlam_out = 27.6\n"
                      "br = np.array([420.0, 420.0, 1.936, 2.114, 17.031])\n"
                      "Z_lo = -3.0\nZ_hi = 3.0\nZ_acc = -6.50\nzz = 0.0\n"),
            "call": "interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_rate_constant(E0, lam_out, br, Z_lo, Z_hi, Z_acc, zz)",
            "tol": 0.001,
        },
    ]
