"""
Step 04 - Electrostatic work of assembling the reactant ion pair.

Marcus theory is written for a precursor complex, not for two reactants at infinite separation. The measured or tabulated standard free energy refers to separated species, so before it can be used in the activation expression it has to be corrected by the work of bringing the two reactants together, and by the work of separating the two products afterwards.

Here the correction is one-sided. The reactants are a hydroxide anion and a tripositive metal complex, so assembling them releases Coulomb energy; the products are a neutral hydroxyl radical and a dipositive complex, and a neutral partner makes the product work vanish identically. What survives is a single attractive term, screened by whatever dielectric constant actually describes the intervening medium, and it makes the precursor complex more stable than the separated reactants.

The size of that term is not fixed by the reaction, because both the separation and the screening depend on where the donor sits. A donor drawn out into the vapour tail is further from the acceptor, which weakens the attraction, but the path between them is also less polarisable, which strengthens it. The two effects work against each other, so the work term varies slowly and non-monotonically across the interface and cannot be folded into a constant offset.

Returns
-------
numpy.ndarray, electrostatic work of assembling the reactant pair in kcal/mol
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def ion_pair_work(theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Screened Coulomb work of bringing the two reactants into contact.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, strictly below the donor.
    charge_product : float
        Product of the formal charges of the two reactants, in units of the
        elementary charge, signed.

    Returns
    -------
    work : numpy.ndarray
        Electrostatic work of assembling the reactant pair at each dehydration
        level, kcal/mol, same shape as theta. Negative for oppositely charged
        reactants.

    Raises
    ------
    ValueError
        If any argument is not finite, if any element of theta does not lie
        strictly inside (0, 1), or if the donor does not lie strictly above the
        acceptor at every requested dehydration level.
    '''
    return work

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_ion_pair_work(theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Screened Coulomb work of the reactant pair at each dehydration level."""
    _COULOMB_KCAL = 332.0637
    import numpy as np
    th = np.asarray(theta, dtype=float)
    Za = float(Z_acceptor)
    zz = float(charge_product)
    if not (np.isfinite(Za) and np.isfinite(zz)):
        raise ValueError("Z_acceptor and charge_product must be finite")
    eps_eff = _oracle_effective_dielectric(th, Za)
    separation = _depth_from_theta_d(th) - Za
    return zz * _COULOMB_KCAL / (eps_eff * separation)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the hydroxide / hexaamminecobalt(III) pair across the slab ---
        {
            "setup": ("import numpy as np\nZ_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.10, 0.40, 0.73, 0.90])\n"),
            "call": "ion_pair_work(theta, Z_acc, zz)",
            "gold_call": "_oracle_ion_pair_work(theta, Z_acc, zz)",
        },
        # --- Normal: a singly charged acceptor, weaker attraction ---
        {
            "setup": ("import numpy as np\nZ_acc = -6.50\nzz = -1.0\n"
                      "theta = np.array([0.20, 0.55, 0.88])\n"),
            "call": "ion_pair_work(theta, Z_acc, zz)",
            "gold_call": "_oracle_ion_pair_work(theta, Z_acc, zz)",
        },
        # --- Boundary: a neutral acceptor, the work term vanishes identically ---
        {
            "setup": "import numpy as np\nZ_acc = -6.50\nzz = 0.0\ntheta = 0.5\n",
            "call": "ion_pair_work(theta, Z_acc, zz)",
            "gold_call": "_oracle_ion_pair_work(theta, Z_acc, zz)",
        },
        # --- Edge: like-charged reactants far apart, small positive work ---
        {
            "setup": ("import numpy as np\nZ_acc = -12.0\nzz = 2.0\n"
                      "theta = np.array([0.30, 0.95])\n"),
            "call": "ion_pair_work(theta, Z_acc, zz)",
            "gold_call": "_oracle_ion_pair_work(theta, Z_acc, zz)",
        },
    ]
