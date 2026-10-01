"""
Step 05 - Driving force of the concerted transfer inside the precursor complex.

The driving force is where dehydration does its work. Hydroxide is stabilised by water enormously more than the neutral radical it becomes, so removing part of that hydration shell destabilises the reactant far more than the product and pushes the reaction downhill by an amount comparable to a chemical bond. Taking every solvent-dependent free energy to vary linearly between the bulk and gas-like limits makes the standard reaction free energy a straight line in the dehydration level, anchored at the bulk value and falling with a slope fixed by the difference in solvation between the two donor states.

That line describes separated reactants. The quantity the activation expression needs is the free energy change inside the precursor complex, which differs from it by the work terms. With a neutral product the correction reduces to subtracting the assembly work of the reactant pair, and because that work is attractive the correction is uphill: the precursor complex starts lower, so less of the reaction free energy is available.

The competition between the two contributions is what makes the interface interesting. A donor drawn towards the vapour gains driving force linearly from dehydration and gives some of it back through a work term that varies with both distance and screening, so the corrected driving force is not linear in the dehydration level even though the standard part is.

Returns
-------
numpy.ndarray, work-corrected reaction free energy in kcal/mol at each dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def interfacial_driving_force(E0_acceptor: float, dG_sol_anion: float, dG_sol_radical: float, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Work-corrected reaction free energy versus dehydration level.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor, V vs SHE.
    dG_sol_anion : float
        Absolute solvation free energy of the donor anion, kcal/mol, signed.
    dG_sol_radical : float
        Absolute solvation free energy of the neutral radical, kcal/mol, signed.
    theta : array_like
        Dehydration level or levels, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal, angstrom.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    dG : numpy.ndarray
        Reaction free energy of the concerted electron transfer inside the
        precursor complex at each dehydration level, kcal/mol, same shape as
        theta.

    Raises
    ------
    ValueError
        If any argument is not finite, if any element of theta does not lie
        strictly inside (0, 1), or if the donor does not lie strictly above the
        acceptor at every requested dehydration level.
    '''
    return dG

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_interfacial_driving_force(E0_acceptor: float, dG_sol_anion: float, dG_sol_radical: float, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Standard driving force made linear in dehydration, then work-corrected."""
    import numpy as np
    E = float(E0_acceptor)
    ga = float(dG_sol_anion)
    gr = float(dG_sol_radical)
    th = np.asarray(theta, dtype=float)
    if not (np.isfinite(E) and np.isfinite(ga) and np.isfinite(gr)):
        raise ValueError("scalar arguments must be finite")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    dG_bulk = float(_oracle_acceptor_energetics(E)[1])
    ddG_sol = gr - ga
    dG_separated = dG_bulk - th * ddG_sol
    work_r = _oracle_ion_pair_work(th, Z_acceptor, charge_product)
    return dG_separated - work_r

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target acceptor scanned across the interface ---
        {
            "setup": ("import numpy as np\nE0 = 0.108\nga = -106.4\ngr = -3.9\n"
                      "Z_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.05, 0.25, 0.5, 0.75, 0.94])\n"),
            "call": "interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
        },
        # --- Normal: acceptor already spontaneous in bulk water ---
        {
            "setup": ("import numpy as np\nE0 = 2.10\nga = -106.4\ngr = -3.9\n"
                      "Z_acc = -6.50\nzz = -3.0\n"
                      "theta = np.array([0.1, 0.4, 0.9])\n"),
            "call": "interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
        },
        # --- Boundary: neutral acceptor, work term switched off entirely ---
        {
            "setup": ("import numpy as np\nE0 = -0.449\nga = -106.4\ngr = -3.9\n"
                      "Z_acc = -6.50\nzz = 0.0\n"
                      "theta = np.array([0.02, 0.98])\n"),
            "call": "interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
        },
        # --- Edge: a weakly solvated donor with a shallow dehydration slope ---
        {
            "setup": ("import numpy as np\nE0 = 1.03\nga = -72.0\ngr = -1.5\n"
                      "Z_acc = -9.0\nzz = -2.0\ntheta = 0.6666666666666666\n"),
            "call": "interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
            "gold_call": "_oracle_interfacial_driving_force(E0, ga, gr, theta, Z_acc, zz)",
        },
    ]
