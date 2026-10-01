"""
Step 06 - Symmetric part of the cross-reaction reorganisation energy.

Most of the reorganisation in this reaction is carried by coordinates whose free-energy surfaces have the same curvature before and after the electron moves: the polarisation of the water around both partners and the small internal distortion of the donor. For those coordinates the Marcus cross-relation applies, and the cross-reaction value is the arithmetic mean of the two self-exchange values that feed it.

The donor contributes its whole self-exchange reorganisation here. That quantity splits into a small internal part, which survives any amount of dehydration, and a large outer-sphere part that is the response of the surrounding water and is removed in proportion to the fraction of the hydration shell that has gone. The acceptor contributes only its outer-sphere self-exchange reorganisation, which does not depend on the donor's environment because the acceptor stays fully solvated in the liquid.

What is deliberately left out is the acceptor's metal-ligand reorganisation. For a metal complex whose bond lengths change on reduction, the two oxidation states generally have different metal-ligand force constants, so that coordinate does not have equal curvatures and cannot be folded into a mean of self-exchange values without an approximation. It is handled explicitly in the steps that follow, so the quantity returned here is only the part that is exact under the cross-relation.

Returns
-------
numpy.ndarray, symmetric reorganisation energy in kcal/mol at each dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def solvent_reorganisation_energy(lam_acceptor_outer: float, theta: npt.ArrayLike) -> np.ndarray:
    '''Symmetric (equal-curvature) part of the cross-reaction reorganisation energy.

    Parameters
    ----------
    lam_acceptor_outer : float
        Outer-sphere self-exchange reorganisation energy of the acceptor couple,
        kcal/mol, positive and independent of the donor's hydration.
    theta : array_like
        Dehydration level or levels of the donor, each in [0, 1].

    Returns
    -------
    lam_sym : numpy.ndarray
        Cross-relation reorganisation energy of the equal-curvature coordinates
        (donor self-exchange and acceptor outer sphere) in kcal/mol at each
        dehydration level, same shape as theta.

    Raises
    ------
    ValueError
        If lam_acceptor_outer is not finite or not positive, if theta is not
        finite, or if any element of theta lies outside the closed interval
        [0, 1].
    '''
    return lam_sym

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_solvent_reorganisation_energy(lam_acceptor_outer: float, theta: npt.ArrayLike) -> np.ndarray:
    """Cross-relation mean of the equal-curvature self-exchange contributions."""
    _LAM_DONOR_INNER = 2.0
    _LAM_DONOR_OUTER = 67.0
    import numpy as np
    lam_a = float(lam_acceptor_outer)
    th = np.asarray(theta, dtype=float)
    if not np.isfinite(lam_a) or lam_a <= 0.0:
        raise ValueError("lam_acceptor_outer must be finite and positive")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th < 0.0) or np.any(th > 1.0):
        raise ValueError("theta must lie in [0, 1]")
    lam_donor = _LAM_DONOR_INNER + (1.0 - th) * _LAM_DONOR_OUTER
    return 0.5 * (lam_a + lam_donor)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target acceptor outer sphere across the whole dehydration range ---
        {
            "setup": ("import numpy as np\nlam_out = 27.6\n"
                      "theta = np.array([0.0, 0.2, 0.5, 0.8, 1.0])\n"),
            "call": "solvent_reorganisation_energy(lam_out, theta)",
            "gold_call": "_oracle_solvent_reorganisation_energy(lam_out, theta)",
        },
        # --- Normal: a compact acceptor with a small outer sphere ---
        {
            "setup": ("import numpy as np\nlam_out = 12.0\n"
                      "theta = np.array([0.1, 0.6, 0.94])\n"),
            "call": "solvent_reorganisation_energy(lam_out, theta)",
            "gold_call": "_oracle_solvent_reorganisation_energy(lam_out, theta)",
        },
        # --- Boundary: the two limits of the dehydration range ---
        {
            "setup": "import numpy as np\nlam_out = 30.0\ntheta = np.array([0.0, 1.0])\n",
            "call": "solvent_reorganisation_energy(lam_out, theta)",
            "gold_call": "_oracle_solvent_reorganisation_energy(lam_out, theta)",
        },
        # --- Edge: scalar dehydration level with a very large outer sphere ---
        {
            "setup": "import numpy as np\nlam_out = 95.0\ntheta = 0.3333333333333333\n",
            "call": "solvent_reorganisation_energy(lam_out, theta)",
            "gold_call": "_oracle_solvent_reorganisation_energy(lam_out, theta)",
        },
    ]
