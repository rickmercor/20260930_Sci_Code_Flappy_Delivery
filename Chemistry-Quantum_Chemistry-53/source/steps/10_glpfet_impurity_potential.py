"""
Run the complete gLPFET workflow, validate the exact reduced-density and static-response FCI reference, and return the converged impurity chemical potential of the requested site.

This final step chains all earlier calculations. It checks the embedding density constraint, exact one-particle density matrix, natural occupations, local double occupations, energy reconstruction, and the symmetry, gauge null mode and negative semidefiniteness of the exact static density response before returning the requested chemical potential.

Returns
-------
float: the self-consistent gLPFET impurity chemical potential of the requested site.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def glpfet_impurity_potential(L, t, v_ext, U, n_elec, site):
    '''Converged gLPFET impurity chemical potential of one site of the ring.

    Chains every earlier step: ring_one_body, solve_correlation_potential,
    then density_mismatch at the converged potential (the density mapping must
    hold to 1e-8), gks_density_matrix, hubbard_fci_reference (including its
    reduced-density energy closure and static-response invariants),
    bath_orbital and impurity_chemical_potential for the requested site, and
    finally cluster_hamiltonian and cluster_site_density to rebuild the site's
    cluster and confirm that its occupation matches the reference density.

    Parameters
    ----------
    L : int
        Number of sites, 3 <= L <= 6.
    t : float
        Nearest-neighbour hopping amplitude.
    v_ext : array_like of float, shape (L,)
        External on-site potential.
    U : float
        On-site repulsion.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2.
    site : int
        Index of the embedded site whose impurity chemical potential is
        requested, 0 <= site < L.

    Returns
    -------
    mu : float
        The gLPFET impurity chemical potential mu_imp of the requested site
        at self-consistency. Raises ValueError for invalid inputs.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_glpfet_impurity_potential(L, t, v_ext, U, n_elec, site):
    import numpy as np

    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer")

    h = _oracle_ring_one_body(L, t, v_ext)

    if site < 0 or site >= h.shape[0]:
        raise ValueError("site index out of range")

    v_c = _oracle_solve_correlation_potential(h, U, n_elec)

    r = _oracle_density_mismatch(h, U, v_c, n_elec)
    if np.linalg.norm(r) > 1e-8:
        raise RuntimeError(
            "the converged potential does not satisfy the density constraints"
        )

    gamma = _oracle_gks_density_matrix(h, U, v_c, n_elec)

    fci = np.asarray(
        _oracle_hubbard_fci_reference(h, U, n_elec),
        dtype=float,
    )
    L_ref = h.shape[0]
    expected_size = 1 + 2 * L_ref * L_ref + L_ref

    if fci.shape != (expected_size,) or not np.all(np.isfinite(fci)):
        raise RuntimeError(
            "the exact FCI benchmark is not finite or has the wrong shape"
        )

    rdm_end = 1 + L_ref * L_ref
    double_end = rdm_end + L_ref

    fci_rdm = fci[1:rdm_end].reshape(L_ref, L_ref)
    fci_double = fci[rdm_end:double_end]
    fci_response = fci[double_end:].reshape(L_ref, L_ref)

    if not np.allclose(
        fci_rdm,
        fci_rdm.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "the exact one-particle density matrix is not symmetric"
        )

    occupations = np.linalg.eigvalsh(fci_rdm)
    if (
        np.min(occupations) < -1e-10
        or np.max(occupations) > 2.0 + 1e-10
    ):
        raise RuntimeError(
            "the exact natural occupations are not physical"
        )

    fci_density = np.diag(fci_rdm)
    density_is_physical = (
        np.all(fci_density >= -1e-10)
        and np.all(fci_density <= 2.0 + 1e-10)
    )
    if (
        not density_is_physical
        or abs(np.sum(fci_density) - n_elec) > 1e-8
    ):
        raise RuntimeError(
            "the exact FCI density is not physically normalized"
        )

    lower_double = np.maximum(0.0, fci_density - 1.0)
    upper_double = 0.5 * fci_density
    if (
        np.any(fci_double < lower_double - 1e-10)
        or np.any(fci_double > upper_double + 1e-10)
    ):
        raise RuntimeError(
            "the exact local double occupations are not physical"
        )

    if not np.allclose(
        fci_response,
        fci_response.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "the exact static density response is not symmetric"
        )

    if np.max(np.abs(np.sum(fci_response, axis=0))) > 1e-8:
        raise RuntimeError(
            "the exact static density response violates its gauge null mode"
        )

    if np.max(np.linalg.eigvalsh(fci_response)) > 1e-9:
        raise RuntimeError(
            "the exact static density response is not negative semidefinite"
        )

    rebuilt_energy = float(
        np.sum(h * fci_rdm.T) + U * np.sum(fci_double)
    )
    if not np.isclose(
        rebuilt_energy,
        fci[0],
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "the exact reduced densities do not reconstruct the FCI energy"
        )

    b = _oracle_bath_orbital(gamma, int(site))
    mu = float(_oracle_impurity_chemical_potential(b, v_c))

    h_cl = _oracle_cluster_hamiltonian(
        h,
        U,
        gamma,
        int(site),
    )
    n_cl = _oracle_cluster_site_density(h_cl, mu)

    if abs(n_cl - 2.0 * gamma[int(site), int(site)]) > 1e-8:
        raise RuntimeError(
            "cluster and reference densities disagree at the requested site"
        )

    return mu

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    cfg = (
        "import numpy as np\n"
        "v = [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0]\n"
    )

    return [
        {
            "setup": cfg,
            "call": (
                "round(glpfet_impurity_potential("
                "6, 1.0, v, 7.0, 6, 2), 5)"
            ),
            "gold_call": (
                "round(_oracle_glpfet_impurity_potential("
                "6, 1.0, v, 7.0, 6, 2), 5)"
            ),
        },
        {
            "setup": cfg,
            "call": (
                "round(glpfet_impurity_potential("
                "6, 1.0, v, 7.0, 6, 3), 5)"
            ),
            "gold_call": (
                "round(_oracle_glpfet_impurity_potential("
                "6, 1.0, v, 7.0, 6, 3), 5)"
            ),
        },
        {
            "setup": cfg,
            "call": (
                "round(glpfet_impurity_potential("
                "6, 1.0, v, 2.0, 6, 0), 5)"
            ),
            "gold_call": (
                "round(_oracle_glpfet_impurity_potential("
                "6, 1.0, v, 2.0, 6, 0), 5)"
            ),
        },
        {
            "setup": cfg,
            "call": (
                "np.round(["
                "glpfet_impurity_potential("
                "6, 1.0, v, 0.0, 6, 2), "
                "glpfet_impurity_potential("
                "6, 1.0, v, 2.0, 6, 5)"
                "], 5)"
            ),
            "gold_call": (
                "np.round(["
                "_oracle_glpfet_impurity_potential("
                "6, 1.0, v, 0.0, 6, 2), "
                "_oracle_glpfet_impurity_potential("
                "6, 1.0, v, 2.0, 6, 5)"
                "], 5)"
            ),
        },
        {
            "setup": cfg,
            "call": (
                "round(glpfet_impurity_potential("
                "4, 1.0, [0.5, -0.5, 1.5, -1.5], "
                "3.0, 2, 1), 5)"
            ),
            "gold_call": (
                "round(_oracle_glpfet_impurity_potential("
                "4, 1.0, [0.5, -0.5, 1.5, -1.5], "
                "3.0, 2, 1), 5)"
            ),
        },
        {
            "setup": cfg + """
def run_model():
    try:
        glpfet_impurity_potential(
            6, 1.0, v, 7.0, 6, 6
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_glpfet_impurity_potential(
            6, 1.0, v, 7.0, 6, 6
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": cfg + """
def run_model():
    try:
        glpfet_impurity_potential(
            2, 1.0, [0.0, 0.0], 7.0, 2, 0
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_glpfet_impurity_potential(
            2, 1.0, [0.0, 0.0], 7.0, 2, 0
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
