"""
Compute the fraction of the Hubbard-ring correlation energy recovered by orbital-optimized seniority eigenstate configuration interaction at a chosen local seniority.

The correlation energy is measured from a closed-shell mean-field reference, the restricted determinant built from the lowest one-electron orbitals of the ring, whose energy for the Hubbard interaction is $E_{\mathrm{ref}}=2\sum_{i\in\mathrm{occ}}\varepsilon_i+U\sum_j\rho_j^2$ with $\rho_j$ the per-spin site density. Comparing the orbital-optimized SECI energy with the exact full configuration interaction energy through $f=(E_{\mathrm{ref}}-E_{\mathrm{SECI}})/(E_{\mathrm{ref}}-E_{\mathrm{FCI}})$ shows how much strong correlation a single fixed-local-seniority sector captures when the charge of the pairing levels is coupled to the spin of the singly occupied levels.

Returns
-------
float: recovered correlation fraction f
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seci_correlation_fraction(n_sites: int, hopping_t: float, onsite_U: float, n_electrons: int, n_spin_levels: int, seed: int = 20260927) -> float:
    '''Return the correlation-energy fraction recovered by orbital-optimized SECI.
 
    System: periodic Hubbard ring of M = n_sites sites with hopping[i, i+1 mod M] =
    hopping[i+1 mod M, i] = -hopping_t (all other entries zero), on-site repulsion U, and
    n_electrons electrons with S^z = 0 (n_up = n_down = n_electrons / 2).
      * E_ref: energy of the closed-shell restricted determinant whose up and down
        electrons both occupy the n_electrons/2 lowest eigenvectors phi_i of the hopping
        matrix, E_ref = 2 sum_i eps_i + U sum_j rho_j^2 with rho_j = sum_i phi_i(j)^2. The
        shell must be closed (gap between the highest occupied and lowest unoccupied
        one-electron energies larger than 1e-9).
      * E_SECI: global minimum over orbital rotations C_up = expm(X), C_dn = expm(X)
        expm(Y) of the lowest eigenvalue of the local-seniority-conserving Hamiltonian in
        the sector with k = n_spin_levels spin levels (k/2 up, k/2 down; Y confined to
        their block) and (n_electrons - k)/2 pairs on the M - k pairing levels.
        E_SECI is the value returned by optimize_seci_orbitals(hopping, U,
        (n_electrons - k)/2, k, seed).
      * E_FCI: exact ground-state energy with n_up = n_down = n_electrons / 2.
    Return f = (E_ref - E_SECI) / (E_ref - E_FCI), accurate to 1e-7.
 
    Parameters
    ----------
    n_sites : int
        Number of ring sites M >= 3.
    hopping_t : float
        Hopping amplitude t > 0.
    onsite_U : float
        On-site repulsion U > 0.
    n_electrons : int
        Even electron number with 0 < n_electrons < 2 M. A completely filled ring
        (n_electrons = 2 M) is excluded: it has a single determinant, so
        E_ref = E_SECI = E_FCI and f = 0/0 is undefined.
    n_spin_levels : int
        Even k with 0 <= k <= M and 0 <= (n_electrons - k)/2 <= M - k.
    seed : int, optional
        Seed forwarded to optimize_seci_orbitals for its random starting points
        (default 20260927); f must not depend on it.
 
    Returns
    -------
    fraction : float
        Recovered fraction of the correlation energy.
 
    Raises
    ------
    ValueError
        If the inputs violate the conditions above (including a completely filled
        ring) or the reference shell is open.
    '''
    return fraction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_seci_correlation_fraction(n_sites: int, hopping_t: float, onsite_U: float, n_electrons: int, n_spin_levels: int, seed: int = 20260927) -> float:
    m, ne, k = int(n_sites), int(n_electrons), int(n_spin_levels)
    if m < 3 or not hopping_t > 0 or not onsite_U > 0 or ne % 2 or not 0 < ne < 2 * m:
        raise ValueError("invalid ring or electron number")
    if k % 2 or not 0 <= k <= m or (ne - k) % 2 or not 0 <= (ne - k) // 2 <= m - k:
        raise ValueError("invalid seniority sector")
    hopping = np.zeros((m, m))
    for i in range(m):
        hopping[i, (i + 1) % m] = hopping[(i + 1) % m, i] = -float(hopping_t)
    n_occ = ne // 2
    levels, vectors = np.linalg.eigh(hopping)
    if n_occ < m and levels[n_occ] - levels[n_occ - 1] <= 1e-9:
        raise ValueError("open-shell reference determinant")
    density = np.sum(vectors[:, :n_occ] ** 2, axis=1)
    e_ref = 2.0 * np.sum(levels[:n_occ]) + onsite_U * np.sum(density ** 2)
    e_seci = _oracle_optimize_seci_orbitals(hopping, onsite_U, (ne - k) // 2, k, seed)
    e_fci = _oracle_hubbard_fci_energy(hopping, onsite_U, n_occ, n_occ)
    return float((e_ref - e_seci) / (e_ref - e_fci))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: benchmark, 6 electrons on the 8-site ring at local seniority 4, U/t = 4
        {"setup": "", "call": "seci_correlation_fraction(8, 1.0, 4.0, 6, 4)",
         "gold_call": "_oracle_seci_correlation_fraction(8, 1.0, 4.0, 6, 4)", "tol": 1e-7},
        # Normal: low seniority (two pairs, two spin levels) on the half-filled 6-site ring
        {"setup": "", "call": "seci_correlation_fraction(6, 1.0, 4.0, 6, 2)",
         "gold_call": "_oracle_seci_correlation_fraction(6, 1.0, 4.0, 6, 2)", "tol": 1e-7},
        # Boundary: seniority zero (orbital-optimized DOCI) on the half-filled 6-site ring
        {"setup": "", "call": "seci_correlation_fraction(6, 1.0, 4.0, 6, 0)",
         "gold_call": "_oracle_seci_correlation_fraction(6, 1.0, 4.0, 6, 0)", "tol": 1e-7},
        # Edge: two electrons on a 3-site ring at seniority 2
        {"setup": "", "call": "seci_correlation_fraction(3, 1.0, 2.0, 2, 2)",
         "gold_call": "_oracle_seci_correlation_fraction(3, 1.0, 2.0, 2, 2)", "tol": 1e-7},
        # Invalid: completely filled ring (8 electrons on 4 sites), where f = 0/0
        {"setup": """def run_model():
    try:
        seci_correlation_fraction(4, 1.0, 4.0, 8, 0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_seci_correlation_fraction(4, 1.0, 4.0, 8, 0)
        return 0
    except ValueError:
        return 1
""", "call": "run_model()", "gold_call": "run_oracle()"},
        # Invalid: open-shell reference (4 electrons on the 6-site ring)
        {"setup": """def run_model():
    try:
        seci_correlation_fraction(6, 1.0, 4.0, 4, 2)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_seci_correlation_fraction(6, 1.0, 4.0, 4, 2)
        return 0
    except ValueError:
        return 1
""", "call": "run_model()", "gold_call": "run_oracle()"},
    ]
