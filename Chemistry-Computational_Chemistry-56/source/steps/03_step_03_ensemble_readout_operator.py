"""
Average the Heisenberg-picture site-population readout over an ensemble of donor-acceptor dimers with static site-energy disorder.

Pigments in a protein ensemble do not all see the same local electrostatic environment, so their excitation
energies scatter from molecule to molecule. On the femtosecond to picosecond time scale of energy transfer this
inhomogeneity is frozen: every molecule evolves with its own site energies, while the electronic coupling, the
vibrational baths and the loss rates are shared. An optical experiment on the whole sample prepares every molecule in
the same way and records a population that is averaged over the molecules, so the signal is governed by the
ensemble average of each molecule's readout operator.

Only the difference of the two site energies enters the population dynamics of a dimer, because a common shift of
both sites commutes with the electronic Hamiltonian, the site couplings to the baths and the loss channels. When the
two site energies are independent Gaussian variables, the energy gap is Gaussian too, and a Gauss-Hermite rule
turns the ensemble average into a weighted sum over a few representative gaps with exponential accuracy for smooth
integrands.

Returns
-------
numpy.ndarray of shape (n_t, 8): the Gauss-Hermite disorder average of the heom_readout_operator rows
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ensemble_readout_operator(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray", n_matsubara: int,
                              depth: int, n_nodes: int, readout_site: int, times_ps: "np.ndarray") -> "np.ndarray":
    '''Disorder-averaged Heisenberg-picture site-population readout of an inhomogeneous dimer ensemble.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa] with J and the mean gap mean(eps_D) - mean(eps_A) in cm^-1 and
        the loss rates in ps^-1, as in heom_readout_operator.
    site_sigma_cm : float
        Standard deviation (cm^-1, >= 0) of each site energy. eps_D and eps_A are independent normal variables with
        this common standard deviation; J, the baths and the rates are the same for every molecule.
    bath : np.ndarray
        Shape (3,), [E_R, gamma_c, T] as in heom_readout_operator.
    n_matsubara : int
        Matsubara terms per bath, >= 0, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, >= 0, as in heom_readout_operator.
    n_nodes : int
        Number of nodes, >= 1, of the Gauss-Hermite rule for the normal distribution (probabilists' Hermite nodes
        x_i with weights normalized to sum to 1) used for the average over the gap distribution; the molecule at node
        i has the gap mean_gap + s x_i, where s is the standard deviation of the gap.
    readout_site : int
        0 for the donor population, 1 for the acceptor population.
    times_ps : np.ndarray
        Shape (n_t,), non-negative delays in ps.

    Returns
    -------
    readout : np.ndarray
        Shape (n_t, 8), the weighted sum over the nodes of the rows returned by heom_readout_operator,
        [M_DD, M_AA, Re M_DA, Im M_DA, dM_DD/dt, dM_AA/dt, Re dM_DA/dt, Im dM_DA/dt], so that Tr[M(t) rho_0] is the
        ensemble-averaged population of the readout site at delay t when every molecule starts in rho_0.

    Raises
    ------
    ValueError
        If site_sigma_cm is negative or n_nodes is smaller than 1, and in the cases listed for heom_readout_operator.
    '''
    return readout

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ensemble_readout_operator(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray",
                                      n_matsubara: int, depth: int, n_nodes: int, readout_site: int,
                                      times_ps: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if site_sigma_cm < 0 or n_nodes < 1:
        raise ValueError("site_sigma_cm must be non-negative and n_nodes at least 1")
    dimer = np.asarray(dimer, dtype=float)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    weights = weights / weights.sum()
    gap_sigma = np.sqrt(2.0) * site_sigma_cm      # difference of two independent site energies
    total = None
    for x, w in zip(nodes, weights):
        member = dimer.copy()
        member[1] = dimer[1] + gap_sigma * x
        rows = _oracle_heom_readout_operator(member, bath, n_matsubara, depth, readout_site, times_ps)
        total = w * rows if total is None else total + w * rows
    return total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: acceptor readout of a disordered reference ensemble on a shallow hierarchy ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "times = np.array([0.05, 0.2, 0.35])\n",
            "call": "ensemble_readout_operator(dimer.copy(), 70.0, bath.copy(), 1, 2, 7, 1, times.copy())",
            "gold_call": "_oracle_ensemble_readout_operator(dimer, 70.0, bath, 1, 2, 7, 1, times)",
            "tol": 1e-6,
        },
        # --- Boundary: a single node samples only the mean gap, whatever the disorder width ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "times = np.array([0.1, 0.3])\n",
            "call": "ensemble_readout_operator(dimer.copy(), 200.0, bath.copy(), 1, 3, 1, 1, times.copy())",
            "gold_call": "_oracle_ensemble_readout_operator(dimer, 200.0, bath, 1, 3, 1, 1, times)",
            "tol": 1e-6,
        },
        # --- Edge: broad disorder around a resonant pair, where many molecules have the acceptor above the donor ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([60.0, 20.0, 0.0, 0.5])\n"
                     "bath = np.array([40.0, 90.0, 250.0])\n"
                     "times = np.array([0.04, 0.12, 0.3, 0.6])\n",
            "call": "ensemble_readout_operator(dimer.copy(), 110.0, bath.copy(), 0, 3, 9, 1, times.copy())",
            "gold_call": "_oracle_ensemble_readout_operator(dimer, 110.0, bath, 0, 3, 9, 1, times)",
            "tol": 1e-6,
        },
        # --- Edge: donor readout of a narrow ensemble at low temperature with two Matsubara terms ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([100.0, 300.0, 0.2, 0.0])\n"
                     "bath = np.array([15.0, 45.0, 120.0])\n"
                     "times = np.array([0.08, 0.5])\n",
            "call": "ensemble_readout_operator(dimer.copy(), 12.0, bath.copy(), 2, 2, 5, 0, times.copy())",
            "gold_call": "_oracle_ensemble_readout_operator(dimer, 12.0, bath, 2, 2, 5, 0, times)",
            "tol": 1e-6,
        },
        # --- Edge: very broad disorder against a fast weak bath, donor readout of a strongly trapped pair ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([120.0, 350.0, 0.0, 2.0])\n"
                     "bath = np.array([10.0, 200.0, 350.0])\n"
                     "times = np.array([0.02, 0.1])\n",
            "call": "ensemble_readout_operator(dimer.copy(), 150.0, bath.copy(), 0, 4, 12, 0, times.copy())",
            "gold_call": "_oracle_ensemble_readout_operator(dimer, 150.0, bath, 0, 4, 12, 0, times)",
            "tol": 1e-6,
        },
        # --- Invalid: a negative disorder width must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(dimer, -5.0, bath, 1, 2, 4, 1, np.array([0.1]))\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(ensemble_readout_operator)",
            "gold_call": "run(_oracle_ensemble_readout_operator)",
        },
    ]
