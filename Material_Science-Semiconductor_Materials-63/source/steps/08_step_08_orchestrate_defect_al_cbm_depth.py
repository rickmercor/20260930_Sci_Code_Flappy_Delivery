"""
*equal-weight temperature average. If n_per_group is None, derive floor(n_labeled / 3) from the force screen. Call prior public compute_* APIs only.*

The coupled workflow gates electronic acquisition on force uncertainty, then reports a temperature-aggregated As_Ga mid-gap depth below the CBM.

Returns
-------
float, final As_Ga CBM-depth metric in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

import numpy as np


def orchestrate_defect_al_cbm_depth(
    force_ensembles: np.ndarray,
    force_threshold: float,
    h_ensembles: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    n_per_group: int | None = None,
    asga_defect_id: int = 4,
) -> float:
    """Run the full AL + As_Ga conduction-referenced depth evaluation.

    Parameters
    ----------
    force_ensembles : np.ndarray
        Shape (n_force, 3, n_atoms, 3). Forces in eV/Å.
    force_threshold : float
        Force labeling threshold in eV/Å.
    h_ensembles : np.ndarray
        Shape (n_pool, 3, n_orb, n_orb). Ensemble Hamiltonians in eV.
    defect_ids : np.ndarray
        Shape (n_pool,). Defect labels.
    temperatures : np.ndarray
        Shape (n_pool,). Temperatures in K.
    trained_hamiltonians : np.ndarray
        Shape (n_pool, 6, 6). Trained near-gap Hamiltonians in eV.
    n_per_group : int | None, optional
        Per-(defect, temperature) acquisition quota. If None, derive it from
        the force-labeling outcome using the campaign budget rule.
    asga_defect_id : int
        Arsenic-antisite label (default 4).

    Returns
    -------
    mean_depth : float
        Equal-weight average of per-temperature As_Ga mean depths (eV).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orchestrate_defect_al_cbm_depth(
    force_ensembles: np.ndarray,
    force_threshold: float,
    h_ensembles: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    n_per_group: int | None = None,
    asga_defect_id: int = 4,
) -> float:
    force_scores = compute_force_disagreement_scores(force_ensembles)
    force_mask = compute_force_labeling_mask(force_scores, force_threshold)
    n_labeled = int(np.sum(force_mask))
    if n_labeled < 1:
        raise ValueError("force active learning selected no configurations")
    if n_per_group is None:
        n_per_group = n_labeled // 3
    n_per_group = int(n_per_group)
    if n_per_group < 1:
        raise ValueError("derived or provided n_per_group must be >= 1")

    h_scores = compute_hamiltonian_disagreement_scores(h_ensembles)
    selected = compute_balanced_acquisition_indices(
        h_scores, defect_ids, temperatures, n_per_group
    )
    ordered = compute_asga_ordered_depth_series(
        selected, defect_ids, temperatures, trained_hamiltonians, asga_defect_id
    )
    paired = compute_temperature_paired_asga_means(ordered, n_per_group)
    p = np.asarray(paired, dtype=float).reshape(-1)
    if p.size < 1:
        raise ValueError("empty temperature-paired means")
    return float(p.sum() / float(p.size) + 0.0 * float(n_labeled))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "force_ensembles = np.zeros((10, 3, 6, 3))\n"
                "B0 = np.array([[0.02,-0.01,0.00],[-0.03,0.02,0.01],[0.01,0.00,-0.02],"
                "[0.00,0.04,-0.01],[-0.02,-0.01,0.03],[0.01,0.02,0.00]], dtype=float)\n"
                "for i in range(4):\n"
                "    force_ensembles[i,0]=B0; force_ensembles[i,1]=B0+0.001*(i+1); "
                "force_ensembles[i,2]=B0-0.001*(i+1)\n"
                "B1 = np.array([[0.10,-0.05,0.02],[-0.08,0.06,0.04],[0.05,0.01,-0.07],"
                "[0.03,0.09,-0.02],[-0.06,-0.04,0.08],[0.04,0.05,-0.03]], dtype=float)\n"
                "P1 = np.array([[1,0,-1],[0,1,0],[-1,0,1],[1,-1,0],[0,1,-1],[1,0,1]], dtype=float)\n"
                "P2 = np.array([[0,1,0],[1,0,-1],[0,-1,1],[-1,0,1],[1,1,0],[0,-1,0]], dtype=float)\n"
                "for j,i in enumerate(range(4,10)):\n"
                "    force_ensembles[i,0]=B1\n"
                "    force_ensembles[i,1]=B1+0.02*(j+1)*P1\n"
                "    force_ensembles[i,2]=B1-0.025*(j+1)*P2\n"
                "force_threshold = 0.01\n"
                "defect_ids=[]; temperatures=[]; h_ensembles=[]; trained_hamiltonians=[]\n"
                "Pw=np.zeros((6,6)); Pw[0,1]=Pw[1,0]=0.02; "
                "Pw[0,2]=Pw[2,0]=0.01; Pw[1,2]=Pw[2,1]=0.015; "
                "Pw[1,3]=Pw[3,1]=0.005; Pw[2,3]=Pw[3,2]=0.01; "
                "Pw[2,4]=Pw[4,2]=0.004; Pw[3,4]=Pw[4,3]=0.008; "
                "Pw[3,5]=Pw[5,3]=0.003; Pw[4,5]=Pw[5,4]=0.012\n"
                "Pv=np.zeros((6,6)); Pv[0,1]=Pv[1,0]=0.005; "
                "Pv[0,3]=Pv[3,0]=0.018; Pv[0,4]=Pv[4,0]=0.011; "
                "Pv[1,2]=Pv[2,1]=0.007; Pv[1,4]=Pv[4,1]=0.014; "
                "Pv[1,5]=Pv[5,1]=0.009; Pv[2,3]=Pv[3,2]=0.016; "
                "Pv[2,5]=Pv[5,2]=0.006; Pv[3,4]=Pv[4,3]=0.013\n"
                "w=np.array([1.0,2.5,4.0,5.5,5.5,1.0,2.5,4.0,5.55,1.05,2.55,4.05,4.05,5.55,1.05,2.55,"
                "4.1,5.6,1.1,2.6,2.6,4.1,5.6,1.1,2.65,4.15,5.65,1.15,1.15,2.65,4.15,5.65,"
                "0.91,3.91,4.1,3.2,5.01,2.2,1.85,4.38], dtype=float)\n"
                "v=np.array([1.5,5.0,0.5,4.0,4.0,1.5,5.0,0.5,4.03,1.53,5.03,0.53,0.53,4.03,1.53,5.03,"
                "0.56,4.06,1.56,5.06,5.06,0.56,4.06,1.56,5.09,0.59,4.09,1.59,1.59,5.09,0.59,4.09,"
                "4.34,2.93,2.44,3.54,4.52,5.8,6.39,4.35], dtype=float)\n"
                "amps=[(-1.0,0.0),(0.0,1.0),(1.0,-1.0)]\n"
                "for d in range(5):\n"
                "  for T in (100.0, 500.0):\n"
                "    for c in range(4):\n"
                "      i=8*d+4*(T==500.0)+c\n"
                "      Eg=0.745-1.5e-4*(T-100.0); VBM=-0.3725; CBM=VBM+Eg\n"
                "      frac=0.28+0.05*d\n"
                "      Ed=VBM+frac*Eg+0.008*(c-1.5)+0.5e-4*(T-100.0)\n"
                "      D=np.array([VBM-1.20,VBM-0.60,VBM,Ed,CBM,CBM+0.55], dtype=float)\n"
                "      Href=np.diag(D)+2.0*(w[i]*Pw+v[i]*Pv)\n"
                "      Hs=np.stack([Href+0.002*(a*w[i]*Pw+b*v[i]*Pv) for a,b in amps])\n"
                "      defect_ids.append(d); temperatures.append(T)\n"
                "      h_ensembles.append(Hs); trained_hamiltonians.append(Href)\n"
                "defect_ids=np.asarray(defect_ids,dtype=int)\n"
                "temperatures=np.asarray(temperatures,dtype=float)\n"
                "h_ensembles=np.asarray(h_ensembles,dtype=float)\n"
                "trained_hamiltonians=np.asarray(trained_hamiltonians,dtype=float)\n"
                "asga_defect_id=4\n"
            ),
            "call": (
                "orchestrate_defect_al_cbm_depth(force_ensembles, force_threshold, h_ensembles, "
                "defect_ids, temperatures, trained_hamiltonians, None, asga_defect_id)"
            ),
            "gold_call": (
                "_oracle_orchestrate_defect_al_cbm_depth(force_ensembles, force_threshold, h_ensembles, "
                "defect_ids, temperatures, trained_hamiltonians, None, asga_defect_id)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "force_ensembles = np.zeros((1, 3, 1, 3))\n"
                "force_ensembles[0, 1, 0, 0] = 0.02\n"
                "force_ensembles[0, 2, 0, 0] = -0.02\n"
                "force_threshold = 0.01\n"
                "defect_ids = np.array([4, 4])\n"
                "temperatures = np.array([100.0, 100.0])\n"
                "trained_hamiltonians = np.stack([\n"
                " np.diag([-2.0, -1.5, -1.0, 0.0, 0.5, 1.0]),\n"
                " np.diag([-2.0, -1.5, -1.0, 0.1, 0.5, 1.0]),\n"
                "])\n"
                "h_ensembles = np.zeros((2, 3, 6, 6))\n"
                "for i, sc in enumerate([2.0, 1.0]):\n"
                "    for m, amp in enumerate((-1.0, 0.0, 1.0)):\n"
                "        h_ensembles[i, m] = trained_hamiltonians[i] + amp * 0.01 * sc * np.eye(6)\n"
                "n_per_group = 1\n"
                "asga_defect_id = 4\n"
            ),
            "call": (
                "orchestrate_defect_al_cbm_depth(force_ensembles, force_threshold, h_ensembles, "
                "defect_ids, temperatures, trained_hamiltonians, n_per_group, asga_defect_id)"
            ),
            "gold_call": (
                "_oracle_orchestrate_defect_al_cbm_depth(force_ensembles, force_threshold, h_ensembles, "
                "defect_ids, temperatures, trained_hamiltonians, n_per_group, asga_defect_id)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "force_ensembles = np.zeros((2, 3, 2, 3))\n"
                "force_ensembles[1, 1, 0, 0] = 0.05\n"
                "force_ensembles[1, 2, 0, 0] = -0.05\n"
                "force_threshold = 0.01\n"
                "defect_ids = np.array([4, 4, 0, 0, 4, 4, 0, 0])\n"
                "temperatures = np.array([100.0, 100.0, 100.0, 100.0, 500.0, 500.0, 500.0, 500.0])\n"
                "eigs = np.vstack([\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.01, 0.3725, 0.9225]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, 0.00, 0.3725, 0.9225]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.05, 0.3725, 0.9225]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.05, 0.3725, 0.9225]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.02, 0.3125, 0.8625]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.01, 0.3125, 0.8625]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.05, 0.3725, 0.9225]),\n"
                " np.array([-1.5725, -0.9725, -0.3725, -0.05, 0.3725, 0.9225]),\n"
                "])\n"
                "trained_hamiltonians = np.stack([np.diag(e) for e in eigs])\n"
                "h_ensembles = np.zeros((8, 3, 6, 6))\n"
                "for i, scale in enumerate([3.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0]):\n"
                "    h_ensembles[i] = np.stack([trained_hamiltonians[i] + s * 0.002 * scale * np.eye(6) for s in (-1, 0, 1)])\n"
                "n_per_group = 1\n"
                "asga_defect_id = 4\n"
            ),
            "call": (
                "orchestrate_defect_al_cbm_depth(force_ensembles, force_threshold, h_ensembles, "
                "defect_ids, temperatures, trained_hamiltonians, n_per_group, asga_defect_id)"
            ),
            "gold_call": (
                "_oracle_orchestrate_defect_al_cbm_depth(force_ensembles, force_threshold, h_ensembles, "
                "defect_ids, temperatures, trained_hamiltonians, n_per_group, asga_defect_id)"
            ),
        },
    ]
