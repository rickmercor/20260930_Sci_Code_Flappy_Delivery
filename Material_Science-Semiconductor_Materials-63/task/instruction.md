# Material_Science-Semiconductor_Materials-63

## Background

Point defects in compound semiconductors such as GaAs control carrier concentrations, non-radiative recombination, and mid-gap trapping that limit photovoltaic and optoelectronic performance. First-principles supercell calculations can locate defect levels, but artificial defect–defect interactions and the need for large cells make density-functional evaluation along thermally disordered molecular-dynamics trajectories prohibitively expensive.

Machine-learned equivariant force fields can generate those trajectories, yet they do not by themselves supply electronic Hamiltonians. Equivariant Hamiltonian learning can map atomic geometries to electronic structure, but training still requires carefully chosen DFT labels.

## Problem

Finite-temperature GaAs defect electronics are analyzed with a recent equivariant message-passing study that couples uncertain-snapshot labeling to near-gap spectral readout. On the fixed instance below, report the temperature-aggregated arsenic-antisite mid-gap depth below the conduction-band minimum (eV) that the study's coupled labeling workflow assigns to this pool. The reported scalar is the equal-weight arithmetic mean of the per-temperature arsenic-antisite mean depths.

Forces `(10,3,6,3)` eV/Å: `B0=[[0.02,-0.01,0.00],[-0.03,0.02,0.01],[0.01,0.00,-0.02],[0.00,0.04,-0.01],[-0.02,-0.01,0.03],[0.01,0.02,0.00]]`, `B1=[[0.10,-0.05,0.02],[-0.08,0.06,0.04],[0.05,0.01,-0.07],[0.03,0.09,-0.02],[-0.06,-0.04,0.08],[0.04,0.05,-0.03]]`, `P1=[[1,0,-1],[0,1,0],[-1,0,1],[1,-1,0],[0,1,-1],[1,0,1]]`, `P2=[[0,1,0],[1,0,-1],[0,-1,1],[-1,0,1],[1,1,0],[0,-1,0]]`; frame `i=0..3` uses `B0`, `B0+0.001*(i+1)`, `B0-0.001*(i+1)`; frames `i=4..9` (`j=i-4`) use `B1`, `B1+0.02*(j+1)*P1`, `B1-0.025*(j+1)*P2`.

The campaign's acquisition budget grants each defect–temperature group one slot for every three snapshots that clear the study's force-labeling screen on the ensemble above, and each group spends its slots on its own most-disagreeing electronic candidates. A candidate's ensemble disagreement is the root-mean-square, over the matrix elements, of the element-wise population standard deviation across that row's three Hamiltonians.

Electronic pool: 40 rows indexed `0..39` in nested order over defect label `d∈{0,1,2,3,4}`, temperature `T∈{100,500}` K, and candidate `c∈{0,1,2,3}`, with `i=8*d+4*(T==500)+c`; arsenic antisite (`As_Ga`) is `d=4`. Row `i` carries a seed diagonal `D_i=(E_0,…,E_5)` and two coupling scales `w[i]`, `v[i]`. Hybridization is carried by two distinct fixed symmetric templates with upper-triangle entries `Pw[0,1]=0.02`, `Pw[0,2]=0.01`, `Pw[1,2]=0.015`, `Pw[1,3]=0.005`, `Pw[2,3]=0.01`, `Pw[2,4]=0.004`, `Pw[3,4]=0.008`, `Pw[3,5]=0.003`, `Pw[4,5]=0.012` and `Pv[0,1]=0.005`, `Pv[0,3]=0.018`, `Pv[0,4]=0.011`, `Pv[1,2]=0.007`, `Pv[1,4]=0.014`, `Pv[1,5]=0.009`, `Pv[2,3]=0.016`, `Pv[2,5]=0.006`, `Pv[3,4]=0.013` (and `M[j,i]=M[i,j]` for both). The row's trained Hamiltonian is `H_ref,i=diag(D_i)+2.0*(w[i]*Pw+v[i]*Pv)`, and the near-gap levels of an acquired row are the eigenvalues of that matrix. The three ensemble members scatter around the trained Hamiltonian by a much smaller model uncertainty, `H_m,i=H_ref,i+0.002*(a_m*w[i]*Pw+b_m*v[i]*Pv)`, with amplitude pairs `(a_m,b_m)∈{(-1,0),(0,1),(1,-1)}`.

Seed diagonal table `D_i=(E_0,…,E_5)` in eV with row scales `w` and `v`: `w=[1.0,2.5,4.0,5.5,5.5,1.0,2.5,4.0,5.55,1.05,2.55,4.05,4.05,5.55,1.05,2.55,4.1,5.6,1.1,2.6,2.6,4.1,5.6,1.1,2.65,4.15,5.65,1.15,1.15,2.65,4.15,5.65,0.91,3.91,4.1,3.2,5.01,2.2,1.85,4.38]`, `v=[1.5,5.0,0.5,4.0,4.0,1.5,5.0,0.5,4.03,1.53,5.03,0.53,0.53,4.03,1.53,5.03,0.56,4.06,1.56,5.06,5.06,0.56,4.06,1.56,5.09,0.59,4.09,1.59,1.59,5.09,0.59,4.09,4.34,2.93,2.44,3.54,4.52,5.8,6.39,4.35]`.
`[-1.5725,-0.9725,-0.3725,-0.1759,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1679,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1599,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1519,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1727,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1647,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1567,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1487,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1386,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1306,0.3725,0.9225]`
`[-1.5725,-0.9725,-0.3725,-0.1226,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1146,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.1384,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1304,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1224,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1144,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.1014,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0934,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0854,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0774,0.3725,0.9225]`
`[-1.5725,-0.9725,-0.3725,-0.1042,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0962,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0882,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0802,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0641,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0561,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0481,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0401,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0699,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0619,0.3125,0.8625]`
`[-1.5725,-0.9725,-0.3725,-0.0539,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0459,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0269,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0189,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0109,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0029,0.3725,0.9225] [-1.5725,-0.9725,-0.3725,-0.0357,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0277,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0197,0.3125,0.8625] [-1.5725,-0.9725,-0.3725,-0.0117,0.3125,0.8625]`.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_score_force_disagreement

Goal
----
Score per-frame force-model disagreement for defective-supercell active learning.

```python
import numpy as np


def compute_force_disagreement_scores(force_ensembles: np.ndarray) -> np.ndarray:
    """Return per-configuration force-ensemble disagreement scores.

    Parameters
    ----------
    force_ensembles : np.ndarray
        Shape (n_configs, n_models, n_atoms, 3). Forces in eV/Å.

    Returns
    -------
    scores : np.ndarray
        Shape (n_configs,). Disagreement scores in eV/Å.
    """
    return np.asarray([], dtype=float)
```

### Step 2

02_force_labeling_mask

Goal
----
Build the binary force active-learning keep mask from disagreement scores.

```python
import numpy as np


def compute_force_labeling_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return a 0/1 mask for force-AL labeling eligibility.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_configs,). Force disagreement scores in eV/Å.
    threshold : float
        Labeling threshold in eV/Å.

    Returns
    -------
    mask : np.ndarray
        Shape (n_configs,) with values in {0.0, 1.0}.
    """
    return np.asarray([], dtype=float)
```

### Step 3

03_score_hamiltonian_disagreement

Goal
----
Score per-structure Hamiltonian-model disagreement for electronic AL

```python
import numpy as np


def compute_hamiltonian_disagreement_scores(h_ensembles: np.ndarray) -> np.ndarray:
    """Return per-structure Hamiltonian ensemble disagreement scores.

    

    Parameters
    ----------
    h_ensembles : np.ndarray
        Shape (n_configs, n_models, n_orb, n_orb). Hamiltonians in eV.

    Returns
    -------
    scores : np.ndarray
        Shape (n_configs,). Disagreement scores in eV.
    """
    return np.asarray([], dtype=float)
```

### Step 4

04_balanced_acquisition_indices

Goal
----
Acquire a balanced active-learning index set over defect and temperature classes

```python
import numpy as np


def compute_balanced_acquisition_indices(
    scores: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    n_per_group: int,
) -> np.ndarray:
    """Return selected pool indices for the balanced acquisition schedule.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_configs,). Hamiltonian disagreement scores.
    defect_ids : np.ndarray
        Shape (n_configs,). Integer defect labels.
    temperatures : np.ndarray
        Shape (n_configs,). Temperatures in K.
    n_per_group : int
        Per-class acquisition count for this instance. Equal scores within a
        class are ranked by ascending original pool index.

    Returns
    -------
    indices : np.ndarray
        1D float array of selected indices.
    """
    return np.asarray([], dtype=float)
```

### Step 5

05_resolve_gap_spectrum

Goal
----
Resolve valence, mid-gap, and conduction levels from truncated near-gap spectra.

```python
import numpy as np

def compute_gap_spectrum_table(hamiltonians: np.ndarray) -> np.ndarray:
    """Map near-gap Hamiltonians to band edges, defect level, and depth.

    Parameters
    ----------
    hamiltonians : np.ndarray
        Shape (6, 6) or (n, 6, 6). Symmetric near-gap Hamiltonians in eV.

    Returns
    -------
    table : np.ndarray
        Shape (4,) or (n, 4) with columns [VBM, Ed, CBM, CBM-Ed] in eV.
    """
    return np.asarray([], dtype=float)
```

### Step 6

06_asga_ordered_depth_series

Goal
----
Assemble the ordered arsenic-antisite CBM-depth series from acquired indices.

```python
import numpy as np


def compute_asga_ordered_depth_series(
    selected_indices: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    asga_defect_id: int = 4,
) -> np.ndarray:
    """Return ordered As_Ga CBM-Ed depths for acquired structures.

    Parameters
    ----------
    selected_indices : np.ndarray
        Selected pool indices from balanced acquisition.
    defect_ids : np.ndarray
        Shape (n_pool,). Defect labels.
    temperatures : np.ndarray
        Shape (n_pool,). Temperatures in K.
    trained_hamiltonians : np.ndarray
        Shape (n_pool, 6, 6). Trained near-gap Hamiltonians in eV.
    asga_defect_id : int
        Arsenic-antisite label (default 4).

    Returns
    -------
    depths : np.ndarray
        1D array of CBM-Ed values in the required order.
    """
    return np.asarray([], dtype=float)
```

### Step 7

07_temperature_paired_asga_means

Goal
----
Reduce the ordered As_Ga depth series to per-temperature means.

```python
import numpy as np


def compute_temperature_paired_asga_means(
    ordered_depths: np.ndarray, n_per_temperature: int
) -> np.ndarray:
    """Return per-temperature means from an ordered As_Ga depth series.

    Parameters
    ----------
    ordered_depths : np.ndarray
        1D ordered As_Ga depths (T ascending blocks).
    n_per_temperature : int
        Number of As_Ga depths per temperature block.

    Returns
    -------
    paired_means : np.ndarray
        Shape (n_temperatures,), block-wise arithmetic means.
    """
    return np.asarray([], dtype=float)
```

### Step 8

08_orchestrate_defect_al_cbm_depth

Goal
----
*equal-weight temperature average. If n_per_group is None, derive floor(n_labeled / 3) from the force screen. Call prior public compute_* APIs only.*

```python
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
```
