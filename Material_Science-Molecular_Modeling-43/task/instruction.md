# Material_Science-Molecular_Modeling-43

## Background

An interatomic potential can fit its training labels yet fail on independent structures or short atomic separations. This deterministic three-system fixture permits a reproducible audit without running molecular dynamics or electronic-structure software. Its analytic teacher and ridge student are not the neural architectures in the paper. The synthetic data are not paper measurements.

Instance controls: use NumPy float64 and the draw order below. The non-Li framework MSD limit is 0.02 square angstrom from frame 10; feature coverage uses 24 bins and PCA eigenvalue cutoff 1.0. The periodic RDF box has side 12 angstrom with 25 equally spaced radial edges from zero to 6 angstrom; frame spacing is 0.1 ps. The student ridge penalty is 0.1. These are supplied fixture settings, not source-method claims. For each system, concatenate its model-MD and AIMD embedding rows; subtract the combined panel mean and divide by its population standard deviation feature by feature, replacing a zero divisor by one, then fit PCA on that same standardized panel. The RDF includes all 200 atoms. Average all 36 AIMD frames to form each system's reference RDF. At 0.1 ps per sampled frame, a complete 1 ps model window comprises 10 frames; use all 27 overlapping trailing windows, ending at 1.0, 1.1, ..., 3.6 ps. For any radial integral required by the source method, use rectangular-bin quadrature with the supplied bin widths. These finite-sampling and quadrature choices are fixture conventions. The synthetic threshold-excess burden includes every complete window, not just the first failing or maximum-discrepancy window.

import numpy as np

def make_fixture(seed: int = 7261, pseudo_size: int = 2000) -> list[dict]:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(pseudo_size, bool) or not isinstance(pseudo_size, (int, np.integer)) or pseudo_size < 100:
        raise ValueError("pseudo_size must be an integer at least 100")
    rng = np.random.default_rng(seed)
    systems = []
    for system in range(3):
        initial = rng.uniform(1.0, 11.0, size=(200, 3))
        direction = rng.normal(size=(200, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        time = np.arange(36, dtype=np.float64)
        displacement = time[:, None, None] * direction[None, :, :] * (0.0015 + 0.0001 * system)
        displacement[:, :8] *= 12
        trajectory = initial[None, :, :] + displacement
        aimd_positions = trajectory + rng.normal(scale=0.030, size=trajectory.shape)
        model_positions = aimd_positions + rng.normal(scale=(0.002 if system != 1 else 0.036), size=trajectory.shape)
        common = rng.normal(size=(160, 9))
        mace_embeddings = common + rng.normal(scale=0.22 + 0.02 * system, size=common.shape)
        aimd_embeddings = common + rng.normal(scale=0.14, size=common.shape)
        r = np.linspace(0.2, 1.5, 22)
        energy = 3.0 / r**2
        if system == 1:
            energy = energy - 3.0 * np.exp(-((r - 0.91) / 0.12) ** 2)
        pseudo_features = rng.normal(size=(pseudo_size, 6))
        test_features = rng.normal(size=(1000, 6))
        reference_coeff = np.array([0.55, -0.31, 0.20, 0.11, -0.19, 0.24]) + 0.02 * system
        teacher_coeff = reference_coeff + np.array([0.025, -0.018, 0.013, 0.0, 0.01, -0.007])
        dft_test = test_features @ reference_coeff + 0.055 * np.sin(test_features[:, 0] * test_features[:, 1])
        systems.append({
            "name": ("LGPS", "LATP", "LYC")[system],
            "trajectory": trajectory, "non_li": np.arange(200) >= 8,
            "aimd_positions": aimd_positions, "model_positions": model_positions,
            "mace_embeddings": mace_embeddings, "aimd_embeddings": aimd_embeddings,
            "distance": r, "diatomic_energy": energy,
            "pseudo_features": pseudo_features, "teacher_coeff": teacher_coeff,
            "test_features": test_features, "dft_test": dft_test,
        })
    return systems

def student_fixture_design(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return np.column_stack((np.ones(len(x)), x, x[:, 0] * x[:, 1], x[:, 0] ** 2))

def teacher_label_pseudodata(pseudo_features: np.ndarray, teacher_coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, c = np.asarray(pseudo_features, float), np.asarray(teacher_coeff, float)
    if x.ndim != 2 or x.shape[0] < 100 or x.shape[1] != 6 or c.shape != (6,):
        raise ValueError("expected at least 100 pseudo rows, six features and six coefficients")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("nonfinite pseudo data")
    labels = x @ c + 0.055 * np.sin(x[:, 0] * x[:, 1])
    return x.copy(), labels

def fit_fixture_student(pseudo_features, teacher_coeff, test_features):
    x, labels = teacher_label_pseudodata(pseudo_features, teacher_coeff)
    design = student_fixture_design(x)
    ridge = 0.1
    coeff = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ labels)
    return student_fixture_design(test_features) @ coeff

## Problem

Audit a synthetic LGPS, LATP and LYC instance using the attached study's framework, coverage, relative-error, structural-stability, short-range-force and teacher-labelling methods; the supplied arrays and analytic teacher/student are benchmark fixtures, not the study's measured trajectories, DFT calculations or neural models.

For each system in that order, report feature coverage, held-out transition-energy RRMSE, first RDF stability-failure time or 'none', and the number of teacher-labelled pseudo-configurations. Separately report whether the LYC non-Li framework passes the supplied displacement limit and the LATP Li-Li turning position. For the LATP discrete diatomic curve, set segment force to minus its energy secant slope and linearly interpolate the first positive-to-nonpositive crossing between adjacent segment midpoints. Give numerical coverage, turning position and per-system RRMSE to at least six digits after the decimal point, and each system's maximum RDF discrepancy and threshold-excess burden to at least eight digits after the decimal point, without presenting fixture values as paper measurements.

For each system, compute the 27 complete one-picosecond RDF discrepancies D from the study's stability test and report their maximum. Also compute its threshold-excess burden: the arithmetic mean over those 27 windows of max(D − 0.2, 0). The final number J is the arithmetic mean of the three system burdens, reported to eight decimal places. This burden summary is a synthetic audit statistic, not a quantity claimed by the paper; use the paper's strict D > 0.2 rule only for individual-window failure. Explain in the reasoning section the independent DFT-labelled test rows, teacher pseudo-labels versus new DFT labels, grounding source-method claims in the paper.

Use the source paper, not the synthetic fixture, for a separate evidence check in the reasoning section. Report the pretrained MACE-MP-0 baseline’s approximate feature-space coverage and its mean relative transition-energy error before fine-tuning, and explain why coverage alone does not establish energy accuracy. State whether MACE trained from scratch sustained stable molecular dynamics across all three electrolytes even with 800 DFT-labelled configurations, and whether fine-tuning the pretrained potential maintained stable high-temperature dynamics with only 10 DFT-labelled configurations. Give the fine-tuned potential’s mean transition-energy RRMSE at a 200-configuration DFT budget. Identify the teacher model for the distilled NEP pseudo-labels, state whether those labels required new DFT calculations, and report the distilled NEP’s relative transition-energy RMSE reduction against the DFT-only compact baseline. Label these as paper results, not as values calculated from the fixture. Keep the synthetic J as the sole final answer.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Wrap the scientific reasoning, source-method explanations, and requested diagnostics in <reasoning>...</reasoning> tags.
Place only the arithmetic mean of the three per-system RDF threshold-excess burdens, rounded to eight decimal places, inside <final_answer>...</final_answer> tags.
Do not place units, prose, or a second value inside the final-answer tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

framework_msd_screen

Goal
----
Screen a candidate high-temperature trajectory for abnormal migration of non-Li framework atoms.

```python
def framework_msd_screen(positions: "np.ndarray", non_li: "np.ndarray", max_msd: float, start: int = 0) -> tuple["np.ndarray", bool]:
    """Return the non-Li MSD trace and whether its maximum from ``start`` is at most ``max_msd``.

    Positions are unwrapped Cartesian coordinates, shape (time>=2, atoms, 3).
    ``non_li`` is an atom-length boolean mask with at least one selected atom.
    ``max_msd`` is a finite nonnegative fixture limit in square angstrom.
    ``start`` is the inclusive zero-based frame index; default 0.
    Raise ValueError on malformed geometry, mask, limit or start.
    """
    return None
```

### Step 2

feature_coverage

Goal
----
Measure how much of the AIMD pretrained-feature space is represented by a model-MD trajectory.

```python
def feature_coverage(mace: "np.ndarray", aimd: "np.ndarray", bins: int, eigen_cutoff: float = 1.0) -> tuple[float, "np.ndarray"]:
    """Return the source feature-coverage summary and its component ratios.

    Inputs are finite pretrained embedding matrices with at least two rows
    each and the same feature width. bins is an integer at least two;
    eigen_cutoff defaults to 1.0. For fixture preprocessing, use the
    population standard deviation on the joined panel, replacing a zero
    divisor by one. A value on the uppermost histogram edge belongs to the
    last bin. Source Eqs. 4-5 determine component retention and coverage.
    Raise ValueError on invalid inputs or if no component is retained.
    """
    return None
```

### Step 3

relative_rmse

Goal
----
Compute the source's relative root mean squared error on a held-out reference panel.

```python
def relative_rmse(reference: "np.ndarray", prediction: "np.ndarray") -> float:
    """Return the source Eq. 1 relative-error metric as a percentage.

    Inputs are matched, finite, one-dimensional test vectors of equal length
    at least two. Raise ValueError for invalid shapes, nonfinite entries, or
    a zero reference normalization denominator.
    """
    return None
```

### Step 4

radial_distribution

Goal
----
Compute a frame-averaged radial distribution function from periodically imaged pair distances.

```python
def radial_distribution(positions: "np.ndarray", box_length: float, edges: "np.ndarray") -> "np.ndarray":
    """Return the source Eq. 2 frame-mean RDF on the supplied radial bins.

    Positions have shape (frames>=1, atoms>=2, 3) in a cubic periodic box.
    Pair distances use the minimum-image convention. Bin edges are strictly
    increasing, finite and nonnegative, ending no farther than half the box
    length. Use left-closed bins and include the final right edge. Return one
    radial value per bin. Raise ValueError on invalid geometry or controls.
    """
    return None
```

### Step 5

rdf_stability

Goal
----
Detect when model-MD structure deviates from the AIMD RDF reference.

```python
def rdf_stability(aimd_rdf: "np.ndarray", model_rdf_by_frame: "np.ndarray", edges: "np.ndarray", frame_dt_ps: float, window_ps: float = 1.0, threshold: float = 0.2) -> tuple["np.ndarray", float | None]:
    """Return source Eq. 3 deviations and the first stability-failure time.

    The model RDF is sampled every frame_dt_ps; only complete trailing
    windows are evaluated. The default window is 1 ps and the supplied
    threshold defaults to 0.2. Use rectangular quadrature on the supplied
    radial bin edges. Return the first failing window end time in ps, or None
    when no complete window fails. Raise ValueError on mismatched arrays or
    invalid controls.
    """
    return None
```

### Step 6

diatomic_turning_point

Goal
----
Find the short-range Li–Li force reversal used in the paper's stability diagnosis.

```python
def diatomic_turning_point(distance: "np.ndarray", energy: "np.ndarray") -> float:
    """Return the first short-range repulsive-to-attractive Li-Li force reversal.

    Both vectors must be finite, equal length at least three, and distances
    strictly increasing. Segment force is minus the energy secant slope.
    Interpolate the first positive-to-nonpositive crossing between adjacent
    segment midpoints; return 0.1 angstrom as the paper's plotting marker
    if no crossing occurs. Raise ValueError on malformed curves.
    """
    return None
```

### Step 7

compute_source_method_audit

Goal
----
Apply the source-defined diagnostics to three deterministic synthetic systems and return their per-system audit record with J equal to the arithmetic mean of each system's complete-window RDF threshold-excess burden. For each system, average max(D − 0.2, 0) over its 27 complete one-picosecond windows, where D is the paper-defined integrated RDF discrepancy. The synthetic fixture and compact ridge student are disclosed stand-ins, not actual MACE, ELoRA, NEP or DFT results. The public orchestrator calls all six preceding public functions. Keep both maximum RDF deviations and held-out transition-energy RRMSEs in the native three-row record as separate diagnostics.

```python
def compute_source_method_audit(seed: int = 7261, pseudo_size: int = 2000) -> tuple[float, "np.ndarray"]:
    """Return mean threshold-excess RDF burden and its per-system audit.

    Default seed is 7261 and default teacher-labelled pseudo-set size is
    2000. Seed must be a nonnegative integer; pseudo_size is an integer at
    least 100. Generate the fixed synthetic instance specified in the task's
    instance-data appendix. Call all six preceding public steps in system
    order LGPS, LATP, LYC. The teacher labels the pseudo-rows without a DFT
    call. A disclosed ridge student stand-in is fit only to those labels; it
    is never refit on the independent DFT-labelled test rows. Return the
    arithmetic mean of the three systemwise mean excesses max(D−0.2,0) across all complete one-picosecond RDF windows and a (3,9)
    float64 record. Rows are LGPS, LATP, LYC; columns are framework-pass
    flag (1/0), terminal framework MSD in square angstrom, coverage ratio,
    maximum RDF deviation, mean threshold-excess burden, first RDF failure time in ps (-1 if absent),
    diatomic turning position in angstrom, pseudo-row count, and held-out
    RRMSE percent, in that order.
    Raise ValueError on invalid seed or pseudo_size.
    """
    return None
```
