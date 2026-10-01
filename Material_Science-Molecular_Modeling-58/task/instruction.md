# Cross-system reliability audit for fine-tuned molecular-crystal potentials

## Background

Molecular-crystal polymorphs can differ by only a few kilojoules per mole, so a potential may have a low average validation error and still choose the wrong stable form. A useful audit therefore keeps compound-level errors separate and asks whether energetic rankings, crystal packing, orientational order, pair structure, energy conservation, and constant-pressure stability agree. The source paper fixes the scientific definitions and validation hierarchy. Those choices are intentionally not repeated here. The deterministic panel, graph coupling, robustness tensor, and final scalar reduction are disclosed benchmark conventions used to stress-test whether the source-defined checks remain mutually consistent.

## Problem

The source reports nine system-specific molecular-crystal potentials and argues that validation MAE alone is not enough for production molecular dynamics. Recover and justify the source choices needed to interpret the curation, ranking, packing, and ensemble-specific validation stages. Do not treat a low aggregate validation error as sufficient evidence of production reliability.

Use the nine energy and force validation MAEs from the source's training-performance table, in printed compound order, as `e0=[0.128,0.069,0.161,0.159,0.184,0.116,0.146,0.158,0.151]` and `f0=[0.762,0.848,0.414,0.501,1.043,0.562,0.695,0.627,0.377]`. The corresponding compounds are benzoic acid, benzamide, coumarin, durene, isonicotinamide, nicotinic acid, niacinamide, pyrazinamide, and resorcinol. The table reports energy MAE in kJ mol^-1 atom^-1 and force MAE in kJ mol^-1 angstrom^-1. For the synthetic physical-gate panel, divide both arrays by `96.48533212` to obtain eV-based reference scales `E0` and `F0`; keep the original table units only for the final source-gap check. Use `default_rng` throughout and make no intermediate rounding.

1. With `seed=58031` and 96 configurations, create `rng=default_rng(seed)` and draw `z` once with independent standard-normal entries using `rng.normal(0.0,1.0,size=(96,9,4))`. For zero-based configuration `i` and system `j`, set `e=abs(E0[j]*(1+0.16*z[i,j,0])+0.0003*sin((i+1)*(j+2)/13))` and `f=abs(F0[j]*(1+0.13*z[i,j,1])+0.0012*cos((i+2)*(j+1)/17))`. Whenever `(7*i+11*j+seed) mod 53=0`, add `0.225+0.015*(j mod 3)` to `e` and `10.2+0.3*(j mod 2)` to `f`. The supplied seed controls both the RNG and this selector. These are synthetic relative per-atom energies and force magnitudes in eV-based units, not raw validation errors. Store `[e,f,sqrt((e/E0[j])^2+(f/F0[j])^2),0.55*z2+0.25*z3+0.2*sin((i+1)*(j+1)/7)]`.

2. Apply the source-motivated physical energy and force gates first: energy at most 0.2 and force at most 10.0. Within each system separately, robust-standardize the eligible points in joint energy-force space using that system's median and `1.4826*MAD` per coordinate (scale 1.0 for a zero MAD). Apply DBSCAN with Euclidean epsilon `radius/4`, default radius 3.4, and `min_samples=5` including the point itself. Retain the largest density-connected cluster; break equal-size cluster ties by the smallest original configuration index, and assign a border point reachable from multiple core clusters to the first cluster seeded by original index. Within each system stable-sort the retained configurations by leverage, assign `floor(0.85*n)` to training, and record `[kept,train,validation,mean_e,mean_f,population_covariance,max_robust_radius]`, where the final radius is the maximum scaled Euclidean distance from the eligible-point median among retained configurations.

3. With `seed=8123` and six forms, create one RNG and draw exactly three `(9,6)` arrays in this order: DFT base noise, fine-tuned noise, foundation noise. For system `q` and form `k`, build the DFT landscape `0.33*(k-(q mod 6))^2+0.21*sin((q+1)*(k+2)/3)+0.018*mean_e[q]*(k+1)+Normal(0,0.018)`. Add the second draw with row scale `0.065+0.025*mean_e[q]` to form the fine-tuned landscape and the third draw with row scale `0.24+0.11*mean_f[q]` to form the foundation landscape. Reverse foundation system 4, replace foundation system 6 by its mean plus `0.015*(k-mean(k))`, then subtract each method's own minimum over forms.

4. Against DFT, score the fine-tuned and foundation landscapes with mean absolute error, Kendall tau-a over all unordered form pairs, DFT energy at the predicted minimum, stable top-two overlap, and absolute second-order-statistic gap error. The top-two overlap is the number of indices shared by the two stably sorted lowest-two sets, divided by two; its possible values are 0, 0.5, and 1.

5. Compute the source-defined RMSD15 on actual synthetic periodic crystals, rather than drawing an RMSD proxy. The independent packing stream uses `default_rng(9917)` and 240 frames. Each of the nine systems has 27 molecules, each with three uniquely labeled species in atom order C, N, O, so species-resolved atom pairing has no within-species tie. Enumerate integer offset triples `(a,b,c)` in lexicographic order from `{-1,0,1}^3`, as produced by `meshgrid([-1,0,1],[-1,0,1],[-1,0,1],indexing='ij')` flattened in C order. System `k` has reference fractional molecule centers `mod([0.93,0.07,0.89]+0.16*[a,b,c]+[0.003*k,0.002*k,-0.001*k],1)`, a row-vector reference cell `diag(9.1+0.07*k,8.7+0.04*k,9.4+0.03*k)` in angstrom, and Cartesian molecular motif `[[0,0,0],[0.71,0.21,0.05],[-0.26,0.63,-0.08]]` in angstrom. Reference atom positions are center times cell plus motif, then wrapped to fractional coordinates modulo one. Molecule 13 is the reference shell center. For every frame `t=0..239`, draw a `(3,3)` independent-normal strain matrix with mean zero and standard deviation 0.012, and set the candidate row-vector cell to `reference_cell@(I+strain)`. Let `theta=0.16*sin((t+1)*(k+2)/37)`, and rotate the motif in its xy plane by the usual row-vector 2D rotation matrix `[[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]`. Candidate atom positions are center times candidate cell plus this rotated motif and independent Cartesian noise of shape `(27,3,3)` with mean zero and standard deviation `0.060+0.014*k+0.003*mean_e[k]` angstrom. Wrap candidate positions to fractional coordinates modulo one, then reorder whole molecules by one `rng.permutation(27)` draw. Compute the source-defined RMSD15 on each generated periodic reference/candidate pair, with molecule 13 as the shell center. The synthetic instance has no exact distance or correspondence ties; if one occurs numerically, resolve by original molecule index. Average the 240 RMSD15 values and report the fraction strictly below 0.3 angstrom for each system. These synthetic crystals are an instance benchmark, not the paper's experimental CSD structures.

6. Use a separate `default_rng(9917)` stream for the dynamical benchmark. For each system, consume draws in this exact order and shape: a nominal RMSD-proxy `(240,)` draw from `Normal(0.135+0.012*(k mod 4)+0.02*mean_e[k],0.052)`, orientation `(240,)`, RDF noise `(96,)`, energy `(240,)`, volume `(240,)`. Discard the nominal RMSD-proxy draw; it is a stream-alignment convention for this dynamics benchmark and has no role in Step 5's geometric RMSD15. Set `theta=0.18*sin((t+1)*(k+2)/37)+Normal(0,0.075+0.006*k)` and `P2=0.5*(3*cos(theta)^2-1)`. On `r=linspace(0,6,96)`, use `g_ref=exp(-0.5*((r-(2.2+0.04*k))/0.34)^2)+0.48*exp(-0.5*((r-4.1)/0.55)^2)` and `g_model=g_ref*(1+0.025*sin((k+1)*r))+Normal(0,0.006)`, using composite trapezoidal quadrature on the stated `r` grid for both the integral of `abs(g_model-g_ref)` and the integral of `abs(g_ref)`, and dividing the former by the latter. Use `energy=0.003*sin(2*pi*t/53+k)+Normal(0,0.00045)+(k-4)*2e-6*t` and `volume=1+0.008*sin(2*pi*t/71+0.3*k)+Normal(0,0.0012)+(k-4)*1.5e-6*t`. Let `h=floor(n_frames/2)`. Define each drift as `abs(mean(x[h:])-mean(x[:h]))` for its energy or volume series, then retain `[mean_P2,population_std_P2,RDF_L1,energy_half_drift,volume_half_drift]`.

7. The orchestrator calls Steps 1 through 6, then applies the following disclosed audit conventions. Form eleven graph features per system: curated mean energy, curated mean force, fine-tuned MAE, fine-tuned tau, fine-tuned penalty, mean RMSD15, one minus packing fraction, one minus mean P2, RDF error, energy drift, and volume drift. Robust-standardize each column with `1.4826*MAD` when `MAD>1e-12`, otherwise denominator 1.0. Connect each system to its three nearest other systems, symmetrize the mask, compute `median_positive_distance` from all strictly positive entries of the complete pairwise distance matrix before applying the graph mask, then use `exp(-distance/(median_positive_distance+1e-12))` row-normalized weights, and initialize `h=0.31*z2+0.24*(1-tau)+0.18*z5+0.15*z8+0.12*z10`. For zero-based iteration `q=0..28`, use `msg=W@h`, `gate=sigmoid(0.7*z0-0.45*z3+0.03*(q+1))`, and `h=tanh((0.68-0.09*gate)*h+(0.27+0.11*gate)*msg+0.017*sin((q+1)*(system_index+1)))`, then subtract mean `h`. Keep the nine risks followed by norm, index-weighted checksum, range, weighted-distance checksum, largest eigenvalue, and second-largest eigenvalue of `(W+W.T)/2`, in that order.

Evaluate a `4 by 5 by 4 by 3` tensor over `a0=[0.72,0.86,1,1.14]`, `a1=[0.55,0.75,0.95,1.15,1.35]`, `a2=[0.03,0.07,0.11,0.17]`, and `a3=[0.82,0.94,1.06]`. Let `rank_gap=mean(fine_tuned_MAE+0.5*(1-fine_tuned_tau)+fine_tuned_penalty)` and `dyn_gap=mean(RMSD15+RDF_error+8*energy_drift+20*volume_drift)`. For each scenario initialize `v=risk*a0+(system_index-4)*a2` and zero momentum. For `q=0..40`, use `lap=roll(v,1)-2*v+roll(v,-1)`, `force=tanh(a1*lap+a3*risk-0.13*rank_gap+0.09*dyn_gap)`, `momentum=0.73*momentum+(0.19+0.002*q)*force`, and `v=v+momentum/(1+0.025*q)-0.04*mean(v)`. Score each scenario as `sqrt(mean(v^2))+0.08*max(abs(v))+0.03*abs(sum(v*(system_index+1)))`. Flatten in C order and append median, population standard deviation, 0.9 quantile, range, index-weighted checksum using weights 1 through 240 on the 240 C-order scenario scores, and largest singular value of the C-order `20 by 12` reshape.

Compute the panel checksum from the C-order panel using repeating weights `((flat_index mod 17)+1)`. Let `source_gap=abs(mean(e0)-0.141)+abs(mean(f0)-0.648)` and `J=(log1p(abs(tensor_checksum))/(1+singular_value)+tensor_median+tensor_std+tensor_q90/(1+tensor_range))*(1+tuned_MAE+max(0,0.4-tuned_tau)+tuned_penalty+mean_RMSD15+(1-packing_fraction)+20*max_volume_drift+source_gap)`.

Report the source decisions and empirical validation outcomes relevant to curation, polymorph ranking, RMSD15 packing, and the NVE/NVT/NPT checks, together with J and enough intermediate diagnostics to make the deterministic computation auditable. Round J to eight decimal places.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

build_error_panel

Goal
----
Builds a deterministic nine-system error panel from the paper-order validation metrics.

```python
def build_error_panel(seed: int = 58031, n_configs: int = 96) -> object:
    """Return the deterministic error panel.

    Parameters
    ----------
    seed : int
        Controls both the default_rng stream and the modular outlier selector.
    n_configs : int
        Number of configurations, at least 24.

    Returns
    -------
    object
        A float64 array with shape (n_configs, 9, 4), ordered as energy,
        force, joint normalized error, and leverage.

    Raises
    ------
    ValueError
        If seed is not an integer or n_configs is not an integer at least 24.

    Notes
    -----
    Create rng=default_rng(seed) and draw z once with independent standard-normal
    entries using rng.normal(0.0, 1.0, size=(n_configs, 9, 4)). Use the supplied
    seed in the modular outlier selector.

    Use these fixed source-table values in printed compound order, in
    kJ mol^-1 atom^-1 and kJ mol^-1 A^-1 respectively:
    e0 = [0.128, 0.069, 0.161, 0.159, 0.184, 0.116, 0.146, 0.158, 0.151]
    f0 = [0.762, 0.848, 0.414, 0.501, 1.043, 0.562, 0.695, 0.627, 0.377]
    The order is Benzoic acid, Benzamide, Coumarin, Durene, Isonicotinamide,
    Nicotinic acid, Niacinamide, Pyrazinamide, Resorcinol. Convert these
    reference scales to eV units with 1 eV = 96.48533212 kJ/mol before
    constructing the synthetic per-atom-energy and force-magnitude panel.
    """
    return None
```

### Step 2

curate_active_pool

Goal
----
Applies the paper's energy-force curation method and requested split.

```python
def curate_active_pool(panel: object, radius: float = 3.4, train_fraction: float = 0.85) -> object:
    """Return per-system curation and split diagnostics.

    Parameters
    ----------
    panel : object
        Finite array of shape (n, 9, 4), with n at least 24.
    radius : float
        Positive scale for the specified DBSCAN epsilon, `radius/4`.
    train_fraction : float
        Fraction strictly between zero and one used for the training count.

    Returns
    -------
    object
        A float64 array of shape (9, 7), with columns kept, train,
        validation, mean energy, mean force, population covariance, max robust radius.

    Raises
    ------
    ValueError
        If inputs are nonfinite, shapes or parameters are invalid, or fewer than
        eight configurations survive for a system.

    Notes
    -----
    Within each system, apply the two physical gates before robust scaling and
    density-based filtering. The DBSCAN epsilon is radius/4 and min_samples=5,
    counting each point itself. Use median and 1.4826 times MAD of the physically
    eligible points in that system; if a MAD is zero, use scale 1.0. Retain the
    largest density-connected cluster (tie by smallest original index) and
    stable-sort its indices by leverage before the requested train/validation allocation.
    """
    return None
```

### Step 3

build_relative_landscapes

Goal
----
Builds source-defined relative polymorph landscapes for DFT, fine-tuned, and foundation models.

```python
def build_relative_landscapes(summary: object, seed: int = 8123, n_forms: int = 6) -> object:
    """Return independently zeroed DFT, fine-tuned, and foundation landscapes.

    Parameters
    ----------
    summary : object
        Finite curation summary with shape (9, 7).
    seed : int
        Seed for one default_rng stream.
    n_forms : int
        Number of polymorph forms, at least four.

    Returns
    -------
    object
        Float64 array of shape (9, n_forms, 3), ordered as DFT,
        fine-tuned, and foundation.

    Raises
    ------
    ValueError
        If the summary, seed, or number of forms is invalid.

    Notes
    -----
    For zero-based system q and form k, the quadratic DFT term is
    0.33 * (k - (q % n_forms))**2. The modulus is the supplied n_forms.
    Draw exactly three arrays, in this order: base noise with shape (9, n_forms),
    fine-tuned noise with shape (9, n_forms), and foundation noise with shape
    (9, n_forms). Use the row-dependent scales by broadcasting. Apply the two
    foundation alterations before independently subtracting each method's minimum.
    """
    return None
```

### Step 4

rank_polymorphs

Goal
----
Scores source-defined polymorph ordering and wrong-minimum penalties.

```python
def rank_polymorphs(landscapes: object) -> object:
    """Score two model energy landscapes against DFT.

    Parameters
    ----------
    landscapes : object
        Finite array of shape (9, n_forms, 3), with n_forms >= 4 and columns
        ordered as DFT, fine-tuned, and foundation energies.

    Returns
    -------
    object
        Array of shape (9, 2, 5): MAE, Kendall tau-a, DFT energy at the
        predicted minimum, stable top-two overlap, and gap error. Top-two
        overlap is the shared count between stably sorted lowest-two sets
        divided by two, so it is 0, 0.5, or 1.

    Raises
    ------
    ValueError
        If landscapes is non-finite or has an incompatible shape.
    """
    return None
```

### Step 5

compute_packing_fidelity

Goal
----
Computes geometric packing fidelity for synthetic periodic molecular crystals.

```python
def compute_packing_fidelity(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    """Return mean RMSD15 and its fraction below 0.3 angstrom per system.

    Parameters
    ----------
    summary : object
        Finite curation summary with shape (9, 7).
    seed : int
        Seed for the shared benchmark stream.
    n_frames : int
        Number of frames, at least 60.

    Returns
    -------
    object
        Float64 array of shape (9, 2): mean RMSD15 and fraction below 0.3.

    Raises
    ------
    ValueError
        If the summary, seed, or frame count is invalid.

    Notes
    -----
    The supplied instance-generation convention is defined in the task prompt.
    Use the source's RMSD15 geometry method for each generated frame. This stage
    uses its own RNG stream; Step 6 retains its independently defined stream.
    """
    return None
```

### Step 6

compute_dynamic_stability

Goal
----
Computes the paper's orientational, pair-structure, NVE, and NPT stability diagnostics.

```python
def compute_dynamic_stability(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    """Return five structural and dynamical diagnostics per system.

    Parameters
    ----------
    summary : object
        Finite curation summary with shape (9, 7).
    seed : int
        Seed for the shared benchmark stream. For each system, consume
        RMSD15, orientation, RDF-noise, energy, and volume draws in that order.
    n_frames : int
        Number of frames, at least 60. With h=floor(n_frames/2), each energy
        or volume drift is abs(mean(x[h:]) - mean(x[:h])).

    Returns
    -------
    object
        Float64 array of shape (9, 5) containing mean P2, population standard
        deviation of P2, normalized RDF L1 error, energy drift, and volume drift.

    Raises
    ------
    ValueError
        If summary, seed, or n_frames is invalid.
    """
    return None
```

### Step 7

compute_molcryst_audit

Goal
----
Orchestrates the six preceding public steps and returns the complete molecular-crystal audit.

```python
def compute_molcryst_audit(seed: int = 58031, n_configs: int = 96, rank_seed: int = 8123, dyn_seed: int = 9917) -> object:
    """Call Steps 1 through 6 and return seventeen audit values ending in J.

    Parameters
    ----------
    seed : int
        Error-panel seed, also used by the modular outlier selector.
    n_configs : int
        Number of error-panel configurations, at least 24.
    rank_seed : int
        Relative-landscape seed.
    dyn_seed : int
        Shared packing and dynamics seed.

    Returns
    -------
    object
        Float64 array containing panel checksum, retained count, three ranking
        means, two packing anchors, maximum volume drift, the index-weighted
        checksum of the nine final centered graph risks, six
        tensor anchors, source gap, and J.

    Raises
    ------
    ValueError
        If a seed or n_configs is not an integer, or n_configs is below 24.

    Notes
    -----
    Output index 8 (zero-based) is sum((i+1)*h[i] for i=0..8), where h is
    the nine final centered graph risks. It is not the weighted-distance
    checksum sum(W*distance) also named in the graph diagnostics.
    Use the graph, tensor, checksum, and J conventions stated in the task prompt.
    The panel checksum uses C-order flattening with repeating weights 1 through 17.
    """
    return None
```
