# Material_Science-Molecular_Modeling-68

## Background

Machine-learned interatomic potentials (MLIPs) map atomic configurations to energies and forces using descriptors of local atomic environments, and their quality depends heavily on how structurally diverse their training data is. A key challenge is that standard sampling approaches under-sample slow, collective degrees of freedom, since they either stay near equilibrium or bias only toward regions of high model uncertainty, which does not always coincide with unexplored structural space. One approach treats the MLIP's own descriptor as a general-purpose reaction coordinate, reducing it to a low-dimensional collective variable and biasing dynamics along that variable's under-visited regions. When computing the normalization constant for the on-the-fly density estimate, each reference point's contribution is evaluated using only the other reference points, excluding its own self-kernel term, to avoid artificially inflating the estimated density near every reference point.

## Problem

Standard molecular dynamics trajectories used to train machine-learned interatomic potentials tend to stay trapped near a single free-energy minimum, under-sampling the configurations the potential needs to see. An enhanced-sampling approach addresses this by defining a low-dimensional collective variable directly from the same atomic descriptor a potential would use, then biasing the dynamics along that variable using an on-the-fly probability density estimate built from previously visited reference points.


Your task is to solve one concrete, deterministic instance of this bias-construction pipeline. Given a fixed set of N_ref = 20 previously collected 12-dimensional system-averaged descriptor vectors (seed = 7) and the current configuration's 5 per-atom 12-dimensional descriptors (seed = 11): average the current configuration's per-atom descriptors into a single system-level vector; reduce the descriptor space to a k = 3 dimensional collective-variable space using the leading directions of variation in the 20 reference descriptors; construct an on-the-fly probability density over this reduced space using isotropic Gaussian kernels centered on the projected reference points, with bandwidth σ = 0.3; normalize this density using each reference point's own leave-one-out density estimate (excluding that point's own self-kernel contribution), averaged over all reference points; and evaluate the resulting bias potential at the current configuration's projected point using a barrier parameter ΔE = 15 eV and T = 300 K. Your final answer must be a single number: the value of the bias potential in eV.


Generate the two input arrays with independent fresh NumPy generators, with no earlier draws:

```python

import numpy as np

S_ref = np.random.default_rng(7).standard_normal((20, 12))

G = np.random.default_rng(11).standard_normal((5, 12))

```

Both arrays contain independent standard-normal draws and use NumPy's default float64 dtype. Rows of S_ref are reference descriptors; rows of G are the current configuration's per-atom descriptors.


Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:

- The tags are required. Do not omit them or leave them empty.

- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.

- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.

Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

average_descriptor

Goal
----
Implement average_descriptor, which averages per-atom descriptor vectors into a single system-level descriptor. ERBS treats the machine-learned interatomic potential's own atomic descriptor as a general-purpose reaction coordinate. Rather than biasing individual atoms, the method averages the per-atom descriptor vectors over the whole system to obtain a single system-level descriptor, which is later reduced via PCA into a low-dimensional collective variable.

```python
def average_descriptor(G: np.ndarray) -> np.ndarray:
    '''Average per-atom descriptors into a single system-level descriptor.

    Parameters
    ----------
    G : np.ndarray
        Array of shape (N_atoms, D) containing one D-dimensional descriptor
        vector per atom.

    Returns
    -------
    s_prime : np.ndarray
        Array of shape (D,), the descriptor averaged over all atoms.

    Raises
    ------
    ValueError
        If G is not a 2D array, or if G has zero rows (N_atoms == 0), or if
        G contains any non-finite values (NaN or Inf).
    '''
    return s_prime  # placeholder
```

### Step 2

compute_pca_basis

Goal
----
Implement compute_pca_basis, which computes the PCA mean and truncated projection basis from a set of reference system-averaged descriptors. The reference set of previously collected descriptors is used to find the dominant directions of variation via PCA. Centering removes the mean structural offset; the SVD of the centered matrix gives principal directions ordered by explained variance, of which the leading k are kept to define the reduced collective-variable (CV) space.

```python
def compute_pca_basis(S_ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:

    '''Compute the PCA mean and truncated projection basis from reference descriptors.


    Parameters

    ----------

    S_ref : np.ndarray

        Array of shape (N_ref, D), reference system-averaged descriptors.

    k : int

        Number of principal components to keep (1 <= k <= D).


    Returns

    -------

    mu : np.ndarray

        Array of shape (D,), the empirical mean of S_ref.

    V_k : np.ndarray

        Array of shape (D, k), the first k right-singular vectors of the

        centered reference matrix, ordered by descending singular value.

        Signs are arbitrary. Within a tied singular-value eigenspace, any

        orthonormal basis is accepted. Include null-space directions when

        necessary so that the returned basis has exactly k columns.


    Raises

    ------

    ValueError

        If S_ref is not a 2D array, if S_ref has fewer than 2 rows, if k is

        not an integer with 1 <= k <= D (D = S_ref.shape[1]), or if S_ref

        contains any non-finite values.

    '''

    return mu, V_k  # placeholder
```

### Step 3

project_to_cv

Goal
----
Implement project_to_cv, which projects one or more system-averaged descriptors into the reduced collective-variable (CV) space defined by a PCA mean and basis. This is applied both to the current configuration's descriptor and to the full reference set, since the projected reference points later serve as kernel centers for the density estimate.

```python
def project_to_cv(S: np.ndarray, mu: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    '''Project one or more descriptors into the reduced CV space.

    Parameters
    ----------
    S : np.ndarray
        Array of shape (N, D), one or more descriptor vectors to project.
    mu : np.ndarray
        Array of shape (D,), the PCA mean (from compute_pca_basis).
    V_k : np.ndarray
        Array of shape (D, k), the PCA projection basis (from compute_pca_basis).

    Returns
    -------
    S_cv : np.ndarray
        Array of shape (N, k), the projected collective-variable coordinates.

    Raises
    ------
    ValueError
        If S is not 2D, if mu is not 1D, if V_k is not 2D, if S.shape[1] !=
        mu.shape[0], if mu.shape[0] != V_k.shape[0], or if any input
        contains non-finite values.
    '''
    return S_cv  # placeholder
```

### Step 4

gaussian_kernel

Goal
----
Implement gaussian_kernel, which evaluates an isotropic Gaussian kernel between two points in collective-variable space. ERBS builds its on-the-fly probability density from Gaussian kernels deposited at previously visited collective-variable points, each with equal, uncorrelated variance along every CV diemension.

```python
def gaussian_kernel(s: np.ndarray, s_j: np.ndarray, sigma: float) -> float:
    '''Evaluate an isotropic Gaussian kernel between two points in CV space.

    Parameters
    ----------
    s : np.ndarray
        Array of shape (k,), a point in collective-variable space.
    s_j : np.ndarray
        Array of shape (k,), a kernel center in collective-variable space.
    sigma : float
        Positive kernel bandwidth (standard deviation along each CV dimension).

    Returns
    -------
    value : float
        The kernel value K(s, s_j), as a native Python float.

    Raises
    ------
    ValueError
        If s or s_j is not a 1D array, if s.shape != s_j.shape, if sigma is
        not a finite number > 0, or if s or s_j contains non-finite values.
    '''
    return value  # placeholder
```

### Step 5

density_estimate

Goal
----
Implement density_estimate, which estimates the on-the-fly well-tempered probability density at a collective-variable point as the average of the isotropic Gaussian kernel over all previously visited reference points. This kernel density estimate is later compared, through a normalization constant, against the density at the reference points themselves to build the bias potential.

```python
def density_estimate(s: np.ndarray, S_cv: np.ndarray, sigma: float) -> float:
    '''Estimate the well-tempered probability density at a CV point.

    Parameters
    ----------
    s : np.ndarray
        Array of shape (k,), the collective-variable point to evaluate the
        density at.
    S_cv : np.ndarray
        Array of shape (N_ref, k), the reference collective-variable points
        (kernel centers).
    sigma : float
        Positive kernel bandwidth, shared by all kernels.

    Returns
    -------
    density : float
        The estimated density p_n^WT(s), as a native Python float.

    Raises
    ------
    ValueError
        If s is not 1D, if S_cv is not 2D, if S_cv has zero rows, if
        s.shape[0] != S_cv.shape[1], or if sigma is not a finite number > 0.
    '''
    return density  # placeholder
```

### Step 6

normalization_constant

Goal
----
Implement normalization_constant using the task's leave-one-out convention.

For N reference CV points, compute



    Z_n = [1 / (N * (N - 1))] * sum_j sum_{i != j} K(s_j, s_i).



Each inner density uses the other N - 1 centers, and the outer mean uses all N

centers. K is the same isotropic Gaussian density kernel used by gaussian_kernel.

```python
def normalization_constant(S_cv: np.ndarray, sigma: float) -> float:

    '''Compute the mean leave-one-out density over the reference CV points.


    Parameters

    ----------

    S_cv : np.ndarray

        Array of shape (N_ref, k), the reference collective-variable points

        (kernel centers).

    sigma : float

        Positive kernel bandwidth, shared by all kernels.


    Returns

    -------

    Z_n : float

        The mean of the N leave-one-out densities. Each density averages

        exactly N - 1 non-self kernels. Returned as a native Python float.


    Raises

    ------

    ValueError

         If S_cv is not a 2D array, if S_cv has fewer than 2 rows (leave-one-out

         requires at least one other point to average against), or if sigma is

         not a finite number > 0.

    '''

    return Z_n  # placeholder
```

### Step 7

thermo_params

Goal
----
Implement thermo_params, which computes the inverse thermal energy beta, the barrier ratio gamma, and the bias-limiting factor epsilon from a temperature and a barrier parameter Delta E. These enter the final bias-potential formula to impose a soft ceiling on its own strength.

```python
def thermo_params(T: float, delta_E: float) -> tuple[float, float, float]:
    '''Compute beta, gamma, and epsilon from temperature and barrier energy.

    Parameters
    ----------
    T : float
        Temperature in Kelvin, must be > 0.
    delta_E : float
        Barrier parameter Delta E in eV, must be > 0.

    Returns
    -------
    beta : float
        Inverse thermal energy, 1 / (k_B * T), in eV^-1, where k_B is the
        Boltzmann constant in eV/K (8.617333262e-5 eV/K).
    gamma : float
        beta * delta_E (dimensionless).
    epsilon : float
        exp(-gamma / (gamma - 1)) (dimensionless).

    Raises
    ------
    ValueError
        If T is not a finite number > 0, if delta_E is not a finite number
        > 0, or if gamma is within 1e-9 of 1.0 (division by zero in epsilon).
    '''
    return beta, gamma, epsilon  # placeholder
```

### Step 8

bias_potential

Goal
----
Implement bias_potential, the final orchestrator that chains all prior sub-problems into the full ERBS bias-construction pipeline: average the current configuration's per-atom descriptors, build a PCA basis from the reference descriptors, project both the current configuration and the reference set into the reduced collective-variable space, estimate the well-tempered density and its normalization constant, compute the thermodynamic barrier parameters, and combine them into the final scalar bias potential.

```python
def bias_potential(
    G: np.ndarray,
    S_ref: np.ndarray,
    k: int,
    sigma: float,
    T: float,
    delta_E: float,
) -> float:
    '''Compute the ERBS bias potential for a configuration (end-to-end orchestrator).

    Parameters
    ----------
    G : np.ndarray
        Array of shape (N_atoms, D), per-atom descriptors of the current
        configuration.
    S_ref : np.ndarray
        Array of shape (N_ref, D), previously collected reference
        system-averaged descriptors.
    k : int
        Number of principal components to keep (1 <= k <= D).
    sigma : float
        Positive Gaussian kernel bandwidth in CV space.
    T : float
        Temperature in Kelvin, must be > 0.
    delta_E : float
        Barrier parameter Delta E in eV, must be > 0.

    Returns
    -------
    V_n : float
        The bias potential value in eV, as a native Python float.

    Raises
    ------
    ValueError
        If any of the earlier steps' input validity conditions are violated
        (see average_descriptor, compute_pca_basis, project_to_cv,
        density_estimate, normalization_constant, and thermo_params for the
        exact conditions each enforces).
    '''
    return V_n  # placeholder
```
