# Mathematics-Computational_Mechanics-16

## Background

Computational homogenisation solves a cell problem for the local fields induced by a uniform loading and averages the resulting flux to obtain the effective property. Fourier-based methods never assemble a global matrix: the constitutive law is applied pointwise on a regular grid and a uniform comparison medium supplies a Green operator that is diagonal in frequency space. The comparison medium is not a physical choice but a preconditioner, and the resulting integral equation involves a bounded self-adjoint operator whose eigenvalues are confined to an interval fixed by the extreme conductivities of the phases and the reference conductivity.

The eigenstates of that operator are the eigenstates of the composite: some are confined to one phase, others live on the interfaces between phases, and the two families play very different roles in the construction of the cell solution and of the effective property. Keeping only the eigenstates that matter yields a reduced model whose accuracy improves rapidly with the number of retained eigenstates, and the same spectral data describe how the residual of the classical fixed-point iteration decays. For a two-phase composite the eigenvalue problem can be rewritten so that the microstructure enters through its indicator function alone, the contrast between the phases appearing only in a scalar, so that one spectral decomposition provides the effective property for every contrast.

## Problem

Fourier-based solvers for periodic homogenisation converge because the introduction of a uniform comparison medium turns the cell problem into a preconditioned one, and the spectrum of the preconditioned operator governs both the convergence rate of the iterative schemes and the composition of the solution itself. Expanded on the eigenstates of that operator, the cell solution is carried by a small part of the spectrum, which makes a reduced model possible in which the local field and the effective property are reconstructed from a limited number of eigenstates. Your task is to build that reduced model for a two-dimensional periodic three-phase conductor, to quantify how much of the spectrum takes part in the solution, to check the spectral description against a two-phase reduction in which the geometry and the material contrast separate, and to report one dimensionless number, the relative error of the reduced effective conductivity along the loading direction.

The cell $\Omega = [0, L_1) \times [0, L_2)$ carries a scalar conductivity $C(x)$ and is subjected to a uniform macroscopic gradient $E$. The fluctuation $u$ is the periodic zero-mean potential satisfying $\nabla \cdot [C (E + \nabla u)] = 0$, and the effective conductivity is defined through the average flux, $C_{eff} E = \langle C (E + \nabla u) \rangle$, where $\langle \cdot \rangle$ denotes the average over the cell. With a uniform reference conductivity $C_0$, let $A v = -\nabla \cdot (C \nabla v)$ and $A_0 v = -C_0 \Delta v$ act on zero-mean periodic potentials and let $S = I - A_0^{-1} A$ be the preconditioned operator. This operator is self-adjoint for the energetic inner product $(v_1, v_2) = \langle \nabla v_1 \cdot C_0 \nabla v_2 \rangle$, and its eigenstates $\phi_j$, normalised so that $(\phi_j, \phi_j) = 1$ and ordered by ascending eigenvalue $\lambda_j$, form a basis in which the fluctuation reads $u = \sum_j p_j \phi_j$ with components $p_j = (u, \phi_j)$. The reduced model admits the eigenstates in decreasing order of $|p_j|$ and stops at the smallest count $N_{tr}$ for which the relative error of the truncated fluctuation in the energetic norm, $\| u_{tr} - u \| / \| u \|$ with $\| v \|^2 = (v, v)$, falls strictly below a threshold $\tau$; the reduced effective flux is the average flux of the truncated field, $\langle C (E + \nabla u_{tr}) \rangle$.

Use the following deterministic configuration:

- cell $[0, 1)^2$, periodic in both directions, sampled at $n = 45$ points per axis located at $x = (i / 45, j / 45)$ for $i, j = 0, \ldots, 44$
- phase 1 is the matrix with $C_1 = 1$; phase 2 has $C_2 = 3$; phase 3 has $C_3 = 10$
- four circular inclusions: centre $(0.27, 0.31)$ radius $0.24$ phase 3; centre $(0.76, 0.72)$ radius $0.21$ phase 3; centre $(0.78, 0.22)$ radius $0.155$ phase 2; centre $(0.24, 0.79)$ radius $0.135$ phase 2; a sample point belongs to a disc when its periodic minimum-image distance to the centre is strictly below the radius; the discs do not overlap and no sample point lies within $10^{-4}$ of a disc boundary
- reference conductivity $C_0 = (\min_p C_p + \max_p C_p) / 2 = 5.5$
- discretisation: a field is represented by the trigonometric interpolant through its sample values, with integer frequencies $-22, \ldots, 22$ along each axis; the gradient is the exact derivative of that interpolant, the conductivity multiplies the gradient pointwise at the sample points, and the divergence is again taken spectrally; $\langle \cdot \rangle$ is the arithmetic mean over the $2025$ sample points, and the fluctuation is periodic with zero mean
- macroscopic gradient $E = (1, 0)$
- the spectrum accumulates on three values; an eigenvalue is counted on an accumulation value when it lies within $10^{-4}$ of it
- truncation threshold $\tau = 0.1$
- two-phase reduction: phases 2 and 3 are merged into a single inclusion phase of conductivity $z C_1$ and the reference conductivity is set equal to the inclusion conductivity; the eigenvalue problem then separates into a problem that depends on the geometry alone, with eigenvalues $\mu_j$ in $[0, 1]$, and a scalar function of the contrast $z$; with the geometric eigenfunctions $\psi_j$ normalised to $\langle | \nabla \psi_j |^2 \rangle = 1$ and $\chi$ the indicator function of the matrix, the normalised projection of the loading on $\psi_j$ is $E \cdot \langle \chi \nabla \psi_j \rangle$; an eigenvalue $\mu_j$ is counted at an end of $[0, 1]$ when it lies within $10^{-4}$ of it; evaluate the two-phase effective conductivity at $z = 10$
- IEEE float64 arithmetic

State in your reasoning the number of sample points in each phase; the three accumulation values with the number of eigenvalues within $10^{-4}$ of each; the full effective conductivity $(C_{eff})_{11}$; the largest component magnitude $\max_j |p_j|$ and the eigenvalue of the eigenstate that carries it; the retained count $N_{tr}$ and the energetic-norm error it achieves; the reduced effective conductivity $(C_{tr})_{11}$; and, for the two-phase reduction, the number of geometric eigenvalues within $10^{-4}$ of zero and of one, the largest normalised projection magnitude, and the effective conductivity $(C_{eff})_{11}$ at $z = 10$ obtained from the geometric spectrum. Report as the final answer the relative error $|(C_{tr})_{11} - (C_{eff})_{11}| / (C_{eff})_{11}$ to at least six significant figures; it is graded to a relative tolerance of $10^{-4}$.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_rasterise_periodic_disc_microstructure

Goal
----
A rectangular periodic cell is sampled on a regular grid with an odd number of pixels along each axis, the sample points being x_d = i L_d / n for i = 0, ..., n - 1, so that the origin is a sample point and no sample point lies on the far edge of the cell. The odd count matters for everything that follows: the trigonometric interpolant through the samples then has an unambiguous frequency set -m, ..., m with n = 2 m + 1 and no unpaired highest frequency, which is what makes the spectral derivative used later a real symmetric operator. The microstructure is a matrix carrying circular inclusions; each disc is given by its centre, its radius and its phase label, and a sample point belongs to a disc when its periodic minimum-image distance to the centre is strictly below the radius, so discs that straddle a cell edge wrap around. Phase 1 is the matrix and labels 2 and above are inclusions; the conductivity map is the phase conductivity looked up pixel by pixel. The step also returns the pixel counts, the volume fractions and the arithmetic-mean conductivity, which is the Voigt bound the effective conductivity must not exceed.

```python
def rasterise_periodic_disc_microstructure(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
) -> dict:
    """Sample a periodic matrix-inclusion microstructure on an odd pixel grid.

    Raises
    ------
    ValueError
        If n_pixels is not an odd integer of at least three, if a cell length is not strictly positive, if a disc radius is not strictly positive, if a disc phase label is below two or above the number of phases, if a phase conductivity is not strictly positive, or if a sample point lies within 1e-9 of a disc boundary.
    """
    return
```

### Step 2

02_apply_periodic_conductivity_operator

Goal
----
The cell operator of periodic conductivity homogenisation maps a periodic potential v to A v = -div(C grad v). On the odd sample grid it is discretised in the standard Fourier fashion: the gradient is the exact derivative of the trigonometric interpolant, that is multiplication by 2 pi i k_d / L_d for every frequency k_d in -m, ..., m, the conductivity multiplies the gradient pointwise at the sample points, and the divergence is again spectral. Because the frequency set is symmetric and the conductivity is real, the discrete operator is real and symmetric with respect to the pixel-average inner product, and its bilinear form is the average of grad v_1 . C grad v_2 over the sample points, exactly as in the continuous weak form. The result of the divergence has zero frequency component equal to zero by construction, so the operator maps into mean-free fields and its only null vector on mean-free periodic fields is absent; on all periodic fields the constants are its kernel. The function accepts a batch of fields so that the operator can later be applied to a whole basis at once. Conductivity values may be zero, which is what allows the same routine to serve for an indicator function later on.

```python
def apply_periodic_conductivity_operator(
    field: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
) -> np.ndarray:
    """Apply the Fourier-discretised conductivity operator to one or more fields.

    Raises
    ------
    ValueError
        If the last two axes of field are not square with an odd size of at least three, if conductivity does not have that shape or contains a negative or non-finite value, or if a cell length is not strictly positive.
    """
    return
```

### Step 3

03_assemble_mean_free_operator_matrices

Goal
----
The cell problem is posed on periodic potentials with zero mean, a space of dimension N - 1 for N sample points, and the spectral analysis needs matrix representations of two operators on that space: the cell operator A v = -div(C grad v) and the reference operator A_0 v = -C_0 Laplacian v of a uniform comparison medium. Both matrices are formed in the same real orthonormal basis of mean-free fields, orthonormal for the Euclidean inner product over the sample points, and their entries are the pixel averages of q_i times the operator image of q_j, so that a coefficient vector c represents the field sum_i c_i q_i and c_1 . [A] c_2 equals the average of grad v_1 . C grad v_2. The basis used here is trigonometric: for every frequency pair in a half-space of the symmetric frequency set, the normalised cosine and sine modes. In that basis the reference matrix is diagonal with entries C_0 (2 pi)^2 |k / L|^2 / N, positive because the zero frequency has been excluded, so it defines an inner product on coefficient vectors; the cell matrix is real symmetric and reduces to the reference matrix when the conductivity is uniform and equal to C_0. Any other orthonormal mean-free basis represents the same operators up to an orthogonal change of coordinates, which leaves every generalised eigenvalue, every trace and every reconstructed field unchanged.

```python
def assemble_mean_free_operator_matrices(
    conductivity: np.ndarray,
    cell_lengths: tuple,
    reference_conductivity: float,
) -> dict:
    """Represent the cell operator and the reference operator on mean-free periodic fields.

    Raises
    ------
    ValueError
        If conductivity is not a square array of odd size at least three with finite non-negative entries, if a cell length is not strictly positive, or if reference_conductivity is not strictly positive.
    """
    return
```

### Step 4

04_solve_preconditioned_cell_operator_spectrum

Goal
----
Preconditioning the cell operator by the reference operator produces S = I - A_0^{-1} A on mean-free periodic potentials. This operator is bounded and self-adjoint for the energetic inner product (v_1, v_2) = average of grad v_1 . C_0 grad v_2, and its eigenpairs are those of the generalised problem A phi = (1 - lambda) A_0 phi. With the matrices of the previous step the problem is a real symmetric definite pencil, solved so that the eigenvectors are orthonormal for the reference matrix, which is the discrete form of the energetic normalisation. The eigenvalues are returned in ascending order. The spectrum lies in the interval [1 - M / C_0, 1 - m / C_0], with m and M the smallest and largest phase conductivities, because outside that interval the shifted conductivity C + (lambda - 1) C_0 is definite of one sign and the eigenvalue equation has no non-trivial solution. Inside the interval the spectrum accumulates on the values 1 - C_p / C_0, one per phase: those eigenstates fluctuate within phase p and are constant elsewhere, and they play no role in the cell solution, whereas the remaining eigenvalues, which lie in the transitions between the accumulation values, belong to eigenstates with non-trivial behaviour at the interfaces. The step counts the eigenvalues within a stated tolerance of each accumulation value and the eigenvalues that are farther than that tolerance from all of them.

```python
def solve_preconditioned_cell_operator_spectrum(
    operator: np.ndarray,
    reference_operator: np.ndarray,
    phase_conductivities: np.ndarray,
    reference_conductivity: float,
    plateau_tolerance: float,
) -> dict:
    """Solve the generalised eigenvalue problem of the preconditioned cell operator and classify its spectrum.

    Raises
    ------
    ValueError
        If operator and reference_operator are not square matrices of the same shape, if either differs from its transpose by more than 1e-8 in absolute value, if reference_operator is not positive definite, if a phase conductivity or reference_conductivity is not strictly positive, or if plateau_tolerance is not strictly positive.
    """
    return
```

### Step 5

05_project_cell_load_onto_eigenstates

Goal
----
Under a uniform macroscopic gradient E the fluctuation u of the potential solves A u = b with load b = div(C E), the divergence of the flux that the macroscopic gradient alone would produce. Preconditioning gives (I - S) u = A_0^{-1} b, and expanding u on the eigenstates, which are orthonormal for the energetic inner product, yields the components p_j = (A_0^{-1} b, phi_j) / (1 - lambda_j). The numerator is simply the pixel average of b phi_j, and integrating by parts once more turns it into -E . beta_j with beta_j the average of C grad phi_j, a vector attached to each eigenstate that measures how strongly it couples to a uniform loading; the same vectors govern the spectral form of the effective conductivity in the next step. Both routes are computed here and the largest discrepancy between them is returned, a check that the load, the eigenvectors, the normalisation and the spectral derivative are mutually consistent. Eigenstates confined to a single phase have beta_j equal to zero, since the average of C grad phi_j reduces to C_p times the average of grad phi_j over the cell, which vanishes by periodicity, so only eigenstates with non-trivial interface behaviour receive a component. The dominant eigenstate, the one with the largest component magnitude, is identified along with its eigenvalue.

```python
def project_cell_load_onto_eigenstates(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Expand the cell load on the eigenstates and form the spectral coupling vectors.

    Raises
    ------
    ValueError
        If the array shapes are mutually inconsistent, if any eigenvalue is not strictly below one, if macroscopic_gradient does not have two components, or if a cell length is not strictly positive.
    """
    return
```

### Step 6

06_reconstruct_solution_and_effective_tensor

Goal
----
With every component known, the fluctuation is the sum of p_j phi_j over the whole spectrum and its sample values follow by combining the basis fields. Because the eigenstates form a complete orthonormal set for the energetic inner product on the discrete mean-free space, this sum is the exact discrete solution of the cell problem, and the step verifies that by evaluating the equilibrium residual div(C (E + grad u)) at the sample points relative to the largest sample value of the load, or in absolute terms when the load vanishes as it does for a uniform medium. The effective conductivity is defined through the average flux, C_eff E = average of C (E + grad u); for the exact solution this agrees with the energy definition, and because the eigenstates diagonalise the cell operator the two definitions also agree for any expansion truncated to a subset of eigenstates. Inserting the expansion into the flux average gives the spectral form C_eff = average of C plus the sum over j of beta_j beta_j / (lambda_j - 1), a rank-one correction per eigenstate whose weight is the outer product of the spectral vector divided by the shifted eigenvalue. Every term of the correction is negative semi-definite because each eigenvalue lies below one, so the effective conductivity never exceeds the arithmetic mean. The step returns the flux-average column for the given loading, the full spectral tensor, the discrepancy between the two for that loading, and the energetic norm of the fluctuation, which by orthonormality is the root of the sum of the squared components.

```python
def reconstruct_solution_and_effective_tensor(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Rebuild the fluctuation from its eigenstate expansion and form the effective conductivity.

    Raises
    ------
    ValueError
        If the array shapes are mutually inconsistent, if any eigenvalue is not strictly below one, if macroscopic_gradient does not have two components, or if a cell length is not strictly positive.
    """
    return
```

### Step 7

07_build_truncated_reduced_model

Goal
----
Only a small part of the spectrum carries the cell solution, so a reduced model keeps the eigenstates whose components are largest in magnitude and discards the rest. Because the eigenstates are orthonormal for the energetic inner product, the relative error of the truncated fluctuation in the energetic norm is available without forming any field: it is the root of the tail sum of squared components over the total sum. The number of retained eigenstates is therefore chosen as the smallest count for which that error falls strictly below a prescribed threshold when eigenstates are admitted in order of decreasing component magnitude. The reduced effective response is the average flux of the truncated field, average of C times E plus the sum of p_j beta_j over the retained set, which is also what the spectral form gives when the sum is restricted to that set and, since the eigenstates diagonalise the cell operator, what the energy of the truncated field gives as well. The relative error of each component of the reduced flux against the full flux is returned, together with the retained count, the achieved energetic error, the error that one fewer eigenstate would leave, and the retained fraction of the spectrum.

```python
def build_truncated_reduced_model(
    eigenvalues: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    mean_conductivity: float,
    effective_column: np.ndarray,
    macroscopic_gradient: np.ndarray,
    energy_error_threshold: float,
) -> dict:
    """Select the dominant eigenstates by an energetic error threshold and form the reduced effective flux.

    Raises
    ------
    ValueError
        If the array shapes are mutually inconsistent, if every component is zero, if mean_conductivity is not strictly positive, if macroscopic_gradient or effective_column does not have two components, or if energy_error_threshold is not strictly between zero and one.
    """
    return
```

### Step 8

08_compute_contrast_independent_representation

Goal
----
For a two-phase composite the spectral analysis separates the geometry from the material contrast. Writing the conductivity as C_m chi + C_i (1 - chi) with chi the indicator of the matrix, choosing the reference medium equal to the inclusion conductivity C_i, and introducing the contrast z = C_i / C_m together with s = z / (z - 1), the generalised eigenvalue problem of the preconditioned operator becomes div(chi grad psi) = mu Laplacian psi with mu = s lambda. This problem involves the indicator alone, its eigenvalues lie in [0, 1] because chi takes the values zero and one, and it accumulates on the two ends of that interval: eigenstates fluctuating only inside the inclusions have mu near zero and those fluctuating only in the matrix have mu near one. Normalising the eigenstates to unit mean squared gradient, the coupling of each to a macroscopic gradient E is the normalised projection E . g_j with g_j the average of chi grad psi_j, which does not depend on the contrast either. The effective flux for any contrast then follows from the geometric data alone: average of C times E plus (C_i - C_m) times the sum over j of g_j (g_j . E) / (mu_j - s). The step solves the geometric problem, counts the eigenvalues within a tolerance of each end of the interval, evaluates the representation at the requested contrast and compares it with a direct solution of the cell problem at that contrast.

```python
def compute_contrast_independent_representation(
    phase: np.ndarray,
    cell_lengths: tuple,
    matrix_phase: int,
    matrix_conductivity: float,
    contrast: float,
    macroscopic_gradient: np.ndarray,
    limit_tolerance: float,
) -> dict:
    """Solve the geometric eigenvalue problem of a two-phase cell and evaluate its contrast-independent spectral representation.

    Raises
    ------
    ValueError
        If phase is not a square integer array of odd size at least three, if matrix_phase does not occur in phase or every sample point carries it, if matrix_conductivity is not strictly positive, if contrast is not strictly positive or equals one, if macroscopic_gradient does not have two components, if a cell length is not strictly positive, or if limit_tolerance is not strictly positive.
    """
    return
```

### Step 9

09_compute_reduced_model_conductivity_error

Goal
----
This final step runs the whole spectral reduced-model workflow from the raw configuration and reports the relative error of the reduced effective conductivity together with every check quantity. The chain is: step 1 samples the periodic disc microstructure on the odd grid and returns the phase map and the conductivity; step 2 provides the Fourier-discretised cell operator, applied inside step 3 to an orthonormal mean-free basis to produce the matrices of the cell operator and of the reference operator, the reference conductivity being the mean of the smallest and largest phase conductivities; step 4 solves the generalised eigenvalue problem of the preconditioned operator with energetic normalisation, returning the ascending spectrum, its bounds, the accumulation values and the counts on and off them; step 5 expands the load on the eigenstates, yielding the components and the spectral coupling vectors; step 6 reconstructs the exact discrete solution, verifies equilibrium and forms the effective conductivity by the flux average and by the spectral sum; step 7 retains the eigenstates of largest component magnitude until the energetic-norm error of the truncated fluctuation drops below the threshold and evaluates the reduced flux; step 8 merges every inclusion phase into one, solves the geometric eigenvalue problem of the resulting two-phase cell and evaluates its contrast-independent representation at the requested contrast against a direct solution. The reported final quantity is the relative error of the first component of the reduced effective flux against the full one for the given loading.

```python
def compute_reduced_model_conductivity_error(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
    macroscopic_gradient: np.ndarray,
    plateau_tolerance: float,
    energy_error_threshold: float,
    two_phase_contrast: float,
) -> dict:
    """Run the spectral reduced-model workflow end to end and report its relative conductivity error.

    Raises
    ------
    ValueError
        If n_pixels is not an odd integer of at least three, if a cell length or a phase conductivity is not strictly positive, if fewer than two phases are given, if macroscopic_gradient does not have two components, if plateau_tolerance is not strictly positive, if energy_error_threshold is not strictly between zero and one, or if two_phase_contrast is not strictly positive or equals one.
    """
    return
```
