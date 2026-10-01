# Physics-Astrophysics-22

## Background

Black hole perturbation theory studies small deviations from a stationary background spacetime by reducing the linearised field equations to a single master wave equation for an effective one-dimensional scattering problem, with a potential barrier built from the background geometry, the multipole number of the perturbation and the spin of the field carrying it. The Tangherlini family generalises the Schwarzschild solution to $d$ spacetime dimensions and is the natural setting in which to ask how the properties of black hole spectra depend on dimension, a question of long standing interest because higher-dimensional gravity arises in braneworld and string-theoretic scenarios and because varying the dimension exposes which features of four-dimensional black hole physics are generic and which are accidents of $d = 4$.

Non-normal operators, those whose eigenvectors are not mutually orthogonal, can have eigenvalues that are numerically far more sensitive to perturbation than the size of the perturbation alone would suggest; the pseudospectrum formalism makes this sensitivity precise through the eigenvalue condition number, and it has recently been imported from numerical linear algebra into gravitational physics to assess how trustworthy a computed quasinormal or total transmission frequency is once the theory is coupled to an environment or represented only approximately. A frequency with a small condition number is robust: comparable perturbations leave it essentially fixed. A frequency with a large condition number can move by orders of magnitude under a perturbation of modest size, so the two modes cannot be treated as equally reliable predictions even when both solve the same unperturbed eigenvalue problem to the same numerical precision.

## Problem

A total transmission mode of a black hole is a complex-frequency solution of the master perturbation equation that carries the same asymptotic behaviour at the event horizon as at infinity, so that an incident wave crosses the effective potential barrier leaving no reflected component; this is a different family from the quasinormal modes, which are ingoing at the horizon and outgoing at infinity. Because tailored scattering can excite one of these modes selectively, their robustness under perturbations of the evolution operator is a physical question, and the quantity that measures it is the eigenvalue condition number taken in a norm with a physical meaning rather than an arbitrary one.

Work on a $d$-dimensional Tangherlini black hole of horizon radius $r_h = 1$, and take $d = 14$, multipole number $\ell = 2$, and gravitational vector perturbations. Adopt the time coordinate adapted to the left total transmission modes together with the radial coordinate compactified as the ratio of the horizon radius to the areal radius, so that infinity maps to zero and the horizon to one, and rescale the field so that the problem becomes a linear generalised eigenvalue problem in the frequency whose admissible solutions are singled out by regularity at the two ends of the interval, with no boundary condition imposed by hand. Discretise the two members of the pencil on a Chebyshev-Lobatto grid of resolution $N = 200$, that is on $N + 1$ nodes, and normalise the pencil so that the member multiplying the eigenvalue is exactly twice the derivative with respect to the compactified radial coordinate; a condition number is fixed only up to the overall scale of the pencil, and this normalisation pins it.

In this sector exactly one total transmission mode sits on the positive imaginary frequency axis, and it is far better conditioned than the overtones. Compute the condition number of that mode against perturbations of the second-order member of the pencil alone, measured in the energy norm built from the conserved energy of the master field on the slices of the adapted time coordinate, and report it to six decimal places; the answer is graded within $0.00002$. Verify that the value you report is stable under refinement of $N$ rather than drifting with it.

Report also, inside the reasoning: the imaginary part of the frequency of that mode, to eight significant figures; the condition number of the same mode in the $L^2$ norm at the same resolution, to four significant figures; and, for $d = 20$ at the same multipole number and at the same resolution, the imaginary part of the frequency of the corresponding mode to eight significant figures together with its condition number in each of the two norms, to four significant figures.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, and that decimal must be the energy-norm condition number of the mode for $d = 14$ at resolution $N = 200$, not the frequency and not any of the other quantities the paragraph above asks you to report inside the reasoning, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. In <reasoning>, give the construction as well as the numbers, in roughly six hundred words: the two members of the pencil you formed and the coordinate you formed them in, the inner product the norm is taken in and why it is positive definite here, how the left eigenvector entering the condition number is obtained from the ordinary one, why the discretisation you chose is the one whose value settles under refinement, and the scalars requested above. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

step_01_compactified_coefficients

Goal
----
Given a set of points on the compactified radial coordinate together with the spacetime dimension, the multipole number and the spin label, return the compactification weight, its first derivative and the reduced potential at those points, in units where the horizon radius is one, along with the values of the reduced potential at the two ends of the interval.

```python
def compactified_coefficients(
    sigma: np.ndarray,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Evaluate the compactification weight and the reduced potential of a Tangherlini background.

    Parameters
    ----------
    sigma : np.ndarray
        Points on the compactified radial coordinate, in the closed unit interval.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label; zero for scalar and tensor perturbations, two for gravitational vector perturbations.

    Returns
    -------
    dict
        Under the keys weight, weight_derivative, reduced_potential, potential_at_horizon and
        potential_at_infinity.

    Raises
    ------
    ValueError
        When sigma is not a finite one-dimensional array of at least two points inside the closed unit
        interval, when d is not an integer of at least four, when ell is not a non-negative integer, or
        when s is not finite and non-negative.
    """
    return
```

### Step 2

step_02_chebyshev_lobatto_operators

Goal
----
Given a resolution N, return the N + 1 Chebyshev-Lobatto nodes mapped onto the unit interval in increasing order, starting at zero and ending at one, together with the matrix that differentiates nodal values with respect to that coordinate, and the largest absolute row sum of that matrix applied to a constant, which certifies the row-sum construction.

```python
def chebyshev_lobatto_operators(resolution: int) -> dict:
    """Build the Chebyshev-Lobatto nodes and differentiation matrix on the unit interval.

    Parameters
    ----------
    resolution : int
        Grid resolution N; the grid carries N + 1 nodes.

    Returns
    -------
    dict
        Under the keys nodes, derivative and constant_residual.

    Raises
    ------
    ValueError
        When the resolution is not an integer of at least four.
    """
    return
```

### Step 3

step_03_transmission_pencil

Goal
----
Given a resolution, a spacetime dimension, a multipole number and a spin label, assemble the two matrices of the total transmission mode pencil on the Chebyshev-Lobatto grid, using the compactification weight, its derivative and the reduced potential of step 1 together with the grid operators of step 2, in units where the horizon radius is one. The second-order term is expanded by the product rule rather than left in divergence form, so that the operator acts on the interpolating polynomial directly. Return both matrices together with the grid, the differentiation matrix and the smallest singular value of the second matrix, which records that it is singular.

```python
def transmission_pencil(
    resolution: int,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Assemble the operator pencil of the left total transmission mode problem.

    Parameters
    ----------
    resolution : int
        Grid resolution N; the grid carries N + 1 nodes.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.

    Returns
    -------
    dict
        Under the keys operator_a, operator_b, nodes, derivative and
        operator_b_smallest_singular_value.

    Raises
    ------
    ValueError
        When the resolution, dimension, multipole number or spin label is outside the range the
        earlier steps accept.
    """
    return
```

### Step 4

step_04_imaginary_axis_scan

Goal
----
Given the two matrices of the pencil and an interval on the positive imaginary frequency axis, sample the interval uniformly and return the smallest singular value of the shifted pencil at each sample, the imaginary parts at which interior local minima occur in increasing order, and the imaginary part of the deepest such minimum. The pencil is shifted by the eigenvalue equal to the imaginary unit times the frequency. When no interior local minimum exists the list is empty and the reported location is not a number.

```python
def imaginary_axis_scan(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    lower: float,
    upper: float,
    samples: int,
) -> dict:
    """Locate purely imaginary modes of the pencil by sweeping the smallest singular value.

    Parameters
    ----------
    operator_a : np.ndarray
        Second-order member of the pencil.
    operator_b : np.ndarray
        First-order member of the pencil.
    lower : float
        Lower end of the sweep in the imaginary part of the frequency.
    upper : float
        Upper end of the sweep.
    samples : int
        Number of uniformly spaced samples.

    Returns
    -------
    dict
        Under the keys frequencies, smallest_singular_values, minima and location.

    Raises
    ------
    ValueError
        When the matrices are not square, finite and of the same shape with side at least five, when
        the interval is not a finite strictly positive increasing pair, or when the sample count is
        not an integer of at least five.
    """
    return
```

### Step 5

step_05_regular_mode

Goal
----
Given the two matrices of the pencil and a starting guess for the frequency, run two-sided shifted inverse iteration until the correction falls below the requested tolerance or the iteration budget is exhausted, and return the frequency, the right and left eigenvectors in the normalisation described above, the size of the last correction and the number of iterations taken. The pencil is written with the eigenvalue equal to the product of the imaginary unit and the frequency, so the frequency is minus the imaginary unit times the eigenvalue. The iteration starts from the deterministic vector whose entries are the reciprocals of one plus the node index, on both sides.

```python
def regular_mode(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    frequency_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Locate one regular mode of the pencil by two-sided shifted inverse iteration.

    Parameters
    ----------
    operator_a : np.ndarray
        Second-order member of the pencil.
    operator_b : np.ndarray
        First-order member of the pencil.
    frequency_guess : complex
        Starting guess for the frequency, as a complex number. A guess on the imaginary axis at
        height y is the complex number with zero real part and imaginary part y, not the real
        number y.
    tolerance : float
        Relative tolerance on the frequency correction.
    max_iterations : int
        Iteration budget.

    Returns
    -------
    dict
        Under the keys frequency_real, frequency_imag, right_vector, left_vector, correction and
        iterations.

    Raises
    ------
    ValueError
        When the two matrices are not square, finite and of the same shape with side at least five,
        when the guess is not finite, when the tolerance is not finite and strictly positive, or when
        the iteration budget is not a positive integer.
    """
    return
```

### Step 6

step_06_mode_certification

Goal
----
Given a background and a starting guess for the frequency, assemble the pencil of step 3 at two resolutions and solve it at both with the iteration of step 5, then return the refined frequency, the distance between the two frequencies, the relative weight carried by the top quarter of the Chebyshev spectrum of the coarse eigenvector, and the residual of the coarse shifted pencil on that eigenvector. The certification passes when the frequency drift and the spectral tail are both below the requested tolerance. The Chebyshev coefficients are those of the Gauss-Lobatto transform of the nodal values, in which the first and last samples carry half weight.

```python
def mode_certification(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    frequency_guess: complex,
    tolerance: float,
) -> dict:
    """Certify a candidate mode by resolution independence and by spectral decay of its eigenfunction.

    Parameters
    ----------
    resolution : int
        Coarse grid resolution.
    refinement : int
        Increment added to the coarse resolution to form the fine one.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.
    frequency_guess : complex
        Starting guess for the frequency.
    tolerance : float
        Tolerance applied to the drift and to the spectral tail.

    Returns
    -------
    dict
        Under the keys frequency_real, frequency_imag, frequency_drift, spectral_tail, residual and
        certified.

    Raises
    ------
    ValueError
        When the resolution is not an integer of at least eight, when the refinement is not a
        positive integer, when the tolerance is not finite and strictly positive, or when the
        background or the guess is outside the range the earlier steps accept.
    """
    return
```

### Step 7

step_07_energy_gram_matrix

Goal
----
Given a resolution, a spacetime dimension, a multipole number, a spin label and a number of Gauss-Legendre quadrature points, return the Gram matrix of the energy inner product and the Gram matrix of the Lebesgue inner product on the space of nodal values of the Chebyshev-Lobatto grid of step 2, both obtained by exact integration of the Lagrange cardinal polynomials, together with the smallest eigenvalue of the energy Gram matrix. The Lebesgue inner product is one half of the integral of the product of the conjugated fields. The quadrature must be exact for the integrands, which requires at least resolution plus (d + 1) // 2 points.

```python
def energy_gram_matrix(
    resolution: int,
    d: int,
    ell: int,
    s: float,
    quadrature_points: int,
) -> dict:
    """Build the Gram matrices of the energy and Lebesgue inner products by exact integration.

    Parameters
    ----------
    resolution : int
        Grid resolution N; the grid carries N + 1 nodes.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.
    quadrature_points : int
        Number of Gauss-Legendre quadrature points.

    Returns
    -------
    dict
        Under the keys energy_gram, lebesgue_gram and smallest_energy_eigenvalue.

    Raises
    ------
    ValueError
        When the resolution, dimension, multipole number or spin label is outside the range the
        earlier steps accept, or when the quadrature point count is not an integer large enough to
        integrate the entries exactly.
    """
    return
```

### Step 8

step_08_condition_number_in_norm

Goal
----
Given a Gram matrix defining an inner product, the first-order member of the pencil and the right and left eigenvectors of step 4, return the eigenvector adjoint with respect to that inner product, the two norms, the modulus of the pairing and the resulting condition number.

```python
def condition_number_in_norm(
    gram: np.ndarray,
    operator_b: np.ndarray,
    right_vector: np.ndarray,
    left_vector: np.ndarray,
) -> dict:
    """Form the eigenvalue condition number of the pencil in the inner product given by a Gram matrix.

    Parameters
    ----------
    gram : np.ndarray
        Gram matrix of the inner product.
    operator_b : np.ndarray
        First-order member of the pencil.
    right_vector : np.ndarray
        Right eigenvector.
    left_vector : np.ndarray
        Ordinary left null vector of the shifted pencil.

    Returns
    -------
    dict
        Under the keys adjoint_vector, adjoint_norm, right_norm, pairing_modulus and
        condition_number.

    Raises
    ------
    ValueError
        When the matrices are not square, finite and of the same shape, when the Gram matrix is not
        symmetric and positive definite, when either vector is not finite, non-zero and of matching
        length, or when the pairing vanishes.
    """
    return
```

### Step 9

step_09_reported_values

Goal
----
Given the six quantities the problem asks to be reported, return each rounded to the precision the problem requests: the energy-norm condition number of the primary background to six decimal places, its Lebesgue-norm condition number to four significant figures, the imaginary part of its frequency to eight significant figures, and for the comparison background the two condition numbers to four significant figures and the imaginary part of the frequency to eight significant figures.

```python
def reported_values(
    condition_number_energy: float,
    condition_number_lebesgue: float,
    frequency_imag: float,
    comparison_condition_number_energy: float,
    comparison_condition_number_lebesgue: float,
    comparison_frequency_imag: float,
) -> dict:
    """Round the six reported quantities to the precisions the problem requests.

    Parameters
    ----------
    condition_number_energy : float
        Energy-norm condition number of the primary background.
    condition_number_lebesgue : float
        Lebesgue-norm condition number of the primary background.
    frequency_imag : float
        Imaginary part of the frequency of the primary mode.
    comparison_condition_number_energy : float
        Energy-norm condition number of the comparison background.
    comparison_condition_number_lebesgue : float
        Lebesgue-norm condition number of the comparison background.
    comparison_frequency_imag : float
        Imaginary part of the frequency of the comparison mode.

    Returns
    -------
    dict
        Under the keys graded_answer, reported_lebesgue, reported_frequency,
        reported_comparison_energy, reported_comparison_lebesgue and
        reported_comparison_frequency.

    Raises
    ------
    ValueError
        When any argument is not a finite, strictly positive real number.
    """
    return
```

### Step 10

step_10_transmission_mode_conditioning_report

Goal
----
Run the chain end to end for the requested background and resolution, and return the frequency of the mode on the positive imaginary axis, its condition number in the energy norm and in the Lebesgue norm, the frequency and both condition numbers of the corresponding mode at a second spacetime dimension, the energy-norm value recomputed at a raised resolution together with the drift between the two, the smallest eigenvalue of the energy Gram matrix, the location returned by the sweep, the certification flag, and the six reported quantities rounded through step 9.

```python
def transmission_mode_conditioning_report(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    comparison_dimension: int,
    scan_resolution: int,
    scan_lower: float,
    scan_upper: float,
    scan_samples: int,
) -> dict:
    """Run the total transmission mode conditioning chain end to end.

    Parameters
    ----------
    resolution : int
        Grid resolution at which the graded value is computed.
    refinement : int
        Increment used for the resolution drift and the certification.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.
    comparison_dimension : int
        Second spacetime dimension for the cross-dimensional value.
    scan_resolution : int
        Coarse resolution used for the sweep.
    scan_lower : float
        Lower end of the sweep.
    scan_upper : float
        Upper end of the sweep.
    scan_samples : int
        Number of sweep samples.

    Returns
    -------
    dict
        Under the keys graded_answer, reported_lebesgue, reported_frequency,
        reported_comparison_energy, reported_comparison_lebesgue, reported_comparison_frequency,
        frequency_real, frequency_imag, condition_number_energy,
        condition_number_lebesgue, condition_number_comparison,
        condition_number_lebesgue_comparison, frequency_imag_comparison, condition_number_refined,
        refinement_drift, smallest_energy_eigenvalue, scan_location and certified.

    Raises
    ------
    ValueError
        When any argument is outside the range the earlier steps accept, or when the sweep finds no
        mode on the requested interval.
    """
    return
```
