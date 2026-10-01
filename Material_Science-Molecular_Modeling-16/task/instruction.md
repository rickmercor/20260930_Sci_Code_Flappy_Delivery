# Material_Science-Molecular_Modeling-16

## Background

Tensorial MKCT reconstructs basis correlations from bath-weighted moments and auxiliary memory kernels. A finite closure replaces the unresolved hierarchy. Here the system Hamiltonian produces nonzero drift, so the order of matrix factors affects the population.

The finite bath, Padé order and time discretization are specified for this task. The operator definition fixes the row convention when deriving kernel initial values.

## Problem

Compute the population of the +1 eigenstate of sigma_z at time 0.8. Use system-first tensor ordering and

$$
\hbar=1,\qquad \phi=(I,\sigma_x,\sigma_y,\sigma_z)/\sqrt{2}.
$$

The Hamiltonian is

$$
H_s=0.13\sigma_x+0.07\sigma_y+0.19\sigma_z,
\qquad H_b=\operatorname{diag}(0,0.7,1.3),
$$

$$
B_x=
\begin{pmatrix}
0&0.23&0.11\\
0.23&0&0.17\\
0.11&0.17&0
\end{pmatrix},
\qquad
B_z=
\begin{pmatrix}
0&0.13&-0.19\\
0.13&0&0.07\\
-0.19&0.07&0
\end{pmatrix},
$$

$$
H=H_s\otimes I_3+I_2\otimes H_b
+\sigma_x\otimes B_x+\sigma_z\otimes B_z.
$$

Start from

$$
\rho(0)=
\frac{I_2+0.3\sigma_x-0.4\sigma_y+0.5\sigma_z}{2}
\otimes\rho_b,
\qquad
\rho_b=\frac{e^{-1.4H_b}}{\operatorname{Tr}(e^{-1.4H_b})}.
$$

Use tensorial MKCT with moments through order 12. The evolved operator labels the row:

$$
D(X)=i[H,X],
\qquad
C_{ij}(t)=
\operatorname{Tr}[
(e^{tD}(\phi_i\otimes I_3))
(\phi_j\otimes\rho_b)
].
$$

Define the auxiliary kernels from the projection in step 2. Derive their initial values in this index convention; matrix factors need not commute. Retain the nonzero drift throughout.

Close the third kernel with an elementwise [4/4] Padé approximation matching Taylor coefficients through degree eight. Normalize each denominator to constant coefficient one. Use minimum-norm least squares with relative singular-value cutoff 1e-12.

Propagate the first two kernels by classical RK4 on 256 equal intervals. Evaluate the closure at every stage. Integrate the GQME using trapezoidal quadrature for both the causal convolution and the outer time integral. Include the drift and convolution endpoint in the implicit solve. Start from identity correlation and the derivative fixed by the GQME.

Return the population from this discretization, including initial coherences. Direct Hamiltonian propagation is not the requested result.

Cite the tensorial method. Show the drift matrix, the kernel initialization convention, and the final z row of the correlation matrix.

Output format:

Emit <final_answer>...</final_answer> followed by <reasoning>...</reasoning>.

Put one finite decimal in the first tag. In the reasoning, cite the method and report the drift, ordered kernel initialization and final z row.

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

01_compute_moments

Goal
----
Compute bath-weighted operator moments through the requested order in the normalized Pauli basis (I, X, Y, Z)/sqrt(2). Use system-first tensor-product ordering.

```python
def compute_moments(h, hb, beta, order):
    """Compute bath-weighted operator moments.

    Parameters
    ----------
    h : array_like, complex, shape (2*b, 2*b)
        Hermitian total Hamiltonian, system-first tensor ordering.
    hb : array_like, complex, shape (b, b)
        Hermitian bath Hamiltonian; b >= 1.
    beta : float
        Finite nonnegative inverse temperature.
    order : int
        Highest moment order, from 0 through 12.

    Returns
    -------
    ndarray, real, shape (order + 1, 4, 4)
        Moments in basis (I, X, Y, Z)/sqrt(2), with
        Omega[n,i,j] = Tr[D**n(phi_i tensor I)(phi_j tensor rho_b)]
        and D(A) = 1j*(h@A - A@h).

    Raises
    ------
    ValueError
        If shapes are incompatible, a Hamiltonian is nonfinite or
        non-Hermitian at absolute tolerance 1e-12, beta is nonfinite
        or negative, or order is not an integer in [0, 12].
    """
    return np.zeros((order + 1, 4, 4))
```

### Step 2

02_kernel_coefficients

Goal
----
The row index labels the initial operator. Define



$$

PX=

\sum_j

\operatorname{Tr}[X(\phi_j\otimes\rho_b)]

(\phi_j\otimes I_b),

\qquad Q=1-P,

$$



$$

f_i(t)=e^{tQD}QD(\phi_i\otimes I_b),

$$



$$

K_{n,ij}(t)=

\operatorname{Tr}[D^n f_i(t)(\phi_j\otimes\rho_b)].

$$



Derive the initial kernels from this definition. The hierarchy is



$$

K_n'(t)=K_{n+1}(t)-K_1(t)\Omega_n.

$$



Differentiate it to obtain the requested Taylor coefficients. Inputs satisfying the signature are admissible moment arrays; no Hamiltonian-realizability test is required.

```python
def kernel_coefficients(moments):
    """Return third-kernel Taylor coefficients in the stated row convention.

    Parameters
    ----------
    moments : array_like, real, shape (13, 4, 4)
        Omega_0 through Omega_12. Omega_0 is identity within 1e-10
        entrywise. Omega_1 may be any finite matrix; retain it.

    Returns
    -------
    ndarray, real, shape (9, 4, 4)
        Entry m is the m-th derivative of K_3 at zero divided by m!,
        for m = 0,...,8, using the operator definition in the background.

    Raises
    ------
    ValueError
        If shape is incorrect, any entry is nonfinite, or Omega_0
        differs from identity by more than 1e-10 entrywise.
    """
    import numpy as np
    return np.zeros((9, 4, 4))
```

### Step 3

03_fit_closure

Goal
----
Fit an elementwise [4/4] Padé closure to the third-kernel Taylor coefficients. Return numerator and denominator coefficients in ascending powers.

```python
def fit_closure(coefficients):
    """Fit the elementwise [4/4] third-kernel closure.

    Parameters
    ----------
    coefficients : array_like, real, shape (9, 4, 4)
        Taylor coefficients c_0 through c_8, not raw derivatives.

    Returns
    -------
    ndarray, real, shape (2, 5, 4, 4)
        Numerator and denominator coefficients in ascending powers.
        Every denominator has constant coefficient one.
        Solve coefficient matching by minimum-norm least squares
        with relative singular-value cutoff 1e-12.

    Raises
    ------
    ValueError
        If shape is incorrect, an entry is nonfinite, or a scalar
        matching system has maximum residual greater than
        1e-10*max(1, max(abs(c))) after the prescribed solve.
    """
    return np.zeros((2, 5, 4, 4))
```

### Step 4

04_propagate_kernel

Goal
----
Propagate the first two auxiliary kernels with classical RK4 and an elementwise rational third-kernel closure, retaining nonzero drift.

```python
def propagate_kernel(moments, closure, duration, steps):
    """Propagate two retained auxiliary kernels with nonzero drift.

    Parameters
    ----------
    moments : array_like, real, shape (13, 4, 4)
        Omega_0 through Omega_12. Use Omega_1 as the drift.
    closure : array_like, real, shape (2, 5, 4, 4)
        Numerator followed by denominator of K_3, ascending powers.
    duration : float
        Finite positive final time.
    steps : int
        Number of uniform classical RK4 intervals, from 1 to 4096.

    Returns
    -------
    ndarray, real, shape (steps + 1, 4, 4)
        K_1 at every grid point, including zero. Use the hierarchy
        and operator-defined initial values in step 2. Evaluate K_3
        at each RK4 stage; do not interpolate endpoint values.

    Raises
    ------
    ValueError
        If either array shape is incorrect, any entry is nonfinite,
        duration is nonfinite or nonpositive, steps is not an integer
        in [1, 4096], or any scalar closure denominator has magnitude
        at most 1e-10 at an RK4 stage time.
    """
    import numpy as np
    return np.zeros((steps + 1, 4, 4))
```

### Step 5

05_propagate_correlation

Goal
----
Integrate a matrix GQME with drift and a sampled causal memory kernel using nested trapezoidal quadrature and an implicit endpoint.

```python
def propagate_correlation(kernel, drift, dt):
    """Integrate the drift-plus-memory GQME on a uniform grid.

    Parameters
    ----------
    kernel : array_like, real, shape (N + 1, d, d)
        K_1 sampled at times n*dt; N >= 1 and d >= 1.
    drift : array_like, real, shape (d, d)
        Constant left-acting drift A. It need not commute with K_1.
    dt : float
        Finite positive time step.

    Returns
    -------
    ndarray, real, shape (N + 1, d, d)
        Correlation matrices from the two trapezoidal quadratures
        in the background, with C(0) = identity. Retain both the
        initial drift derivative and the implicit convolution endpoint.

    Raises
    ------
    ValueError
        If shapes are incorrect, any entry is nonfinite, dt is
        nonfinite or nonpositive, or the implicit coefficient matrix
        has 2-norm condition number greater than 1e12.
    """
    import numpy as np
    return np.zeros_like(kernel, dtype=float)
```

### Step 6

06_measure_population

Goal
----
Compute the population of the +1 eigenstate of axis·sigma from the final basis-correlation matrix and the initial Bloch vector.

```python
def measure_population(correlation, initial, axis):
    """Recover a qubit measurement probability.

    Parameters
    ----------
    correlation : array_like, real, shape (4, 4)
        Bath-weighted Heisenberg basis-correlation matrix in the
        ordered normalized basis (I, X, Y, Z)/sqrt(2).
    initial : array_like, real, shape (3,)
        Initial Bloch vector, with norm <= 1.
    axis : array_like, real, shape (3,)
        Unit Bloch vector defining the measured +1 eigenstate.

    Returns
    -------
    float
        Reconstructed population. Do not clip or renormalize it.

    Raises
    ------
    ValueError
        If shapes are incorrect, an entry is nonfinite, the initial
        norm exceeds 1+1e-12, or the axis norm differs from one
        by more than 1e-12.
    """
    return 0.0
```

### Step 7

07_solve

Goal
----
Compute the measured population by composing moments, third-kernel closure, kernel propagation and drift-aware correlation propagation. Compute the measured population by composing moments, third-kernel closure, kernel propagation and drift-aware correlation propagation.

```python
def solve(h, hb, beta, initial, axis, duration, steps):
    """Return the population from the complete nonzero-drift calculation.

    Parameters
    ----------
    h : array_like, complex, shape (2*b, 2*b)
        Finite Hermitian total Hamiltonian, system-first ordering.
    hb : array_like, complex, shape (b, b)
        Finite Hermitian bath Hamiltonian; b >= 1.
    beta : float
        Finite nonnegative inverse temperature.
    initial : array_like, real, shape (3,)
        Finite Bloch vector of norm at most 1 + 1e-12.
    axis : array_like, real, shape (3,)
        Finite measurement axis with norm within 1e-12 of one.
    duration : float
        Finite positive final time.
    steps : int
        Number of uniform intervals, from 1 to 4096.

    Returns
    -------
    float
        Population from moments through order 12, elementwise [4/4]
        closure, classical RK4 for two retained kernels and nested
        trapezoidal correlation propagation with drift Omega_1.
        Compose the preceding six functions; do not propagate H directly.

    Raises
    ------
    ValueError
        For invalid input shapes or values as specified above, Hermitian
        defects above 1e-12 entrywise, Padé matching residual above
        1e-10*max(1,max(abs(c))) with rcond=1e-12, a stage denominator
        magnitude at most 1e-10, or implicit condition number above 1e12.
    """
    return 0.0
```
