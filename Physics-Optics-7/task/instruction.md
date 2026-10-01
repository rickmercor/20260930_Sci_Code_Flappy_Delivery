# Physics-Optics-7

## Background

All coordinates, phases and frequencies are dimensionless. The reduced phase is held fixed by setting the plasma column density separately for each monochromatic trial. The arrival-time function is $T(\mathbf x,\mathbf y)=\tfrac12|\mathbf x-\mathbf y|^2+\varphi(\mathbf x)$ and the normalized field is the Abel limit
$$\Psi(\omega,\mathbf y)=\lim_{\eta\downarrow0}\frac{\omega}{2\pi i}\int_{\mathbb R^2}\exp\{i\omega T(\mathbf x,\mathbf y)-\eta|\mathbf x|^2\}\,d^2\mathbf x.$$
It equals one for an empty screen. Let $\mathbf y_c=\mathbf x_c+\nabla\varphi(\mathbf x_c)$, and let $(\mathbf e,\mathbf f)$ be an orthonormal eigenframe of the arrival-time Hessian at the fold, with $\mathbf e$ in its null direction, $D_{\mathbf e}^3\varphi(\mathbf x_c)>0$, $\mathbf f=(-e_2,e_1)$, and transverse eigenvalue $\lambda>0$. Set $c=(2/D_{\mathbf e}^3\varphi(\mathbf x_c))^{1/3}$ and $\mathbf y(\omega,\zeta)=\mathbf y_c-\zeta\mathbf e/(c\omega^{2/3})$.

The uniform contribution is defined by the local Taylor expansion of this arrival time at $\mathbf x_c$ in that eigenframe, with the fold coordinate of order $\omega^{-1/3}$ and the transverse coordinate of order $\omega^{-1/2}$, followed by termwise canonical oscillatory integration. Its relative order is measured after extracting its leading $\omega^{1/6}$ factor; truncation through $m$ retains all terms through $\omega^{-m/3}$. The regular contribution is the stationary-phase expansion at the unique minimum image $\mathbf x_r$ satisfying $\mathbf x_r-\mathbf y+\nabla\varphi(\mathbf x_r)=0$ and $\alpha\exp(-\mathbf x_r^TA\mathbf x_r/2)<1/\lambda_{\max}(A)$. Its phase and the fold phase use the same $T$ and the positive-frequency outgoing convention; the fold contribution represents the coalescing image pair. The method and its arbitrary-order multidimensional extension are the scientific object of this constructed benchmark; the grid, error functional and truncation selection are task-specific. Use unrounded intermediate values and enough precision to make the complex reference field accurate to $10^{-8}$ in absolute value. NumPy, SciPy and the Python standard library are available. The six reported scalars fit the short reasoning contract.

## Problem

A two-dimensional plasma lens has reduced phase $\varphi(\mathbf x)=\alpha\exp(-\mathbf x^TA\mathbf x/2)$, with $A=\begin{pmatrix}1&0.18\\0.18&0.58\end{pmatrix}$ and a prescribed fold point $\mathbf x_c=(0.55,0.28)^T$; choose the smallest positive $\alpha$ that makes this a fold with a positive transverse Hessian eigenvalue. Four independently tuned monochromatic trials keep this reduced lens fixed at dimensionless frequencies $\omega=(40,45,50,80)$, with incident field amplitude $1.7$ and detector power gain $2.3$. At each frequency evaluate the four source positions with fold unfoldings $\zeta=(-1.2,-0.4,0.3,1.1)$ and corresponding weights $(1,2,2,1)$, using the conventions below. Let $\Psi_m$ combine the derivative-based two-dimensional uniform fold expansion through relative order $\omega^{-m/3}$ with the nondegenerate minimum-image expansion through $\omega^{-2}$, for every $m\in\{0,\ldots,6\}$, and let $\Psi$ be the full diffraction field obtained from the convergent Gaussian-screen expansion. Define $E_m(\omega)^2=\sum_j w_j|\Psi_m(\omega,\zeta_j)-\Psi(\omega,\zeta_j)|^2/\sum_j w_j|\Psi(\omega,\zeta_j)|^2$ and $R_m=\max_\omega E_m(\omega)$. Return $100R_{m_*}$, where $m_*$ minimizes $R_m$ and a tie selects the smaller order. In the short reasoning give the five-component certificate $(m_*,\omega_*,100R_0,100R_6,100(R_{m_{(2)}}-R_{m_*}))$, where $\omega_*$ is the first listed frequency attaining the selected worst error and $m_{(2)}$ is the runner-up under the same ordering, and justify the mixed-coordinate grading, the first nonzero fold correction, the coherent matching, the reference calculation, and the cancellation of the common incident amplitude and detector gain. Report errors and the gap to four significant figures with absolute tolerance $0.005$ percentage points; the order and frequency are exact.

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

fold_frame

Goal
----
Recover the oriented Gaussian fold and its cubic scale.

```python
def fold_frame(precision: "np.ndarray", point: "np.ndarray") -> "np.ndarray":
    """Recover the oriented Gaussian fold and its cubic scale.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A.
    point : ndarray, shape (2,), float
        Prescribed simple fold location x_c. The smallest eigenvalue of
        (A x_c)(A x_c)^T-A is strictly negative and simple, its chosen
        strength gives a positive transverse eigenvalue, and the cubic is nonzero.
    Returns
    -------
    ndarray, shape (9,), float
        [alpha, y_c[0], y_c[1], V[0,0], V[0,1], V[1,0], V[1,1], lambda, c].
        V=[e,f] with D_e^3 phi>0 and f=(-e[1],e[0]); c is positive.
        All quantities are dimensionless.
    """
    return result
```

### Step 2

gaussian_jet

Goal
----
Compute the mixed Taylor coefficients of the Gaussian screen in an orthonormal frame.

```python
def gaussian_jet(precision: "np.ndarray", alpha: float, point: "np.ndarray", basis: "np.ndarray", degree: int) -> "np.ndarray":
    """Compute the mixed Taylor coefficients of the Gaussian screen in an orthonormal frame.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A.
    alpha : float
        Real reduced screen strength, including zero for the empty-screen limit.
    point : ndarray, shape (2,), float
        Expansion location.
    basis : ndarray, shape (2,2), float
        Orthonormal matrix whose columns define local coordinates (u,v).
    degree : int
        Total Taylor degree, 0 through 12.
    Returns
    -------
    ndarray, shape (degree+1,degree+1), float
        Entry [a,b] is the coefficient of u^a v^b in
        alpha*exp(-(point+basis@(u,v))^T A (point+basis@(u,v))/2).
        Entries with a+b>degree are zero. Coefficients include factorial division.
        All entries are dimensionless.
    """
    return result
```

### Step 3

fold_powers

Goal
----
Construct the graded perturbation polynomials for the full two-dimensional fold.

```python
def fold_powers(jet: "np.ndarray", scale: float, transverse: float, order: int) -> "np.ndarray":
    """Construct the graded perturbation polynomials for the full two-dimensional fold.

    jet : ndarray, shape (D+1,D+1), float
        Phase Taylor coefficients J[a,b] including factorial division, from a
        simple fold. The cubic canonical coefficient satisfies J[3,0]*scale^3=1/3.
        D>=floor(order/2)+3; entries outside total degree D are zero.
    scale : float
        Positive null-coordinate cubic scale c.
    transverse : float
        Nonzero transverse Hessian eigenvalue lambda.
    order : int
        Highest epsilon power K, 0 through 12.
    Returns
    -------
    ndarray, shape (order+1,2*order+1,order+1), complex
        [k,a,b] is the coefficient of epsilon^k t^a s^b in the exponential
        perturbation multiplying the canonical fold/transverse integrand.
        Unused coefficients are zero. All entries are dimensionless.
    """
    return result
```

### Step 4

fold_contractions

Goal
----
Evaluate the Airy and transverse Gaussian contractions of each graded polynomial.

```python
def fold_contractions(powers: "np.ndarray", unfolding: float, transverse: float) -> "np.ndarray":
    """Evaluate the Airy and transverse Gaussian contractions of each graded polynomial.

    powers : ndarray, shape (K+1,2*K+1,K+1), complex
        Graded perturbation coefficient tensor P[k,a,b], 0<=K<=12.
    unfolding : float
        Real normalized Airy argument zeta.
    transverse : float
        Nonzero transverse Hessian eigenvalue lambda.
    Returns
    -------
    ndarray, shape (K+1,), complex
        The coefficient u[k] after normalized oscillatory integration of P[k].
        The normalization is by 2*pi times sqrt(2*pi)*exp(i*pi*sign(lambda)/4).
        Thus the constant polynomial contracts to Ai(zeta). All entries are
        dimensionless and ordered by increasing epsilon power.
    """
    return result
```

### Step 5

saddle_series

Goal
----
Evaluate the nondegenerate two-dimensional refractive correction series.

```python
def saddle_series(jet: "np.ndarray", eigenvalues: "np.ndarray", order: int) -> "np.ndarray":
    """Evaluate the nondegenerate two-dimensional refractive correction series.

    jet : ndarray, shape (D+1,D+1), float
        Phase Taylor coefficients in the nondegenerate Hessian eigenframe,
        including factorial division. D>=2*order+2. Constant, linear and quadratic
        entries can be present; the eigenvalues separately describe total T.
    eigenvalues : ndarray, shape (2,), float
        Nonzero real eigenvalues of the full arrival-time Hessian, in jet order.
    order : int
        Highest inverse-frequency order M, 0 through 3.
    Returns
    -------
    ndarray, shape (order+1,), complex
        Coefficients t[m] of sum_m t[m]/omega^m after removing the full leading
        saddle prefactor, with t[0]=1. All entries are dimensionless.
    """
    return result
```

### Step 6

gaussian_diffraction

Goal
----
Evaluate the full Gaussian-screen diffraction amplitude with controlled cancellation.

```python
def gaussian_diffraction(precision: "np.ndarray", alpha: float, position: "np.ndarray", frequency: float) -> "np.ndarray":
    """Evaluate the full Gaussian-screen diffraction amplitude with controlled cancellation.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A.
    alpha : float
        Nonnegative reduced Gaussian strength, 0 through 3.
    position : ndarray, shape (2,), float
        Source position y.
    frequency : float
        Positive dimensionless frequency, 0.1 through 180.
    Returns
    -------
    ndarray, shape (2,), float
        [Re(Psi), Im(Psi)] for the Abel-regulated full two-dimensional lens field.
        Both components have required absolute accuracy 1e-8. All entries are
        dimensionless; Psi=1 for alpha=0.
    """
    return result
```

### Step 7

coherent_hybrid

Goal
----
Match the uniform fold pair to the regular image at each retained order.

```python
def coherent_hybrid(fold_coefficients: "np.ndarray", regular_coefficients: "np.ndarray", fold_delay: float, regular_delay: float, scale: float, transverse: float, regular_eigenvalues: "np.ndarray", frequency: float) -> "np.ndarray":
    """Match the uniform fold pair to the regular image at each retained order.

    fold_coefficients : ndarray, shape (2*M+1,), complex
        Normalized coefficients u[k] from fold_contractions, 0<=M<=6.
    regular_coefficients : ndarray, shape (L+1,), complex
        Normalized nondegenerate coefficients t[m], 0<=L<=3.
    fold_delay : float
        T(x_c,y), using the actual source position y.
    regular_delay : float
        T(x_r,y), in the same phase convention.
    scale : float
        Positive cubic scale c of the fold.
    transverse : float
        Nonzero transverse eigenvalue lambda at the fold.
    regular_eigenvalues : ndarray, shape (2,), float
        Nonzero Hessian eigenvalues of the regular image.
    frequency : float
        Positive dimensionless frequency.
    Returns
    -------
    ndarray, shape (M+1,), complex
        Entry m is the coherent total field from the fold truncated through
        relative omega^(-m/3) and the supplied regular series. All entries are
        dimensionless; leading fold and Morse phases follow the outgoing Abel
        convention in the problem.
    """
    return result
```

### Step 8

error_profile

Goal
----
Measure the weighted complex-field error of every truncation in each trial.

```python
def error_profile(approximations: "np.ndarray", references: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Measure the weighted complex-field error of every truncation in each trial.

    approximations : ndarray, shape (F,Z,M+1), complex
        Approximate fields indexed by frequency, unfolding, retained order.
    references : ndarray, shape (F,Z), complex
        Full diffraction fields on the same grid.
    weights : ndarray, shape (Z,), float
        Nonnegative unfolding weights, with positive sum. Every reference
        frequency row has strictly positive weighted squared norm.
    Returns
    -------
    ndarray, shape (F,M+1), float
        Entry [i,m] is the square root of weighted squared complex field error
        divided by weighted squared reference norm for frequency i and order m.
        Values are dimensionless fractions, before multiplication by 100.
    """
    return result
```

### Step 9

plasma_fold_benchmark

Goal
----
Compose every preceding public function to select the fold truncation and its error certificate.

```python
def plasma_fold_benchmark(precision: "np.ndarray", point: "np.ndarray", frequencies: "np.ndarray", unfoldings: "np.ndarray", weights: "np.ndarray", max_order: int = 6) -> "np.ndarray":
    """Compose every preceding public function to select the fold truncation and its error certificate.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A, as in fold_frame.
    point : ndarray, shape (2,), float
        Simple fold point x_c. The constructed alpha lies in (0,3].
    frequencies : ndarray, shape (F,), float
        Distinct trial frequencies in [10,180], kept in supplied order.
    unfoldings : ndarray, shape (Z,), float
        Normalized source offsets zeta. All specified source positions admit
        the unique regular minimum image on the stated q branch.
    weights : ndarray, shape (Z,), float
        Nonnegative weights with positive sum and positive reference norms.
    max_order : int, default 6
        Largest retained fold correction index, 1 through 6.
    Returns
    -------
    ndarray, shape (6,), float
        [best error percent, selected integer order, worst listed frequency,
        leading-order worst error percent, max_order worst error percent,
        runner-up minus best error in percentage points]. Minimize worst error
        across frequencies; ties select the smaller order, and ties in the
        maximizing frequency select its first supplied position. All trial
        quantities are dimensionless. The first entry is the requested scalar.
        The public implementation composes and uses every preceding public API.
    """
    return result
```
