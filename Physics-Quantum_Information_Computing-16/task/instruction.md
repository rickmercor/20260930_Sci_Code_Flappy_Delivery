# Physics-Quantum_Information_Computing-16

## Problem

Consider a translationally invariant frustration-free quantum spin chain with periodic $N$-site Hamiltonian
 
$$H_N=\sum_{i=1}^N h_i,$$
 
where each $h_i\succeq 0$ is a translated copy of the same local interaction $h$. The spectral gap is the smallest non-zero eigenvalue of $H_N$. If the condition $H_N^2-\delta H_N\succeq 0$ holds, then the spectral gap of $H_N$ is at least $\delta$ (remember that $M\succeq 0$ means $M$ is positive semidefinite). Directly testing this condition becomes exponentially costly with system size, and infeasible in the thermodynamic limit $N\to\infty$. Recent approaches have developed an efficient method for certifying a non-zero spectral gap for frustration-free, translationally invariant quantum spin chains in the thermodynamic limit: rather than testing the positivity condition on the full many-body Hilbert space, they introduce a hierarchy of locally translationally invariant (LTI) gap semidefinite programs (SDPs). At level $n$, the operator $H^2$ is replaced by an $n$-local truncation $[H^2]_n$, and the method optimizes over $n$-site positive semidefinite generating local terms whose translated copies sum to $[H^2]_n-\delta H$. This turns the search for a thermodynamic gap certificate into a finite semidefinite program whose optimum is a valid lower bound on the gap for every $N\ge 2n$.
 
The certified bound is the optimal value of an SDP whose data depend smoothly on the Hamiltonian; wherever that optimal value is differentiable, its parameter derivatives measure how robust a gap certificate is. The paper discusses such sensitivities through the dual LTI marginal (a relation of Hellmann-Feynman type).
 
Consider the deformed $\mathbb Z_3$ Potts-clock model with deformation parameters $(r,s)$. You have to compute the derivative with respect to $r$, at fixed $s$, of the level-3 certifiable LTI lower bound $\delta_h(r,s)$ for the original interaction $h$, evaluated at $r=0.347$, $s=0.783$. Follow these instructions:
 
- Use the paper's deformed clock coefficients with $\omega=e^{2\pi i/3}$, $\bar\omega=\omega^*$, $\sigma=\operatorname{diag}(1,\omega,\omega^2)$ and $\tau|j\rangle=|j+1\bmod 3\rangle$:
 
\[
\begin{aligned}
f&=-\frac{2}{9}\left[
2\left(rs+\omega\frac{r}{s^2}+\bar\omega\frac{s}{r^2}\right)
-\left(\frac1{rs}+\bar\omega\frac{r^2}{s}+\omega\frac{s^2}{r}\right)\right],\\
g_1&=-\frac{2}{9}\left[\omega\left(\frac{r^2}{s}+\frac{s}{r^2}\right)
+\bar\omega\left(\frac{s^2}{r}+\frac{r}{s^2}\right)+rs+\frac1{rs}\right],\\
g_2&=\frac19\left[3+\frac1{rs}+\frac{s^2}{r}+\frac{r^2}{s}-2\left(rs+\frac{s}{r^2}+\frac{r}{s^2}\right)\right].
\end{aligned}
\]
 
- Use the following explicit two-site representation of the interaction in Eq. (39) of the paper (this removes any ambiguity from the typesetting):
\[
A=-\left[\sigma\otimes\sigma^\dagger+\frac f2(\tau\otimes I_3+I_3\otimes\tau)+g_1\,\tau\otimes\tau+g_2\,\tau\otimes\tau^\dagger\right],
\qquad
h=A+A^\dagger+\epsilon I_9,
\]
where $\epsilon=-\lambda_{\min}(A+A^\dagger)$. Classify eigenvalues with magnitude at most $10^{-10}$ as zero.
 
- Write $H=\sum_i h_i$ with $h_i$ acting on sites $i,i+1$. Let $[H^2]_n$ be the sum of those products $h_ih_j$ whose joint support fits inside $n$ consecutive sites. The level-$n$ certifiable LTI SDP is
\[
\delta_h=\max_{\delta\in\mathbb R,\;Y=Y^\dagger}\delta\quad\text{subject to}\quad
g_n(\delta)+I_3\otimes P_{n-1}^{\perp}YP_{n-1}^{\perp}-P_{n-1}^{\perp}YP_{n-1}^{\perp}\otimes I_3\succeq0,
\]
where $g_n(\delta)$ is an $n$-site operator whose translates sum to $[H^2]_n-\delta H$, $Y$ is a Hermitian operator on $n-1$ sites, and $P_{n-1}^{\perp}$ is the projector onto the eigenvectors of the $(n-1)$-site open chain $\sum_{i=1}^{n-2}h_i$ with eigenvalues greater than $10^{-10}$. Instantiate at $n=3$ and take
\[
g_3(\delta)=h_1^{2}+\{h_1,h_2\}-\delta\,h_1,
\qquad h_1=h\otimes I_3,\qquad h_2=I_3\otimes h .
\]
An alternative generator $g_3'(\delta)$ is admissible only if it differs from this one by
\[
g_3'(\delta)-g_3(\delta)=I_3\otimes P_2^{\perp}KP_2^{\perp}-P_2^{\perp}KP_2^{\perp}\otimes I_3
\]
for some Hermitian $K$ on two sites; generators outside this family are not permitted, since a translate-summing addition can have zero expectation and still act non-trivially on a ground state, rendering the program infeasible for every $\delta$. State which products $h_ih_j$ survive in $[H^2]_3$ and why the others may be dropped, and what $P_2^{\perp}$ is.
 
- The requested quantity is
\[
\left.\frac{\partial\delta_h(r,s)}{\partial r}\right|_{r=0.347,\;s=0.783},
\]
where $\delta_h(r,s)$ is the optimal value of the level-3 certifiable SDP defined above, with every ingredient of the SDP (the coefficients, $h$, its kernel, and $P_2^{\perp}$) recomputed at each value of $r$. Any method that evaluates this derivative is acceptable, but the result must be accurate to $10^{-4}$; solve each SDP to a tolerance of at most $10^{-10}$, and provide an independent numerical check supporting the local consistency of the reported derivative at the target point. In your answer report the coefficients $f$, $g_1$, $g_2$, the shift $\epsilon$, the certified bound $\delta_h(0.347,0.783)$, the minimum eigenvalue of the certifying operator $q_3$ at your optimum on the orthogonal complement of the three-site open-chain ground space together with what that value does and does not certify, how you handled the fact that $q_3$ annihilates that ground space, the method used for the derivative together with its consistency check, a discussion of whether the Hellmann-Feynman gradient relation for the LTI bound applies unchanged to the certifiable bound, and the value of the derivative. Return the derivative as a single plain decimal number rounded to four decimals, with no units, words, or LaTeX around it, even if you are unsure of its accuracy. Note that the optimum of the certifiable SDP, and hence the derivative, is unchanged under the admissible modifications defined above, so the requested value is unique.

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

01_deformation_coefficients.py

Goal
----
Step 1: Evaluate the deformed Z3 clock coefficients f, g1, g2.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
def deformation_coefficients(r: float, s: float) -> tuple[complex, complex, float]:
    '''Return the deformation coefficients (f, g1, g2).
 
    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters.
 
    Returns
    -------
    result : tuple[complex, complex, float]
        (f, g1, g2) with f, g1 complex and g2 a native Python float.
 
    Raises
    ------
    ValueError
        If r or s is non-finite or <= 0.
    '''
    return result  # placeholder
```

### Step 2

02_build_local_interaction.py

Goal
----
Step 2: Construct the PSD two-qutrit interaction h and its shift epsilon.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================
def build_local_interaction(
    r: float,
    s: float,
) -> tuple[np.ndarray, float]:
    """Construct the 9x9 Hermitian PSD local interaction h and its shift epsilon.

    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters.

    Returns
    -------
    result : tuple[np.ndarray, float]
        (h, epsilon): h complex Hermitian of shape (9, 9),
        epsilon a native float.

    Raises
    ------
    ValueError
        If r or s is non-finite or <= 0.
    """
    return result
```

### Step 3

03_excited_subspace_projector.py

Goal
----
Step 3: Projector onto the positive-energy subspace and kernel dimension.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================
def excited_subspace_projector(H: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    '''Return the projector onto eigenvectors of H with eigenvalue > tol, and the kernel dimension.

    Parameters
    ----------
    H : np.ndarray
        Hermitian positive-semidefinite square matrix.
    tol : float
        Positive eigenvalue threshold.

    Returns
    -------
    result : tuple[np.ndarray, int]
        (P_perp, dim_ker): P_perp an orthogonal projector with the shape of H,
        dim_ker the number of eigenvalues <= tol as a native int.

    Raises
    ------
    ValueError
        If H is not a nonempty square Hermitian matrix, tol is not finite and > 0,
        or H has an eigenvalue below -tol.
    '''
    return result
```

### Step 4

04_level3_generator.py

Goal
----
Step 4: Assemble the level-3 certifiable LTI generating operator q3.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def level3_generator(
    x: np.ndarray,
    p2_perp: np.ndarray,
    delta: float,
    Y: np.ndarray,
) -> np.ndarray:
    """Construct the 27x27 level-3 certifiable LTI generator q3(delta, Y).

    Parameters
    ----------
    x : np.ndarray
        Hermitian local interaction of shape (9, 9).
    p2_perp : np.ndarray
        Hermitian orthogonal projector of shape (9, 9).
    delta : float
        Finite candidate gap bound.
    Y : np.ndarray
        Hermitian matrix of shape (9, 9).

    Returns
    -------
    q3 : np.ndarray
        Hermitian array of shape (27, 27).

    Raises
    ------
    ValueError
        If x, p2_perp, or Y is not Hermitian of shape (9, 9),
        if p2_perp is not idempotent, or if delta is not finite.
    """
    return q3
```

### Step 5

05_solve_certifiable_lti_bound.py

Goal
----
Step 5: Solve the level-3 certifiable LTI SDP for one local interaction.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
import numpy as np

def solve_certifiable_lti_bound(
    x: "np.ndarray",
    p2_perp: "np.ndarray",
    solver_tol: float = 1e-11,
) -> float:
    """Solve the level-3 certifiable LTI SDP and return its optimal value.

    Parameters
    ----------
    x : np.ndarray
        Hermitian positive-semidefinite local interaction of shape (9,9).
    p2_perp : np.ndarray
        Hermitian orthogonal projector of shape (9,9).
    solver_tol : float
        Positive finite barrier tolerance (the optimum is accurate to about this value).

    Returns
    -------
    delta : float
        Level-3 certifiable LTI lower bound as a native Python float.

    Raises
    ------
    ValueError
        If x is not Hermitian PSD of shape (9,9), p2_perp is not a Hermitian projector
        of shape (9,9), solver_tol is not finite and > 0, the barrier method cannot find a
        strictly feasible start, or the returned solution fails the numerical feasibility check.
    """
    return delta
```

### Step 6

06_certified_bound_at_parameters.py

Goal
----
Step 6: The certified level-3 bound delta_h(r,s) as a function of the parameters.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
def certified_bound_at_parameters(r: float, s: float, tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    '''Return delta_h(r,s), the level-3 certifiable LTI bound for the original interaction.

    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters.
    tol : float
        Positive eigenvalue threshold for kernels and projectors.
    solver_tol : float
        Positive finite SDP solver tolerance.

    Returns
    -------
    delta_h : float
        Certified bound as a native Python float.

    Raises
    ------
    ValueError
        If r, s, tol or solver_tol is non-finite or <= 0, or a downstream step raises.
    '''
    return delta_h  # placeholder
```

### Step 7

07_central_difference.py

Goal
----
Step 7: Symmetric finite-difference derivative of a scalar function.

```python
# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

import numpy as np
from collections.abc import Callable

def central_difference(
    fn: Callable[[float], float],
    r0: float,
    step: float,
) -> float:
    """Return the symmetric finite-difference derivative of fn at r0.

    Parameters
    ----------
    fn : callable
        Scalar function of one float argument.
    r0 : float
        Finite evaluation point.
    step : float
        Finite positive step; r0 - step must be > 0.

    Returns
    -------
    deriv : float
        (fn(r0 + step) - fn(r0 - step)) / (2 * step) as a native Python float.

    Raises
    ------
    ValueError
        If fn is not callable, r0 or step is non-finite, step <= 0,
        r0 - step <= 0, or fn returns a non-finite value.
    """
    return deriv
```

### Step 8

08_gap_bound_sensitivity.py

Goal
----
Step 8 (ORCHESTRATOR): d delta_h / dr of the level-3 certifiable bound at fixed s.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
def gap_bound_sensitivity(r: float = 0.347, s: float = 0.783, step: float = 1e-4,
                          tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    '''Return d delta_h(r,s) / dr at fixed s by a symmetric finite difference.

    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters at which the derivative is taken.
    step : float
        Finite positive finite-difference step, smaller than r.
    tol : float
        Positive eigenvalue threshold.
    solver_tol : float
        Positive finite SDP solver tolerance.

    Returns
    -------
    deriv : float
        The derivative as a native Python float.

    Raises
    ------
    ValueError
        If r or s is non-finite or <= 0, step is non-finite, <= 0 or >= r,
        or a downstream step raises.
    '''
    return deriv
```
