# Sixth-order response of an eliminated-root condition number

## Background

# Scientific background

Systems of polynomial equations in several unknowns arise throughout geometric estimation, where a handful of measurements determines a model up to a small finite set of candidates. A standard route to solving such a system is elimination: the unknowns other than one are removed, leaving a single univariate polynomial whose roots carry the remaining candidates.

Building that univariate polynomial is the delicate part. Forming it by symbolic expansion grows factorially with the size of the elimination matrix, so practical solvers evaluate the matrix at sample points and interpolate instead. Elimination matrices are also rarely used in their raw form: rows are scaled and recombined to improve numerical behaviour, and those manipulations can introduce roots that satisfy the determinant without satisfying the original equations. Separating the genuine solutions from the introduced ones therefore requires substituting each candidate back into the original polynomials and measuring how well they are satisfied, which makes back-substitution part of the numerical method rather than an optional check.

## Problem

Two polynomials in $y$ whose coefficients are polynomials in $x$ are to be solved simultaneously in double-precision arithmetic with zero-based indexing and no randomness; report the single perturbation-response scalar defined below. Write $f(x,y)=\sum_{m=0}^{4}\alpha_m(x)\,y^m$ and $g(x,y)=\sum_{m=0}^{3}\gamma_m(x)\,y^m$, where row $m$ of each array below lists the ascending $x$-coefficients of $\alpha_m$ and of $\gamma_m$:

$$
\alpha=\begin{bmatrix}1&-0.5&0.25\\-1.5&1&0.5\\2&0.5&-1\\-1&0.25&0.5\\1&0&0\end{bmatrix},\qquad
\gamma=\begin{bmatrix}-2&0.5\\1&-1.5\\0.5&0.75\\1&0.25\end{bmatrix},\qquad
L=\begin{bmatrix}1&0&0&0&0&0&-1\\0&1&2&0&0&0&0\\0&0&1&0&0&0&0\\-2&0&0&1&-2&0&2\\0&0&0&0&1&0&1\\0&0&0&0&0&1&0\\0&0&0&0&0&0&1\end{bmatrix}.
$$

Let $q(x)=(x-0.6)(x-2.5)$, and let $B(x)$ be the $7\times7$ array whose rows, applied to the monomial vector $(y^6,y^5,y^4,y^3,y^2,y,1)^T$, produce $q(x)\,y^2f$, $y\,f$, $f$, $y^3g$, $y^2g$, $y\,g$ and $g$ in that order; set $M(x)=L\,B(x)$. Zero-pad $M(x)=\sum_{\ell=0}^{12}A_\ell x^\ell$ as a degree-major tensor of shape $(13,7,7)$ and obtain the degree-twelve determinant polynomial $D(x)=\sum_{\ell=0}^{12}c_\ell x^\ell$. Restrict the hidden variable to the task-defined closed interval $0\le x\le3$: retain roots of $D$ whose imaginary magnitude is at most $10^{-8}$ and whose real part is in this interval, and at each retained root recover $y$ from the cofactor null vector of $M(x)$ taken along zero-based row $0$, normalising by the anchor entry at column $6$. Discard every recovered pair whose normalized residual $\widehat r=\max\bigl(|f(x,y)|,|g(x,y)|\bigr)/\max\bigl(1,\sqrt{x^2+y^2}\bigr)$ fails to fall below $10^{-3}$, evaluating $f$ and $g$ themselves rather than any row of $B$. For each surviving root compute $\kappa(x)=\lVert(1,x,\ldots,x^{12})\rVert_2/\lvert D'(x)\rvert$ and identify the unique root attaining the largest of those values at the unperturbed coefficients.


Use a confluent extension of the unit-circle determinant interpolation scheme: only four equally spaced Fourier nodes, together with the ordinary derivatives $D^{(r)}$ for $r=0,1,2,3$, are available for interpolation. The derivative data must be obtained analytically from the polynomial matrix, including the case of a singular sample matrix; the four value samples alone alias this degree-twelve polynomial. Recover all thirteen ascending coefficients from the value and derivative data, retaining the stated degree bound and validating the result at $x=0.37$; explain how the derivative information resolves the aliased degree classes. The use of confluent samples is a task-defined extension, not an assertion that the source uses derivative samples. Explain the source's Fourier sign and tensor-axis conventions, its offline test for choosing a recovery minor, and its instance-level numerical-failure criterion, distinguishing that criterion from the per-candidate residual gate defined above.

Now perturb only the constant coefficient of the quartic: $\alpha(t)=\alpha+tE_{00}$, where $E_{00}$ has shape $(5,3)$ with its only nonzero entry equal to one in row zero, column zero; keep $\gamma$, $q$, and $L$ fixed. Let $D(t,x)$ be the corresponding determinant and $x(t)$ the real simple-root branch continuing the maximizing admissible root at $t=0$; the admissibility decisions and root label are fixed at the base point for this local derivative. Define

$$
\ell(t)=\log\left(\frac{\sqrt{\sum_{n=0}^{12}x(t)^{2n}}}
{|\partial_xD(t,x(t))|}\right),\qquad H=\left.\frac{d^6\ell}{dt^6}\right|_{t=0},
$$

where $\log$ denotes the natural logarithm. Compute $H$ by continuing the implicit root and log condition through sixth order from the determinant's parameter Taylor coefficients; ordinary derivatives and Taylor coefficients must be distinguished. Recover those coefficients using eight parameter nodes $t_h=\exp(-2\pi i h/8)$ together with the four elimination-variable nodes and their derivative data specified above: an affine $7\times7$ matrix has determinant parameter degree at most seven, so eight parameter samples suffice. Validate the recovered bivariate determinant at $x=0.37$ for all parameter nodes. Both parameter continuation and confluent sampling are task-defined extensions of the source scheme. Report the final scalar within absolute tolerance $0.1$; this tolerance accommodates the amplification of double-precision interpolation errors in a sixth derivative.

Output Format Requirements:
Return $H$ as one finite decimal inside <final_answer>...</final_answer> and the scientific derivation inside <reasoning>...</reasoning>. Include the source conventions, interpolation and singular-matrix derivative justification, the base candidate and surviving sets, their determinant-derivative magnitudes and condition numbers, and the implicit root and log-condition continuation needed for the sixth derivative. Both tags are required. The final-answer tag must contain only $H$.

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

01_construct_resultant_tensor

Goal
----
*Build the degree-major coefficient tensor of a hidden-variable elimination matrix.*

```python
def construct_resultant_tensor(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
) -> np.ndarray:
    r"""Build the padded coefficient tensor of the elimination matrix.
    
    Parameters
    ----------
    alpha : np.ndarray
        Finite real coefficients of shape $(5,3)$. Entry $\alpha_{m\ell}$
        multiplies $x^\ell y^m$ in $f(x,y)$; both degree axes are ascending.
    gamma : np.ndarray
        Finite real coefficients of shape $(4,2)$. Entry $\gamma_{m\ell}$
        multiplies $x^\ell y^m$ in $g(x,y)$; both degree axes are ascending.
    u : float
        Finite first root of the row multiplier $q(x)=(x-u)(x-v)$.
    v : float
        Finite second root, separated from $u$ by more than $10^{-12}$.
    row_mixing : np.ndarray
        Finite real matrix $L$ of shape $(7,7)$ applied on the left.
        Require $\bigl||\det L|-1\bigr|\le10^{-9}$.
    determinant_degree : int, optional
        Degree bound $k\ge1$, default $12$. Allocates $k+1$ coefficient
        slices and must accommodate every row polynomial.
    
    Returns
    -------
    coefficient_tensor : np.ndarray
        Real array of shape $(k+1,7,7)$ such that
        $M(x)=\sum_{n=0}^{k}M_nx^n$. Slice $n$ is $M_n$ after row
        scaling and multiplication by $L$. Unused high-degree slices are zero.
    
    Raises
    ------
    ValueError
        If coefficient or row-mixing shapes are invalid, any data are nonfinite,
        $|u-v|\le10^{-12}$, $k$ is not a positive integer, the row factor
        fails its determinant-magnitude tolerance, or the allocated degree
        axis is too short for the row polynomials.
    
    Notes
    -----
    Construct the seven-row Sylvester stack for the quartic and cubic,
    scale its first row by $q(x)$, and apply $L$ on the left. The monomial
    order is $(y^6,y^5,y^4,y^3,y^2,y,1)$. Do not mutate input arrays.
    """
    return result
```

### Step 2

02_generate_unit_circle_samples

Goal
----
Generate the ordered roots-of-unity sample points.

```python
def generate_unit_circle_samples(determinant_degree: int) -> np.ndarray:
    r"""Return $x_j=\omega^{-j}$ for $j=0,\ldots,k$.
    
    Parameters
    ----------
    determinant_degree : int
        Integer $k\ge1$ specifying a grid of $S=k+1$ points.
    
    Returns
    -------
    sample_points : np.ndarray
        Complex vector of shape $(k+1,)$ in increasing sample index,
        with $x_j=\exp(-2\pi i j/(k+1))$ and $x_0=1$.
    
    Raises
    ------
    ValueError
        If `determinant_degree` is not an integer or is less than one.
    
    Notes
    -----
    The negative exponential sign pairs these samples with an inverse
    Fourier transform for recovering ascending polynomial coefficients.
    """
    return result
```

### Step 3

03_evaluate_resultant_fft

Goal
----
Evaluate a polynomial matrix and its ordinary derivatives on a Fourier grid.

```python
def evaluate_resultant_fft(
    coefficient_tensor: np.ndarray, sample_points: np.ndarray,
    derivative_order: int = 0,
) -> np.ndarray:
    r"""Evaluate $M$ and its ordinary derivatives with respect to $x$.
    
    Parameters
    ----------
    coefficient_tensor : np.ndarray
        Finite real or complex array of shape $(K,N,N)$ with $K\ge1$.
        Slice $n$ is the coefficient of $x^n$ in the square matrix $M(x)$.
    sample_points : np.ndarray
        Finite complex vector of shape $(S,)$ with $S\ge2$, ordered as
        $x_j=\exp(-2\pi i j/S)$ for $j=0,\ldots,S-1$.
        Each entry must match this grid within absolute tolerance $10^{-12}$.
    derivative_order : int, optional
        Highest ordinary derivative order $R$, default $0$, with $0\le R\le3$.
    
    Returns
    -------
    matrix_samples : np.ndarray
        Complex array of shape $(S,N,N)$ when $R=0$, or $(R+1,S,N,N)$
        when $R>0$. Channel $(r,j)$ is the ordinary derivative $M^{(r)}(x_j)$.
        Derivatives above the polynomial degree are zero. Every coefficient
        contributes even when $K>S$.
    
    Raises
    ------
    ValueError
        If the tensor is not rank three, its degree axis is empty, its
        matrices are nonsquare, points are not a vector of at least two
        entries, either input contains nonfinite values, $R$ is not an
        integer in $[0,3]$, or points fail the grid tolerance.
    
    Notes
    -----
    These channels contain ordinary derivatives, not Taylor coefficients.
    Use the polynomial-degree axis for evaluation, including aliased degrees
    when the coefficient axis is longer than the sampling grid.
    Do not mutate either input array.
    """
    return result
```

### Step 4

04_compute_determinant_samples

Goal
----
Evaluate determinant derivatives without assuming an invertible base matrix.

```python
def compute_determinant_samples(matrix_samples: np.ndarray) -> np.ndarray:
    r"""Return sampled determinants and optional ordinary derivative channels.
    
    Parameters
    ----------
    matrix_samples : np.ndarray
        Finite real or complex values of shape $(S,N,N)$, or ordinary
        matrix derivatives of shape $(J,S,N,N)$, with $S\ge1$ and
        $1\le J\le4$. In the latter form, entry $(r,j)$ is $M^{(r)}(x_j)$.
        The matrices must be square; singular matrices and $N=0$ are valid.
    
    Returns
    -------
    determinant_samples : np.ndarray
        Complex vector of shape $(S,)$ for value-only input, or array of
        shape $(J,S)$ with entry $(r,j)$ equal to $D^{(r)}(x_j)$ for
        derivative input. Here $D(x)=\det M(x)$. For $N=0$, the value
        channel is one and all higher derivative channels are zero.
    
    Raises
    ------
    ValueError
        If the input rank is neither three nor four, the sample axis is
        empty, matrices are nonsquare, the derivative-channel count is
        outside $[1,4]$, or any entry is nonfinite.
    
    Notes
    -----
    Determinant differentiation must include mixed row-derivative terms.
    The definition applies at singular matrices of any rank and requires no
    inverse of the base matrix. Unprovided higher matrix derivatives do not
    affect the requested orders. Do not mutate the input array.
    """
    return result
```

### Step 5

05_recover_determinant_coefficients

Goal
----
Recover a determinant polynomial from confluent Fourier data and certify it off grid.

```python
def recover_determinant_coefficients(
    determinant_samples: np.ndarray, coefficient_tensor: np.ndarray,
    validation_point: float = 0.37, validation_tolerance: float = 1e-9,
) -> np.ndarray:
    r"""Recover the $K$ ascending real coefficients of $D(x)=\det M(x)$.
    
    Parameters
    ----------
    determinant_samples : np.ndarray
        Finite real or complex data of shape $(S,)$ for values or $(J,S)$
        for ordinary derivatives $D^{(r)}(x_j)$, with $1\le J\le4$,
        $S\ge2$, $0\le r<J$, and $x_j=\exp(-2\pi i j/S)$.
        Value-only input means $J=1$. Require $JS\ge K$.
    coefficient_tensor : np.ndarray
        Finite real or complex array of shape $(K,N,N)$, with $K\ge1$,
        representing $M(x)$ in ascending degree order. The determinant
        $D(x)=\det M(x)$ is promised to have degree at most $K-1$ and
        numerically real coefficients.
    validation_point : float, optional
        Finite real off-grid point $x_*$, default $0.37$. Its distance
        from every sampling node must exceed $10^{-12}$.
    validation_tolerance : float, optional
        Finite positive tolerance $\tau$, default $10^{-9}$, for the
        coefficient-reality, reconstructed-sample, and off-grid checks.
    
    Returns
    -------
    coefficients : np.ndarray
        Real vector of shape $(K,)$ containing $c_0,\ldots,c_{K-1}$
        in ascending degree order. Retain trailing padding; the coefficients
        must pass all three consistency checks described below.
    
    Raises
    ------
    ValueError
        If input shapes, channel count, sample count, or tensor dimensions
        are invalid; $JS<K$; data or settings are nonfinite; $\tau\le0$;
        the validation point is within $10^{-12}$ of a grid node; or any
        coefficient-reality, reconstructed-sample, or off-grid check fails.
    
    Notes
    -----
    Within each Fourier residue class, separate aliased coefficients using all
    available derivative channels. Let $V$ be that class's falling-factorial
    moment matrix. Divide equation $r$ by
    $s_r=\max\{1,\max_n|V_{rn}|\}$. If there are more equations than unknowns,
    use the unique least-squares solution of the scaled equations.
    
    Let $\tau$ denote `validation_tolerance` and $x_*$ denote `validation_point`.
    Raise `ValueError` for invalid shapes, insufficient samples, nonfinite data
    or settings, $\tau\le0$, or $\min_j|x_*-x_j|\le10^{-12}$.
    The point $x_*$ is a finite real scalar. Reject recovered coefficients
    with $\max_n|\operatorname{Im}c_n|>\tau\max\{1,\max_n|c_n|\}$.
    Also reject inconsistent derivative data: if $Y$ denotes the supplied
    sample array and $\widetilde Y$ the samples reconstructed from the real
    coefficients, require
    $\max_{r,j}|\widetilde Y_{rj}-Y_{rj}|\le\tau\max\{1,\max_{r,j}|Y_{rj}|\}$.
    Finally, for $p(x)=\sum_{n=0}^{K-1}c_nx^n$, require
    $|p(x_*)-\det M(x_*)|\le\tau(1+|\det M(x_*)|)$; otherwise raise `ValueError`.
    Do not mutate inputs. Exact singular sample matrices are valid data.
    """
    return result
```

### Step 6

06_extract_hidden_candidates

Goal
----
Extract admissible real hidden-variable candidates.

```python
def extract_hidden_candidates(
    coefficients: np.ndarray, imaginary_tolerance: float = 1e-8,
    interval: tuple | None = None
) -> np.ndarray:
    r"""Solve an ascending-order polynomial and retain nearly real roots.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Finite one-dimensional real or complex coefficient array of length
        $K\ge2$, representing $D(x)=\sum_{n=0}^{K-1}c_nx^n$.
        The final entry is the leading coefficient and must be nonnegligible.
    imaginary_tolerance : float, optional
        Finite positive tolerance $\tau$, default $10^{-8}$. A root $z$
        passes the reality gate when $|\operatorname{Im}z|\le\tau$.
    interval : tuple or None, optional
        Default `None` applies no interval gate. Otherwise provide two
        finite real bounds $(a,b)$ with $a\le b$; retain roots satisfying
        $a\le\operatorname{Re}z\le b$, including the endpoints.
    
    Returns
    -------
    candidates : np.ndarray
        Nonempty real vector of retained real parts, sorted in ascending
        order. Preserve multiplicities; do not merge repeated roots.
    
    Raises
    ------
    ValueError
        If coefficients are not a finite vector of length at least two,
        $\tau$ is nonfinite or nonpositive, interval bounds are invalid,
        or no root survives both gates. Also raise when
        $|c_{K-1}|\le100\epsilon\max\{1,\max_n|c_n|\}$, where
        $\epsilon$ is double-precision machine epsilon.
    
    Notes
    -----
    Apply the reality test to each complex root before retaining its real
    part and applying the optional interval gate. No root polishing or
    multiplicity reduction is requested. Do not mutate the inputs.
    """
    return result
```

### Step 7

07_recover_and_filter_solutions

Goal
----
Recover the second variable at each candidate and discard spurious roots.

```python
def recover_and_filter_solutions(
    coefficient_tensor: np.ndarray,
    candidates: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    deletion_row: int = 0,
    anchor_column: int = 6,
    residual_threshold: float = 1e-3,
) -> np.ndarray:
    r"""Recover $y$ from a cofactor null vector and retain valid pairs.
    
    Parameters
    ----------
    coefficient_tensor : np.ndarray
        Finite coefficient tensor of shape $(K,7,7)$ representing $M(x)$
        in ascending degree order.
    candidates : np.ndarray
        Finite real vector of shape $(C,)$ containing hidden-variable
        candidates $x$ to test against the original polynomial equations.
    alpha : np.ndarray
        Real array of shape $(5,3)$; entry $\alpha_{m\ell}$ is the
        coefficient of $x^\ell y^m$ in the original quartic $f(x,y)$.
    gamma : np.ndarray
        Real array of shape $(4,2)$; entry $\gamma_{m\ell}$ is the
        coefficient of $x^\ell y^m$ in the original cubic $g(x,y)$.
    deletion_row : int, optional
        Row $d$ used for signed cofactors, default $0$, with $0\le d<7$.
    anchor_column : int, optional
        Denominator column $a$, default $6$, with $1\le a<7$.
        Recover $y=C_{d,a-1}/C_{d,a}$ in descending monomial order.
    residual_threshold : float, optional
        Finite positive threshold $\eta$, default $10^{-3}$.
        Retain a recovered pair only when its normalized residual is
        strictly less than $\eta$.
    
    Returns
    -------
    solutions : np.ndarray
        Nonempty real array of shape $(V,3)$, with each row
        $(x,y,\widehat r)$ containing a surviving pair and its normalized
        original-equation residual. Sort rows by increasing $x$, with
        $y$ and then residual used to break ties.
    
    Raises
    ------
    ValueError
        If tensor, candidate, or equation-coefficient shapes are invalid;
        indices are outside their stated ranges; the threshold is nonfinite
        or nonpositive; tensor or candidate entries are nonfinite; or
        no candidate yields an accepted recovered pair.
    
    Notes
    -----
    Use signed cofactors with monomial order $(y^6,y^5,y^4,y^3,y^2,y,1)$.
    Skip a candidate if its anchor magnitude is at most
    $100\epsilon\max\{1,\max_j|C_{d,j}|\}$, or if the recovered ratio
    has imaginary magnitude above $10^{-8}$. These individual rejections
    are not errors when other candidates survive.
    Evaluate the original equations using the real part of the accepted ratio.
    The normalized residual is
    $\widehat r=\max\{|f(x,y)|,|g(x,y)|\}/\max\{1,\sqrt{x^2+y^2}\}$.
    Do not mutate input arrays.
    """
    return result
```

### Step 8

08_compute_hidden_root_condition

Goal
----
Run the complete elimination-and-recovery pipeline and report one scalar.

```python
def compute_hidden_root_condition(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
    imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    r"""Return the largest condition number among the retained roots.
    
    Parameters
    ----------
    alpha : np.ndarray
        Finite real array of shape $(5,3)$ containing the coefficients
        of $f(x,y)$ in ascending $y$ degree and then ascending $x$ degree.
    gamma : np.ndarray
        Finite real array of shape $(4,2)$ containing the coefficients
        of $g(x,y)$ with the same degree conventions.
    u : float
        Finite first root of $q(x)=(x-u)(x-v)$.
    v : float
        Finite second root, separated from $u$ by more than $10^{-12}$.
    row_mixing : np.ndarray
        Finite real matrix $L$ of shape $(7,7)$ satisfying
        $\bigl||\det L|-1\bigr|\le10^{-9}$.
    determinant_degree : int, optional
        Positive degree bound $k$, default $12$, large enough for the
        row polynomials and the determinant. The coefficient vector has
        length $k+1$, including its padding.
    imaginary_tolerance : float, optional
        Finite positive root-reality tolerance, default $10^{-8}$.
    residual_threshold : float, optional
        Finite positive original-equation residual threshold, default $10^{-3}$.
        Residual acceptance uses a strict inequality.
    
    Returns
    -------
    condition_number : float
        Finite maximum of $\kappa(x)=\|(1,x,\ldots,x^k)\|_2/|D'(x)|$
        over roots in $[0,3]$ that survive reality and residual filtering.
        The polynomial $D$ includes the supplied row scaling and mixing.
    
    Raises
    ------
    ValueError
        If construction inputs or either tolerance violate the stated
        contracts, any intermediate stage rejects the data, no candidate
        survives filtering, a retained root is numerically multiple, or
        the resulting maximum is nonfinite. A retained root is numerically
        multiple when $|D'(x)|\le100\epsilon\max\{1,\max_n|c_n|\}$,
        with $\epsilon$ the double-precision machine epsilon.
    
    Notes
    -----
    Apply the earlier stages in order. Use ordinary derivative channels
    $r=0,1,2,3$ at $S=\max\{2,\lceil(k+1)/4\rceil\}$ Fourier points,
    then recover coefficients by confluent interpolation. Sampling matrices
    may be singular. Restrict candidates to the closed interval $[0,3]$
    before cofactor recovery and residual filtering. Do not mutate inputs.
    """
    return result
```

### Step 9

09_continue_condition_jets

Goal
----
Continue a simple determinant root and its logarithmic conditioning response.

```python
def continue_condition_jets(
    coefficient_jets: np.ndarray, anchor_root: float,
) -> np.ndarray:
    r"""Compute Taylor coefficients of a real simple root and its log condition.
    
    Parameters
    ----------
    coefficient_jets : np.ndarray
        Finite real array of shape $(P+1,K)$, with $0\le P\le6$ and $K\ge2$.
        Entry $(a,n)$ is the Taylor coefficient $c_{a n}$ in
        $D(t,x)=\sum_{a=0}^{P}\sum_{n=0}^{K-1}c_{a n}t^a x^n+O(t^{P+1})$.
        These are Taylor coefficients, not ordinary derivatives. High-degree
        zero padding is retained: the conditioning numerator uses all $K$ powers.
    anchor_root : float
        Finite real approximation $x_0$ to a simple root of $D(0,x)$.
        With $Q=\sum_n|c_{0n}|\max\{1,|x_0|\}^n$, require $Q>0$ and
        $|D(0,x_0)|\le10^{-8}Q$. The nearby simple root determines the branch.
    
    Returns
    -------
    response : np.ndarray
        Finite real array of shape $(P+1,2)$. Column zero contains the Taylor
        coefficients of the branch $x(t)$ satisfying $D(t,x(t))=0$.
        Column one contains the Taylor coefficients of
        $\ell(t)=\log\bigl(\sqrt{\sum_{n=0}^{K-1}x(t)^{2n}}/
        |\partial_xD(t,x(t))|\bigr)$, where $\log$ is natural logarithm.
        Row $a$ contains the ordinary derivative divided by $a!$.
    
    Raises
    ------
    ValueError
        If the array is complex, nonfinite, has invalid shape, or has zero base
        coefficients; the anchor is nonfinite or violates its residual bound;
        the nearby root is numerically multiple; or the response is nonfinite.
        A derivative magnitude no greater than
        $100\epsilon\sum_{n=1}^{K-1}n|c_{0n}|\max\{1,|x|\}^{n-1}$
        is numerically zero, with $\epsilon$ double-precision machine epsilon.
    
    Notes
    -----
    Refine the base approximation within its simple-root neighborhood before
    continuing the branch. Obtain the response from the supplied Taylor data,
    including the implicit motion of the root and the varying determinant scale.
    No residual or interval gate is applied in this step. Constant rescaling of
    all coefficient jets changes only the zeroth log-condition coefficient.
    Do not mutate inputs.
    """
    return result
```

### Step 10

10_compute_condition_response

Goal
----
Compute a high-order perturbation response of the largest retained condition number.

```python
def compute_condition_response(
    alpha: np.ndarray, gamma: np.ndarray, u: float, v: float,
    row_mixing: np.ndarray, alpha_direction: np.ndarray,
    gamma_direction: np.ndarray, response_order: int = 6,
    determinant_degree: int = 12, imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    r"""Return an ordinary derivative of the locally maximal log condition.
    
    Parameters
    ----------
    alpha : np.ndarray
        Finite real array of shape $(5,3)$ for the base quartic in $y$,
        with ascending $y$ degree followed by ascending $x$ degree.
    gamma : np.ndarray
        Finite real array of shape $(4,2)$ for the base cubic, with the
        same degree conventions.
    u : float
        Finite first root of the fixed row multiplier $q(x)=(x-u)(x-v)$.
    v : float
        Finite second root, with $|u-v|>10^{-12}$.
    row_mixing : np.ndarray
        Finite real matrix $L$ of shape $(7,7)$ with
        $\bigl||\det L|-1\bigr|\le10^{-9}$, fixed under perturbation.
    alpha_direction : np.ndarray
        Finite real array of shape $(5,3)$ defining
        $\alpha(t)=\alpha+t\,\alpha_{\mathrm{direction}}$.
    gamma_direction : np.ndarray
        Finite real array of shape $(4,2)$ defining
        $\gamma(t)=\gamma+t\,\gamma_{\mathrm{direction}}$.
    response_order : int, optional
        Ordinary derivative order $P$, default $6$, with $0\le P\le6$.
    determinant_degree : int, optional
        Degree bound $k$, default $12$, with $K=k+1$. It must hold for
        the determinant for all sufficiently small $t$, including its
        parameter coefficient polynomials, and accommodate the matrix rows.
    imaginary_tolerance : float, optional
        Finite positive base-root reality tolerance, default $10^{-8}$.
    residual_threshold : float, optional
        Finite positive base-pair residual threshold, default $10^{-3}$.
    
    Returns
    -------
    response : float
        The finite ordinary derivative $\ell^{(P)}(0)$, where $\ell(t)$
        is the log condition continued from the unique base root attaining
        the largest admissible condition number. Order zero returns its
        natural logarithm. Base candidates lie in $[0,3]$ and use the
        preceding cofactor recovery and original-equation residual gates.
    
    Raises
    ------
    ValueError
        If construction data, directions, order, or tolerances are invalid;
        coefficient recovery fails the realness or consistency checks below;
        no base root survives; a retained base root is numerically multiple;
        the largest and second-largest base condition numbers differ by at
        most $10^{-8}\max\{1,\kappa_{\max}\}$; or the response is nonfinite.
        Errors from the preceding stages propagate.
    
    Notes
    -----
    Compose the earlier stages, including the base maximum and the continuation
    step. The admissible set, anchor choices, and maximizing root label are
    fixed at $t=0$ for the local derivative; do not differentiate comparisons
    or independently reselect roots at perturbed parameters.
    
    Use eight parameter nodes $t_h=\exp(-2\pi i h/8)$ and
    $S=\max\{2,\lceil K/4\rceil\}$ elimination-variable nodes
    $x_j=\exp(-2\pi i j/S)$. The affine $7\times7$ matrix has determinant
    parameter degree at most seven. Obtain its elimination-variable derivatives
    of orders zero through three analytically at each parameter node, including
    singular matrices. Recover all eight parameter coefficient polynomials,
    then retain Taylor orders zero through $P$ for continuation. Every such
    polynomial has at most $K$ coefficients; preserve padding.
    
    Separate elimination-degree aliases with the same row-scaled
    falling-factorial moment systems as in the earlier confluent recovery step.
    Take an inverse transform in the parameter index to obtain Taylor
    coefficients; this transform does not give ordinary parameter derivatives.
    The recovered real bivariate coefficients must reproduce all sampled
    elimination derivatives within maximum absolute error
    $10^{-7}\max\{1,\max|Y|\}$, where $Y$ is the supplied sample array.
    Their imaginary parts must be at most $10^{-7}$ times the larger of one
    and the maximum coefficient magnitude. Also require the reconstructed
    values at $x=0.37$ and all eight parameter nodes to agree with direct
    matrix determinants within $10^{-7}(1+\max|D(t_h,0.37)|)$.
    Compute the high-order response from these polynomial coefficients and
    implicit continuation. Do not mutate inputs.
    """
    return result
```
