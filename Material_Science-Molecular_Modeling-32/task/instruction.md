# Curvature of phase-averaged band stretching on a driven molecular orbit

## Background

# Scientific background

Molecular dynamics models atomic motion through a potential-energy landscape. Near a stable equilibrium, collective displacements can often be described as coupled harmonic oscillators. If $H$ is the local Hessian and $M$ the positive diagonal mass matrix, vibrational modes satisfy
$$H u=\omega^2M u.$$
Positive squared frequencies describe locally restoring motion. The mass-weighted Hessian is symmetric, which connects vibrational analysis to orthogonal spectral subspaces.

Away from equilibrium, nonlinear forces couple these modes. Anharmonicity can shift frequencies, transfer energy between vibrational bands, and produce responses that depend on driving conditions. Molecular parameters also influence equilibrium geometries and mode shapes, making reduced vibrational descriptions sensitive to the physical model used to define them.

Numerical propagation introduces a second source of structure: the finite timestep. Symplectic methods preserve the canonical geometry of Hamiltonian dynamics, while time-periodic forcing leads naturally to maps over a forcing period. Linear perturbations of these maps distinguish asymptotic orbital stability from finite-time amplification. Quantitative response analysis therefore draws on molecular mechanics, nonlinear dynamics and matrix perturbation theory.

## Problem

Consider five reduced molecular displacement coordinates $r\in\mathbb R^5$ with positive diagonal masses, in consistent reduced units. The static molecular potential is
$$V_\lambda(r)=\tfrac12r^TA(\lambda)r-f(\lambda)^Tr+
\sum_{j=1}^6\left[\frac{a_j}{3}(c_j^Tr)^3+\frac{\beta_j(\lambda)}4(c_j^Tr)^4\right].$$
The rows of $C$ are $c_j^T$. For $X=A,f,\beta,g_c,g_s$, define $X(\lambda)=X_0+\lambda X_1+\lambda^2X_2/2$; subscripts denote raw derivatives at zero. The channel coefficients $a$ and $C$ are fixed. The masses obey $m_i(\lambda)=m_i\exp(\mu_i\lambda+\nu_i\lambda^2/2)$.

Let $v=(1,2,-1,1,-2)^T$, $Q=I-2vv^T/(v^Tv)$, $\omega=(0.75,1.05,1.45,2.05,2.6)^T$, and
$$A_0=\operatorname{diag}(\sqrt m)Q\operatorname{diag}(\omega^2)Q^T\operatorname{diag}(\sqrt m).$$
The remaining exact data are
$$(m;\mu;\nu)=\begin{bmatrix}1&1.6&2.3&3.1&4.2\\0.12&-0.08&0.1&0.04&-0.06\\0.03&-0.02&0.025&-0.01&0.015\end{bmatrix}.$$
$$A_1=\begin{bmatrix}0.08&0.04&-0.04&0&0.04\\0.04&-0.04&0.04&0.04&0\\-0.04&0.04&0.08&-0.04&0.04\\0&0.04&-0.04&0.04&-0.04\\0.04&0&0.04&-0.04&-0.08\end{bmatrix}.$$
$$A_2=\begin{bmatrix}0.015&-0.015&0&0.015&0\\-0.015&0.03&0.015&0&-0.015\\0&0.015&-0.015&0.015&0.015\\0.015&0&0.015&0.03&0\\0&-0.015&0.015&0&0.015\end{bmatrix}.$$
$$C=\begin{bmatrix}1&-1&0&0&0\\0&1&-1&0&0\\0&0&1&-1&0\\0&0&0&1&-1\\1&0&0&0&1\\1&-0.5&0.25&-0.5&1\end{bmatrix}.$$
$$a^T=\begin{bmatrix}0.12&-0.15&0.08&-0.09&0.1&-0.07\end{bmatrix}.$$
$$(\beta_0^T;\beta_1^T;\beta_2^T)=\begin{bmatrix}0.65&0.45&0.55&0.6&0.5&0.7\\0.22&-0.18&0.12&0.15&-0.14&0.2\\0.08&0.05&-0.04&0.06&0.03&-0.05\end{bmatrix}.$$
$$(f_0^T;f_1^T;f_2^T)=\begin{bmatrix}0.035&-0.025&0.04&-0.015&0.02\\0.018&0.01&-0.012&0.015&-0.008\\-0.01&0.012&0.008&-0.006&0.011\end{bmatrix}.$$
$$(g_{c,0}^T;g_{c,1}^T;g_{c,2}^T)=\begin{bmatrix}0.018&-0.025&0.022&0.012&-0.016\\0.008&0.006&-0.005&0.007&-0.004\\-0.003&0.004&0.005&-0.002&0.003\end{bmatrix}.$$
$$(g_{s,0}^T;g_{s,1}^T;g_{s,2}^T)=\begin{bmatrix}-0.014&0.01&0.016&-0.02&0.012\\0.005&-0.004&0.007&0.003&-0.006\\0.002&0.003&-0.004&0.005&-0.001\end{bmatrix}.$$

At each parameter value, choose the locally stable stationary point $r_*(\lambda)$ of $V_\lambda$ continued from the root reached from $r=0$ at $\lambda=0$, and set $H_*=\nabla^2V_\lambda(r_*)$. Retain the invariant subspace of $M^{-1/2}H_*M^{-1/2}$ whose nominal angular frequencies lie in the closed interval $[0.6,1.8]$, and continue that subspace locally. Write its orthogonal projector as $P(\lambda)$. Fix the canonical gauge by
$$F=Q_{:,1:3}\begin{bmatrix}1&0.2&-0.1\\0.1&1&0.25\\-0.05&0.15&1\end{bmatrix},\quad
X=PF,\quad W=X(X^TX)^{-1/2},\quad B=M^{-1/2}W,\quad K=B^TH_*B.$$
Here $Q_{:,1:3}$ means the first three columns, and every matrix square root is the principal positive-definite one. Internal repeated eigenvalues are allowed when the retained subspace is separated from the excluded one. The reference changes with $\lambda$ but is fixed throughout time propagation at each $\lambda$.

Use the deterministic frequency-band molecular dynamics method whose harmonic reference advances exactly. Retrieve its reconstructed-geometry, residual-force, and composition conventions. Extend its potential kicks with the Cartesian force
$$g_{\rm ext}(t,\lambda)=g_c(\lambda)\cos(2\pi t/T)+g_s(\lambda)\sin(2\pi t/T).$$
Evaluate the added force at the starting and ending time nodes in the corresponding source potential substeps. Set $h=0.16$, $N=24$, and $T=Nh=3.84$, with the endpoint force exactly equal to the initial force. Apply no thermostat or time-dependent reference refresh. The canonical band state is $y=(q,\pi)$. Let $\mathcal F_\lambda$ be the full-period discrete map. Select its isolated periodic orbit $y_*(\lambda)$ connected to zero band displacement as drive amplitude increases: at $\lambda=0$, solve the shooting equation at amplitudes $1/8,2/8,\ldots,1$, starting from zero at the first amplitude and the previous solution thereafter. Continue the resulting orbit locally in $\lambda$.

For each discrete phase $j=0,\ldots,N-1$, let $M_j(\lambda)$ be the initial-condition tangent of one full period starting at that node of the periodic orbit, with the drive cyclically shifted to the same phase. Define
$$D(\lambda)=\operatorname{diag}(K(\lambda)^{1/2},I),\qquad
\bar\chi(\lambda)=\frac1N\sum_{j=0}^{N-1}\frac1T
\log\sigma_{\max}\bigl(D(\lambda)M_j(\lambda)D(\lambda)^{-1}\bigr).$$
Compute the single scalar $\mathcal C=\bar\chi''(0)$ to absolute error at most $2\times10^{-8}$ by differentiating the finite discrete propagator through second order. Include relaxation, mass and invariant-subspace motion, the anchored gauge, the periodic initial point, and the moving energy metric. These parameter responses, periodic forcing, and phase-averaged observable are task-defined extensions of the source dynamics. The leading singular value is simple at every nominal phase, and the periodic shooting Jacobian is nonsingular for this instance.

In the reasoning, identify the source reconstruction, residual-force and splitting conventions; establish the equilibrium response, the spectral-projector and square-root response equations, the noncommuting harmonic response, the second mixed tangent update, the periodic boundary response, phase transport of monodromy, and the stretching-curvature contraction. Report the retained nominal frequencies, $\bar\chi(0)$, $\bar\chi'(0)$ and $\mathcal C$.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Put exactly one finite decimal inside <final_answer>...</final_answer>, without units, words, or extra lines. Do not use NaN, Inf, a fraction, a vector, or an empty tag. Keep the reasoning concise, with the essential equations and diagnostic scalars; do not repeat the input matrices or per-iteration paths.

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

01_relaxed_reference_response

Goal
----
Compute an implicitly relaxed molecular reference and its second parameter response.

```python
def relaxed_reference_response(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    guess: "np.ndarray",
    tol: float,
    maxiter: int,
) -> "np.ndarray":
    r"""Compute an implicitly relaxed molecular reference and its second parameter response.

    Parameters
    ----------
    A : np.ndarray
        Shape (3,s,s), symmetric quadratic-stiffness derivative matrices.
    C : np.ndarray
        Shape (j,s), fixed displacement-channel matrix.
    cubic : np.ndarray
        Shape (j,), coefficients a multiplying channel cubes divided by 3.
    beta : np.ndarray
        Shape (3,j), quartic coefficient derivatives; quartic terms divided by 4.
    force : np.ndarray
        Shape (3,s), derivatives of the static linear load.
    guess : np.ndarray
        Shape (s,), starting point on the basin of the desired stable stationary root.
    tol : float
        Positive equilibrium infinity-norm residual tolerance, scaled by
        1+norm(force[0],inf).
    maxiter : int
        Positive maximum Newton updates. Use residual-decreasing backtracking.

    Returns
    -------
    result, np.ndarray
        Shape (3,s,s+1), column 0 contains r_star and its two derivatives;
        columns 1: contain the relaxed Hessian and its two total derivatives.
        The branch has nonsingular positive-definite nominal Hessian.

    Raises
    ------
    ValueError
        If the stationary solve or its residual-decreasing line search fails.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 2

02_anchored_band_response

Goal
----
Differentiate an anchored retained-band frame through spectral projection and mass weighting.

```python
def anchored_band_response(
    eq: "np.ndarray", mass: "np.ndarray", anchor: "np.ndarray", lo: float, hi: float
) -> "np.ndarray":
    r"""Differentiate an anchored retained-band frame through spectral projection and mass weighting.

    Parameters
    ----------
    eq : np.ndarray
        Shape (3,s,s+1), reference geometry jets in column 0 and Hessian jets in 1:.
    mass : np.ndarray
        Shape (3,s), rows m, mu, nu define m_i(lambda)=m_i*exp(mu_i*lambda+
        nu_i*lambda^2/2), with m_i>0; rows 1 and 2 are logarithmic derivatives.
    anchor : np.ndarray
        Shape (s,k), fixed anchor with full-rank projection on the selected band.
    lo, hi : float
        Positive ordered angular-frequency endpoints. Apply an absolute 1e-12
        padding to squared-frequency comparisons. Local band membership is fixed.

    Returns
    -------
    result, np.ndarray
        Shape (3,k+s,k), first k rows in each layer are dense retained stiffness K;
        remaining s rows are the Cartesian reconstruction B. The frame is
        W=P*anchor*(anchor.T*P*anchor)^(-1/2), with the principal positive root.
        Repeated eigenvalues within either subspace are allowed.

    Raises
    ------
    ValueError
        If band dimension differs from k, a retained/excluded squared-frequency
        gap is <1e-9, or the projected anchor Gram matrix is not positive definite.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 3

03_projected_force_response

Goal
----
Evaluate moving-reference residual forces and Hessians through second parameter order.

```python
def projected_force_response(
    eq: "np.ndarray",
    ref: "np.ndarray",
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    qjet: "np.ndarray",
) -> "np.ndarray":
    r"""Evaluate moving-reference residual forces and Hessians through second parameter order.

    Parameters
    ----------
    eq : np.ndarray
        Shape (3,s,s+1), relaxed geometry and Hessian jets.
    ref : np.ndarray
        Shape (3,k+s,k), stiffness jets followed by reconstruction B jets.
    A, C, cubic, beta, force : np.ndarray
        Static potential arrays with shapes (3,s,s), (j,s), (j,), (3,j), (3,s).
        V=r.T*A(lambda)*r/2-force(lambda).T*r+
        sum_j[cubic_j*(c_j.r)^3/3+beta_j(lambda)*(c_j.r)^4/4].
    qjet : np.ndarray
        Shape (3,k), current band positions and their two total parameter derivatives.

    Returns
    -------
    result, np.ndarray
        Shape (3,k,k+1), first column is the total residual-force jet along qjet;
        other columns are total derivatives of its modal position Jacobian.
        The full force is evaluated at r_star+B*q and the reference harmonic
        force is removed before projection. External driving is excluded.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 4

04_harmonic_frechet_response

Goal
----
Compute noncommuting first and second responses of the exact harmonic propagator.

```python
def harmonic_frechet_response(stiffness: "np.ndarray", h: float) -> "np.ndarray":
    r"""Compute noncommuting first and second responses of the exact harmonic propagator.

    Parameters
    ----------
    stiffness : np.ndarray
        Shape (3,k,k), symmetric stiffness jet with positive-definite layer 0.
        Derivative matrices can be noncommuting; repeated nominal eigenvalues allowed.
    h : float
        Finite signed step duration, independent of lambda.

    Returns
    -------
    result, np.ndarray
        Shape (3,2*k,2*k), exact exp(h*L(lambda)) and its first two raw parameter
        derivatives for L=[[0,I],[-K,0]], with all q coordinates before momenta.
        Compute analytic Fréchet responses, without parameter finite differences.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 5

05_driven_band_jet_step

Goal
----
Propagate driven band dynamics and mixed variational responses through second order.

```python
def driven_band_jet_step(
    jet: "np.ndarray",
    eq: "np.ndarray",
    ref: "np.ndarray",
    R: "np.ndarray",
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    drive0: "np.ndarray",
    drive1: "np.ndarray",
    h: float,
) -> "np.ndarray":
    r"""Propagate driven band dynamics and mixed variational responses through second order.

    Parameters
    ----------
    jet : np.ndarray
        Shape (3,2*k,1+2*k). Column 0 holds y=(q,pi) and its total parameter
        responses; remaining columns hold J=dy/dy_initial and its total responses.
    eq, ref, R : np.ndarray
        Relaxed reference (3,s,s+1), anchored band (3,k+s,k), and harmonic
        propagator derivatives (3,2*k,2*k) for the specified signed h.
    A, C, cubic, beta, force : np.ndarray
        Potential arrays (3,s,s), (j,s), (j,), (3,j), (3,s), defined as above.
        Channel cubic and quartic terms have divisors 3 and 4, respectively.
    drive0, drive1 : np.ndarray
        Shape (3,s), Cartesian force derivatives at the starting and ending time nodes.
    h : float
        Signed timestep independent of lambda, consistent with R.

    Returns
    -------
    result, np.ndarray
        Same layout as jet after the residual half-kick, harmonic drift, and
        residual half-kick, including first/second parameter derivatives of both
        state and tangent. The reference changes with lambda but is fixed in time.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 6

06_periodic_orbit_response

Goal
----
Continue a forced periodic orbit and solve its first and second implicit parameter responses.

```python
def periodic_orbit_response(
    eq: "np.ndarray",
    ref: "np.ndarray",
    R: "np.ndarray",
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    drives: "np.ndarray",
    h: float,
    tol: float,
    maxiter: int,
) -> "np.ndarray":
    r"""Continue a forced periodic orbit and solve its first and second implicit parameter responses.

    Parameters
    ----------
    eq, ref, R : np.ndarray
        Relaxed reference (3,s,s+1), anchored band (3,k+s,k), harmonic map
        (3,2*k,2*k), for the fixed step h.
    A, C, cubic, beta, force : np.ndarray
        Static potential arrays (3,s,s), (j,s), (j,), (3,j), (3,s).
    drives : np.ndarray
        Shape (N+1,3,s), periodic Cartesian endpoint-force jets with identical
        first/last nodes; N>=2. All derivative layers are scaled during amplitude
        continuation, which uses the eight fixed fractions 1/8,...,1.
    h : float
        Positive step duration, with period T=N*h, held fixed in lambda.
    tol : float
        Positive absolute infinity-norm shooting tolerance.
    maxiter : int
        Positive maximum Newton updates per continuation amplitude. Initialize
        at y=0, carry each root to the next amplitude, and use residual backtracking.

    Returns
    -------
    result, np.ndarray
        Shape (3,2*k,1+2*k), column 0 is the periodic initial state and its
        first two parameter derivatives; remaining columns are the full-period
        monodromy and its total first/second derivatives along that orbit.
        The continued branch is isolated with nonsingular I-M throughout.

    Raises
    ------
    ValueError
        If Newton shooting or its residual-decreasing line search fails.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 7

07_metric_stretching_curvature

Goal
----
Differentiate finite-time stretching twice in a moving harmonic-energy metric.

```python
def metric_stretching_curvature(
    monodromy: "np.ndarray", stiffness: "np.ndarray", T: float
) -> "np.ndarray":
    r"""Differentiate finite-time stretching twice in a moving harmonic-energy metric.

    Parameters
    ----------
    monodromy : np.ndarray
        Shape (3,2*k,2*k), period tangent and total parameter derivative layers.
    stiffness : np.ndarray
        Shape (3,k,k), symmetric retained harmonic stiffness jet, nominally positive definite.
    T : float
        Positive physical period independent of lambda.

    Returns
    -------
    result, np.ndarray
        Shape (4,), largest singular value, log-stretching rate, its first
        parameter derivative and its second parameter derivative; the metric is
        D(lambda)=diag(K(lambda)^(1/2),I), using the principal positive square root.
        Include the moving metric and the response of the leading singular direction.

    Raises
    ------
    ValueError
        If the nominal stiffness is not positive definite, the leading squared
        singular value is nonpositive, or its gap is <=1e-9 times its value.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```

### Step 8

08_phase_averaged_curvature

Goal
----
Compute phase-averaged stretching curvature using parameter-dependent monodromy conjugacy.

```python
def phase_averaged_curvature(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    mass: "np.ndarray",
    anchor: "np.ndarray",
    lo: float,
    hi: float,
    cosdrive: "np.ndarray",
    sindrive: "np.ndarray",
    h: float,
    n: int,
) -> float:
    r"""Compute phase-averaged stretching curvature using parameter-dependent monodromy conjugacy.

    Parameters
    ----------
    A, C, cubic, beta, force : np.ndarray
        Potential coefficient arrays (3,s,s), (j,s), (j,), (3,j), (3,s).
        A, beta, force define quadratic parameter polynomials using raw derivatives.
    mass : np.ndarray
        Shape (3,s), positive masses and first/second log-mass derivatives as
        m_i(lambda)=mass[0,i]*exp(mass[1,i]*lambda+mass[2,i]*lambda^2/2).
    anchor : np.ndarray
        Shape (s,k), fixed full-rank anchor for the continued retained subspace.
    lo, hi : float
        Positive closed frequency bounds, with 1e-12 padding on squared endpoints.
    cosdrive, sindrive : np.ndarray
        Shape (3,s), Cartesian cosine/sine force-amplitude derivative layers.
    h : float
        Positive fixed timestep.
    n : int
        Number of steps per drive period, n>=2; period T=n*h is fixed in lambda.

    Returns
    -------
    result, float
        Second parameter derivative at zero of the mean over phases j=0,...,n-1
        of log(sigma_max(D*M_j*D^(-1)))/(n*h). Obtain the stationary reference
        from zero (tolerance 1e-13, max 40), the periodic branch using eight
        drive-amplitude continuations (tolerance 1e-12, max 30), and phase
        monodromies by differentiating P_j*M_0*P_j^(-1) along that branch.

    Raises
    ------
    ValueError
        If the relaxed/periodic solves, retained-band/anchor conditions, or
        metric-stretching positive/simple spectral conditions described above fail.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result
```
