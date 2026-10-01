# Relative energy error for a moving three-grid curvilinear discretization

## Background

# Scientific background

Overset discretizations cover a physical domain by several simple component grids. Their central numerical issue is the transfer of information at receiver points: direct replacement by donor values can break the stability inherited from a centered spatial discretization, especially when one grid moves relative to another. A weak mismatch term instead acts only on incoming characteristics and preserves a one-way block dependency for scalar transport.

Grid motion changes the characteristic speed through the inverse time-dependent coordinate map. The grid velocity is subtracted from the physical transport speed, while spatial stretching enters through the inverse mapping Jacobian. Evaluating that Jacobian with the same discrete derivative used on the solution is important near a boundary, where a high-order centered interior stencil is paired with a lower-order closure.

The boundary closure used here is materially harder than the common fourth/second-order example. Six boundary rows must satisfy the discrete integration identity and degree-three moment conditions while remaining compatible with a seven-point sixth-order interior stencil. Those conditions leave one scalar freedom; exact differentiation of the fourth-degree monomial by the sixth closure row selects a unique operator. This constraint-based definition prevents replacing the closure with a memorized lower-order coefficient table.

Curvilinear donor transfer also requires an inverse map. Four physical donor coordinates define a cubic map over equally spaced local computational coordinates, and the receiver must first be located in that computational coordinate before the cardinal weights are evaluated. A bracketed tangent iteration keeps the inverse inside the containing donor cell on strongly stretched but monotone grids.

The coupled matrix is triangular by blocks, so its eigenvalues are inherited from its diagonal blocks and remain in the left half-plane under the stated penalty. That asymptotic statement does not imply monotone decay in the natural discrete energy: donor blocks make the full matrix non-normal. The generalized symmetric quotient in the prompt records the largest instantaneous energy-growth rate in the positive metric induced by the diagonal norm and the grid-relative scaling, thereby distinguishing spectral stability from transient amplification.

Finally, the time-dependent geometry makes the affine operator nonautonomous. Rebuilding the metric and donor coupling at all four stage times is part of the numerical method, and the physical error norm retains every overlapping degree of freedom with its own discrete volume element.

## Problem

For the nondimensional equation $u_t+u_x=0$, compute the final relative error of the exact pulse $u_\star(x,t)=\exp[-80(x+0.4-t)^2]$ propagated with no randomness and binary64 arithmetic across three overlapping grids having $N_L=41$, $N_M=49$, and $N_R=45$ nodes, uniform computational coordinates $\xi_j=j/(N_\kappa-1)$, maps

$$
x_L=-1+0.9\psi_{0.25}(\xi_L),\qquad
x_M=-0.35+0.1\sin(2\pi t)+0.7\psi_{-0.15}(\xi_M),\qquad
x_R=0.1+0.9\psi_{0.20}(\xi_R),
\quad
\psi_\sigma(\xi)=\xi-\frac{\sigma}{\pi}\sin(\pi\xi),
$$

and physical inflow $g(t)=u_\star(-1,t)$.

On each grid form $D_\kappa=H_\kappa^{-1}Q_\kappa$ with $h=1/(N_\kappa-1)$, where $H_\kappa$ is positive diagonal, equals $h$ away from its first and last six entries, and is centrosymmetric; $Q_\kappa+Q_\kappa^T=\operatorname{diag}(-1,0,\ldots,0,1)$ and $Q_\kappa=-JQ_\kappa J$; every nonclosure row of $D_\kappa$ uses $(-1/60,3/20,-3/4,0,3/4,-3/20,1/60)/h$; the first six rows of $Q_\kappa$ vanish beyond column eight (zero-based); those rows differentiate $1,\xi,\xi^2,\xi^3$ exactly; and closure row five also differentiates $\xi^4$ exactly, which selects the unique member of the one-parameter closure family.

Transform with $\xi_t=-x_\tau\xi_x$ and use the discrete rather than analytic map derivative, so the positive nodal magnitudes are $(X_\kappa)_{jj}=|1-x_\tau^{(\kappa)}|/(D_\kappa x_\kappa)_j$, with $x_\tau=0$ on the outer grids and $x_\tau=0.2\pi\cos(2\pi t)$ on the middle grid.

For a receiver $x_r$ inside a monotone donor grid, locate its donor cell, clamp the four-point stencil origin to $s=\min(\max(j-1,0),N-4)$, map the four physical stencil nodes through the cubic cardinal basis on local coordinates $0,1,2,3$, invert that monotone cubic inside the containing cell to residual at most $10^{-14}\max(1,|x_r|)$, and use the resulting four cardinal values as donor weights rather than interpolating directly in physical space.

With $d=\operatorname{sign}(c)=1$, $E_{\rm in}=e_0e_0^T$, and penalties $\tau_L=\tau_M=\tau_R=0.75$, assemble each diagonal block as $-dX_\kappa D_\kappa-\tau_\kappa X_\kappa H_\kappa^{-1}E_{\rm in}$, impose $g(t)$ only on the upstream outer-grid incoming node, and place donor mismatch terms only in the middle-left and right-middle blocks, never overwriting a receiver value.

Order $\Phi=[u_L^T,u_M^T,u_R^T]^T$, initialize every overlapping copy from $u_\star(x,0)$, recompute geometry, metrics, donor rows, matrix, and forcing at each of the four stage times $(0,1/2,1/2,1)\Delta t$, and use the order-four stage coefficients $(1/6,1/3,1/3,1/6)$ with $\Delta t=0.001$ for exactly $1000$ updates to $t_f=1$.

At the $41$ times $t_k=k/40$, evaluate both $\alpha(M)=\max_j\operatorname{Re}\lambda_j(M)$ and

$$
\mu_P(M)=\max_{z\ne0}\frac{z^T(PM+M^TP)z}{2z^TPz},
\qquad
P=\operatorname{blockdiag}(H_LX_L^{-1},H_MX_M^{-1},H_RX_R^{-1}),
$$

require every sampled $\alpha(M)$ to be negative, and report in the reasoning the maxima of both diagnostics before returning

$$
E=\frac{\sqrt{e^TWe}}{\sqrt{u_\star^TWu_\star}},
\qquad
W=\operatorname{blockdiag}(J_LH_L,J_MH_M,J_RH_R),
\qquad
J_\kappa=\operatorname{diag}(D_\kappa x_\kappa(t_f)),
$$

with all overlap copies retained; in the brief reasoning state the closure-selection scalar, both maximum stability diagnostics, both weighted quadratic forms, and the resulting ratio.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number. Before those numerical checkpoints, explicitly
identify the negative moving-coordinate identity, the incoming-only one-way
weak donor coupling with no receiver overwrite, the computational-coordinate
inversion used to obtain donor weights, and the stability threshold
$\tau_\kappa\geq 1/2$.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

01_build_sbp63_operator

Goal
----
Recover a sixth/third-order SBP closure and an arbitrary calibration jet.

```python
def build_sbp63_operator(
    n: int,
    target_jet: np.ndarray = None,
    functional_jet: np.ndarray = None,
) -> np.ndarray:
    """Return a functionally selected operator and its runtime-order jet.

    The grid consists of ``n`` uniformly spaced nodes on ``[0, 1]``. The
    returned matrices must satisfy the normalization, reflection, support,
    interior-order, and degree-three boundary-moment conditions stated in the
    scientific background. The two equally long raw-derivative jets prescribe
    a scalar target and a Frobenius functional at zero. They select a smooth
    path through the one-parameter closure family. Jet length ``r`` requests
    derivatives zero through ``r - 1`` and may range from one to nine.

    Parameters
    ----------
    n : int
        Number of nodes; must be an integer at least eighteen.
    target_jet : np.ndarray
        ``None`` or a finite real raw-derivative jet of shape ``(r,)``, where
        ``1 <= r <= 9``. ``None`` selects ``[342523/518400, 1, 0]``.
    functional_jet : np.ndarray
        ``None`` or a finite real array of shape ``(r, 6, 9)`` containing the
        raw derivatives of ``F``. Its jet length must equal that of
        ``target_jet``. The value functional must not annihilate the
        affine-family direction. ``None`` uses a length-three jet whose value
        is the entry functional at ``[4, 5]`` and whose derivatives vanish.

    Returns
    -------
    np.ndarray
        Array of shape ``(2*r + 1, n, n)`` containing ``H`` followed by the
        interleaved raw-derivative pairs ``Q^(k)(0), D^(k)(0)`` for
        ``k = 0, ..., r - 1``.

    Raises
    ------
    ValueError
        If ``n`` is not an integer or is below eighteen; if either jet has an
        invalid shape or contains complex or non-finite data; if the jet
        lengths disagree; or if ``F(0)`` does not select a unique family member.
    """
    return None
```

### Step 2

02_compute_moving_grid_state

Goal
----
Evaluate the translating-grid velocity and the three stretched physical node vectors in the fixed order used by all later steps.

```python
def compute_moving_grid_state(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    amplitude: float = 0.1,
    period: float = 1.0,
) -> np.ndarray:
    r"""Return $x_\tau(t)$ and the stretched physical nodes of all three grids.

    Parameters
    ----------
    t : float
        Evaluation time.
    counts : np.ndarray
        Integer array ``[N_L, N_M, N_R]`` with every count at least eighteen.
    sigmas : np.ndarray
        Finite stretching parameters ``[sigma_L, sigma_M, sigma_R]``, each of
        magnitude below one.
    amplitude : float
        Translation amplitude $A$.
    period : float
        Positive translation period $T_m$.

    Returns
    -------
    np.ndarray
        Vector of length ``1 + counts.sum()`` ordered as $x_\tau$, left-grid
        nodes, middle-grid nodes, and right-grid nodes.

    Raises
    ------
    ValueError
        If ``counts`` is not an integer array of shape ``(3,)``; any count is
        below eighteen; ``sigmas`` is not a finite array of shape ``(3,)`` or has
        an entry of magnitude at least one; ``t``, ``amplitude``, or ``period``
        is not finite; or ``period`` is not positive.
    """
    return None
```

### Step 3

03_compute_donor_interpolation

Goal
----
Invert a stretched cubic donor map and return a runtime-order weight jet.

```python
def compute_donor_interpolation(
    donor_nodes: np.ndarray,
    receiver_x: float | np.ndarray,
    derivative_order: int = 6,
) -> np.ndarray:
    r"""Return cubic weights and a runtime-order physical derivative jet.

    Parameters
    ----------
    donor_nodes : np.ndarray
        Strictly increasing finite one-dimensional donor coordinates, at least
        four of them.
    receiver_x : float or np.ndarray
        One finite receiver coordinate or a nonempty one-dimensional array of
        receiver coordinates inside the closed donor interval.
    derivative_order : int
        Highest requested receiver-coordinate derivative, from zero through
        eight inclusive. The default is six.

    Returns
    -------
    np.ndarray
        For a scalar receiver, an array of shape
        ``(derivative_order + 1, donor_nodes.size)`` whose rows contain raw
        receiver-coordinate derivatives zero through ``derivative_order``.
        For ``q`` receivers, the leading shape is
        ``(q, derivative_order + 1)``. Every derivative row has at most four
        nonzero entries.

    Raises
    ------
    ValueError
        If ``donor_nodes`` is not one-dimensional, has fewer than four entries,
        contains non-finite data, or is not strictly increasing; if the
        receiver input has invalid shape, is non-finite, or leaves the closed
        donor interval; if ``derivative_order`` is not an integer from zero
        through eight; or if a safeguarded inverse-map iteration fails.
    """
    return None
```

### Step 4

04_compute_relative_metric_scales

Goal
----
Compute the positive nodal magnitudes of the physical characteristic speed relative to each stretched component grid.

```python
def compute_relative_metric_diagonals(
    c: float, grid_velocity: float, counts: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    r"""Return nodal magnitudes $|c-x_\tau|/(D_\kappa x_\kappa)$.

    Parameters
    ----------
    c : float
        Finite nonzero physical advection speed. Its sign determines the
        common incoming boundary.
    grid_velocity : float
        Middle-grid velocity $x_\tau$; the outer-grid velocities are zero.
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``, each at least eighteen.
    nodes : np.ndarray
        Concatenated physical nodes of the three grids in left-middle-right
        order, of length ``counts.sum()``.

    Returns
    -------
    np.ndarray
        Concatenated metric diagonals of length ``counts.sum()`` in
        left-middle-right order.

    Raises
    ------
    ValueError
        If ``counts`` is not an integer array of shape ``(3,)`` or any count is
        below eighteen; if ``nodes`` has the wrong length or is not finite; if
        ``c`` is zero or non-finite; if ``grid_velocity`` is not finite; if the
        middle relative characteristic has a different sign from ``c``; or if
        any discrete Jacobian or metric magnitude is not positive and finite.
    """
    return None
```

### Step 5

05_assemble_weak_overset_system

Goal
----
Assemble a moving weak-SAT affine system and a runtime-order time jet.

```python
def assemble_weak_overset_system(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    beta: float = 80.0,
    x0: float = -0.4,
    jet_order: int = 6,
) -> np.ndarray:
    r"""Assemble $[M(t)\mid b(t)]$ and a runtime-order analytic time jet.

    Parameters
    ----------
    t : float
        Evaluation time.
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``, each at least eighteen.
    sigmas : np.ndarray
        Finite stretching parameters ``[sigma_L, sigma_M, sigma_R]``, each of
        magnitude below one.
    c : float
        Finite nonzero physical advection speed. Its sign selects the incoming
        side on every grid.
    amplitude : float
        Middle-grid translation amplitude.
    period : float
        Positive translation period.
    penalties : np.ndarray
        Three finite penalties, each at least $1/2$.
    beta : float
        Positive Gaussian scaling parameter.
    x0 : float
        Initial Gaussian centre.
    jet_order : int
        Highest requested raw time derivative, from zero through eight
        inclusive. The default is six.

    Returns
    -------
    np.ndarray
        Array of shape
        ``(jet_order + 1, counts.sum(), counts.sum() + 1)``. Channel ``k`` is
        the raw derivative ``d^k[M | b]/dt^k`` at the supplied time.

    Raises
    ------
    ValueError
        For invalid ``counts`` or ``sigmas``; a penalty below ``0.5``;
        non-finite scalar inputs; zero ``c``; non-positive ``period`` or
        ``beta``; a ``jet_order`` outside zero through eight; inconsistent
        grid-relative characteristic signs; or a receiver that leaves its
        donor grid.
    """
    return None
```

### Step 6

06_compute_weighted_stability_diagnostics

Goal
----
Measure frozen decay, weighted instantaneous growth, and optional metric-induced resolvent amplification of a nonnormal system.

```python
def compute_weighted_stability_diagnostics(
    system_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    resolvent_shifts: np.ndarray = None,
) -> np.ndarray:
    r"""Return spectral, instantaneous-growth, and resolvent diagnostics.

    Parameters
    ----------
    system_matrix : np.ndarray
        Finite nonempty real square matrix ``M``.
    energy_matrix : np.ndarray
        Finite real symmetric positive-definite matrix ``P`` with the same
        shape as ``M``.
    resolvent_shifts : np.ndarray
        ``None`` or a nonempty finite one-dimensional complex array. For each
        shift $z$, return the induced-energy norm of $(zI-M)^{-1}$. Repeated
        shifts are retained in input order.

    Returns
    -------
    np.ndarray
        With ``m`` shifts, a real vector of length ``2+m``. Entry zero is the
        spectral abscissa, entry one is the largest value of
        ``x.T @ (P @ M + M.T @ P) @ x / (2 * x.T @ P @ x)``, and the remaining
        entries are the induced-$P$ resolvent norms. With ``None``, return only
        the first two entries.

    Raises
    ------
    ValueError
        If either matrix is complex, non-finite, empty, or not square; if the
        shapes differ; if ``energy_matrix`` is not symmetric positive
        definite; if the shifts have invalid shape or non-finite entries; or
        if a requested shifted system is singular.
    """
    return None
```

### Step 7

07_advance_weak_overset_rk4

Goal
----
Advance the concatenated three-grid state while rebuilding every moving-grid quantity at each of four stage times.

```python
def advance_weak_overset_rk4(
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    dt: float,
    final_time: float,
    beta: float = 80.0,
    x0: float = -0.4,
) -> np.ndarray:
    r"""Advance $\Phi=[u_L^T,u_M^T,u_R^T]^T$ to ``final_time``.

    Parameters
    ----------
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``, each at least eighteen.
    sigmas : np.ndarray
        Finite stretching parameters ``[sigma_L, sigma_M, sigma_R]``, each of
        magnitude below one.
    c : float
        Finite nonzero physical advection speed. Its sign selects the incoming
        side on every component grid.
    amplitude : float
        Middle-grid translation amplitude.
    period : float
        Positive translation period.
    penalties : np.ndarray
        Three penalties, each at least $1/2$.
    dt : float
        Positive RK4 step size dividing ``final_time`` exactly.
    final_time : float
        Positive terminal time.
    beta : float
        Positive Gaussian scaling parameter.
    x0 : float
        Initial Gaussian centre.

    Returns
    -------
    np.ndarray
        Final concatenated state of length ``counts.sum()``.

    Raises
    ------
    ValueError
        If ``dt`` or ``final_time`` is not positive and finite;
        ``final_time`` is not an integer multiple of ``dt``; ``c`` is zero or
        non-finite; ``period`` is not positive; the grid-speed magnitude is not
        below ``abs(c)``; or the integration produces a non-finite state.
    """
    return None
```

### Step 8

08_compute_moving_overset_error

Goal
----
Run the complete stability-audited moving-grid calculation and return its overlap-preserving relative physical error.

```python
def compute_moving_overset_error(
    counts: np.ndarray = None,
    sigmas: np.ndarray = None,
    c: float = 1.0,
    amplitude: float = 0.1,
    period: float = 1.0,
    penalties: np.ndarray = None,
    dt: float = 0.001,
    final_time: float = 1.0,
    beta: float = 80.0,
    x0: float = -0.4,
    spectral_samples: int = 41,
) -> float:
    r"""Return the relative composite physical SBP error $E$.

    Parameters
    ----------
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``. ``None`` selects ``[41, 49, 45]``.
    sigmas : np.ndarray
        Stretching parameters. ``None`` selects ``[0.25, -0.15, 0.20]``.
    c : float
        Finite nonzero physical advection speed. Its sign selects the incoming
        side on every grid.
    amplitude : float
        Middle-grid translation amplitude.
    period : float
        Positive translation period.
    penalties : np.ndarray
        Three penalties, each at least $1/2$. ``None`` selects three values of
        $0.75$.
    dt : float
        Positive RK4 step size dividing ``final_time`` exactly.
    final_time : float
        Positive terminal time.
    beta : float
        Positive Gaussian scaling parameter.
    x0 : float
        Initial Gaussian centre.
    spectral_samples : int
        Number of equally spaced frozen-system audit times.

    Returns
    -------
    float
        One finite relative composite-grid error.

    Raises
    ------
    ValueError
        If ``spectral_samples`` is below two; ``c`` or ``period`` is invalid;
        the grid-speed magnitude is not below ``abs(c)``; the directional
        coupling rows are inconsistent; a frozen system is not strictly
        stable; or the exact-state norm is not positive.
    """
    return None
```
