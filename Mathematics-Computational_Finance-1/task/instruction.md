# Mathematics-Computational_Finance-1

## Background

An option is a bet on where a share price ends up, and pricing one means averaging the payoff over every path the price might take. Doing that by simulation is slow, so the standard trick is to move the question into frequency space: instead of tracking the price itself, you track a summary of its distribution called the characteristic function, and the price falls out of a single integral. For the classical models that summary is a formula you can type out. You evaluate it at a few dozen frequencies, add the results up with a quadrature rule, and you are done in microseconds.
Rough volatility broke that convenience. Market data says volatility is far more jagged than the classical models allow, and the fix is to drive it with noise that has a memory of its whole past. The model that results fits the data much better, but the price of the fit is that the characteristic function is no longer a formula. It is now the solution of a small differential equation with a fractional derivative in it, and the fractional derivative means every step of the solution depends on every earlier step. Worse, the equation has to be solved separately at each frequency. What was a handful of function evaluations becomes a few hundred small nonlinear solves, each of which gets more expensive the finer you make its time grid.
That sets up a budget problem nobody had before. Two knobs control the accuracy: how fine the time grid is inside each solve, and how many frequencies you sum over. Turning either one up costs work, and past a point turning one up alone is wasted, because the other is now the limiting error. Existing work tuned the two in isolation. This research ties them together: it estimates both errors, splits the target accuracy between them, and picks the grid and the frequency count that meet the target for the least total work.
The second idea is where most of the saving comes from, and it is borrowed from simulation. Rather than evaluating a fine-grid solve at every frequency, compute a cheap coarse-grid answer at many frequencies, then add a correction that is the difference between a fine and a coarse solve, evaluated at only a few. The corrections shrink fast as the grid refines, so a handful of points suffices for each, and the expensive solves are used sparingly. There is also a smaller but real gain from reshaping the quadrature rule itself: the standard rule spreads its points assuming a decay rate the integrand may not have, and stretching that rate to match the model's actual decay puts the points where the integrand still has something to contribute.

## Problem

Fourier pricing turns a European option into a deterministic contour integral of the characteristic function against the payoff transform, but the rough Heston model has no closed-form characteristic function: every quadrature point needs its own numerical solution of a fractional Riccati equation, so the time discretisation and the Fourier quadrature compete for the same budget. A recent method controls the two errors jointly, scaling the Laguerre weight to the estimated decay of the integrand and splitting the integrand into a coarse level-zero term plus a hierarchy of level differences, with the number of quadrature points chosen separately at each level so that few expensive fine-level solves are needed. Your task is to run that multilevel pricer on one small, fully specified European call and report the value it returns.
Here is the exact setup to use:
- Model:  Rough Heston with $\alpha = H + \tfrac12 = 0.62$, mean reversion $\gamma = 0.1$, volatility of volatility $\nu = 0.331$, correlation $\rho = -0.681$, initial variance $V_0 = 0.0392$ and long-run level $\theta = 0.3156$. This is the EuRos parameter set of the source paper's numerical experiments.
- Contract: European call, $S_0 = K = 1000$, maturity $T = 1$, $r = 0$.
- Riccati solver: The fractional Adams predictor--corrector scheme applied to the Volterra form $h = I_t^{\alpha} F(\xi, h)$ with $h(\xi, 0) = 0$ and     $$  F(\xi, h) = -\tfrac{1}{2}\bigl(\xi^{2} + i \xi\bigr)  + \gamma\bigl(i \xi\rho\nu - 1\bigr)h  + \frac{(\gamma\nu)^{2}}{2}\,h^{2}. $$
- Characteristic function: Use the fully discrete exponent of the source paper, the one that avoids a separate approximation of the fractional integral, so that     $$  \Phi_{\ell}(\xi) = \exp\bigl(G_{\ell}(\xi)\bigr), $$   with $G_{\ell}$ built from the nodal Riccati values on the same grid.
- Hierarchy: $$  \Delta t_{\ell} = 2^{-\ell}\,\Delta t_{0},  \Delta t_{0} = \frac{T}{32},  M_{\ell} = \frac{T}{\Delta t_{\ell}}$$
    steps at level $\ell$.
- Damping: Take the damping parameter $R$ that the source paper reports for exactly this configuration in its table of selected damping parameters, and use it unchanged.
- Laguerre scaling: Fix the scaling factor of the weight $e^{-\tilde{\sigma} u}$ from the estimated asymptotic decay rate of the characteristic function the source paper adopts, and use the same factor at every level.
- Tolerance: $\varepsilon = 2\times 10^{-3}$, apportioned between the discretisation and quadrature contributions by the paper's default rule.
- Level selection: Compute the first level difference with $\bar{N}_{comp} = 16$ quadrature points and select $L$ by the paper's asymptotic Richardson indicator, using its empirical convergence rate $p = 1 + \alpha$.
- Quadrature allocation: Cost model    $$  W_{\ell} = \Delta t_{\ell}^{-\beta}$$   with the exponent the paper states for the direct fractional Adams implementation. The level-zero algebraic fit is $A_{0} = 0.43$ with smoothness index $s_{0} = 6$; the level differences satisfy   $$   A_{\ell} = C_{A}\,\Delta t_{\ell}^{\,p}$$, with  $$ C_{A} = 4,$$  with common correction index $s = 4$. Round each relaxed optimum up to the next integer.
Run the pipeline and report the multilevel value $V_{N,L}$. The answer is graded to an absolute tolerance of $10^{-6}$. In your reasoning, report the Laguerre scaling factor, the first level difference $D_{1}$, the selected level $L$ and the allocation $\mathbf{N}$, the
total multilevel work, the level-zero quadrature term together with each level-difference term, and the value the standard unscaled Gauss-Laguerre weight would have returned at the same $L$ and $\mathbf{N}$.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the quadrature nodes, the Adams weight arrays, or the nodal Riccati tables.

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

Fractional Adams convolution weights

Goal
----
Produce the two triangular weight arrays the fractional Adams predictor-corrector scheme uses to advance the Volterra form of the fractional Riccati equation on a uniform grid. Step j+1 consumes a corrector row of length j+2 and a predictor row of length j+1, so both arrays are square and lower triangular with row 0 unused. The weights depend only on the fractional order, the step size and the number of steps, never on the Fourier argument, so they are built once per discretization level and reused at every quadrature node; that reuse is what keeps the multilevel estimator affordable.

```python
import numpy as np


def adams_convolution_weights(alpha, dt, M):
    """Convolution weights of the fractional Adams predictor-corrector scheme.

    The scheme advances the Volterra form of the fractional Riccati equation
    on the uniform grid t_j = j*dt, j = 0..M.  Step j+1 uses a corrector row
    a[j+1, 0..j+1] and a predictor row b[j+1, 0..j].  Row 0 of both is unused
    and is returned as zeros.

    Args:
        alpha (float): fractional order in (0, 1).
        dt (float): time-step size.
        M (int): number of steps.

    Expected return:
        np.ndarray of shape (2*(M+1)**2,).  The first (M+1)**2 entries are the
        corrector array a flattened row-major with shape (M+1, M+1); the next
        (M+1)**2 entries are the predictor array b in the same layout.  Entries
        outside the stated index ranges are zero.
     Raises:
        ValueError: if alpha is outside (0, 1), if dt <= 0, or if
        M < 1.
    """
    return np.zeros(2 * (M + 1) ** 2)
```

### Step 2

Solve the fractional Riccati equation

Goal
----
Advance the rough Heston Riccati solution at one complex Fourier argument across the whole grid, consuming the weight arrays from the previous step. Each step predicts a value from the history with the predictor row, evaluates the Riccati right-hand side there, and corrects with the corrector row. The solution is complex but is returned as one flat real array, real parts first, because a step may not return a tuple.

```python
import numpy as np


def solve_riccati_nodal(w_packed, xi_re, xi_im, alpha, dt, M, gam, nu, rho):
    """Nodal values of the rough Heston fractional Riccati solution.

    Solves the Volterra form of the fractional Riccati equation for one
    complex Fourier argument xi = xi_re + 1j*xi_im on the uniform grid
    t_j = j*dt, j = 0..M, by the fractional Adams predictor-corrector scheme,
    reusing the convolution weights produced by the previous step.  The
    Riccati right-hand side is

        F(xi, h) = -0.5*(xi**2 + 1j*xi) + gam*(1j*xi*rho*nu - 1)*h
                   + 0.5*(gam*nu)**2 * h**2,

    the solution starts from h(xi, 0) = 0, and both the predicted and the
    corrected values carry the reciprocal of the Gamma function of the
    fractional order.

    Args:
        w_packed: (2*(M+1)**2,) packed corrector and predictor weights for
            this alpha, dt and M, laid out as in the previous step.
        xi_re, xi_im (float): real and imaginary parts of xi.
        alpha (float): fractional order H + 1/2, in (1/2, 1).
        dt (float): time-step size.
        M (int): number of steps.
        gam, nu, rho (float): mean-reversion speed, volatility of volatility
            and correlation of the rough Heston model.

    Expected return:
        np.ndarray of shape (2*(M+1),) packed as
        [Re h_0, ..., Re h_M, Im h_0, ..., Im h_M].
        Raises:
        ValueError: if M < 1, if alpha is outside (0, 1), or if
        w_packed does not have length 2*(M+1)**2.
    """
    return np.zeros(2 * (int(M) + 1))
```

### Step 3

Fully discrete characteristic exponent

Goal
----
Turn the nodal Riccati values into the exponent of the implemented characteristic function at that level. The exponent is the deterministic drift contribution plus the time integral of a combination of the Riccati solution and its own right-hand side, taken on the same grid the solution lives on. The maturity is recovered from the grid rather than passed separately, so this step cannot silently disagree with the solver about the interval.

```python
import numpy as np


def characteristic_exponent(h_packed, xi_re, xi_im, dt, X0, r, gam, nu, rho,
                            theta, V0):
    """Fully discrete exponent of the rough Heston characteristic function.

    Takes the nodal Riccati values on the grid t_j = j*dt, j = 0..M, produced
    by the previous step, and returns the exponent of the implemented
    characteristic function at that discretization level, so that
    Phi_l(xi) = exp(G_l(xi)).  The maturity is T = M*dt.  The exponent combines
    the deterministic drift contribution with the time integral of

        J(xi, z) = theta*gam*z + V0*F(xi, z),

    where F is the Riccati right-hand side of the previous step, evaluated on
    the same grid and integrated by the quadrature rule the source paper uses
    for the fully discrete exponent.

    Args:
        h_packed: (2*(M+1),) array [Re h_0..Re h_M, Im h_0..Im h_M].
        xi_re, xi_im (float): real and imaginary parts of xi.
        dt (float): time-step size.
        X0 (float): initial log price.
        r (float): risk-free rate.
        gam, nu, rho, theta, V0 (float): rough Heston parameters.

    Expected return:
        np.ndarray of shape (2,) holding [Re G_l(xi), Im G_l(xi)].
        Raises:
        ValueError: if h_packed has odd length or holds fewer than two
        nodes, or if dt <= 0.
    """
    return np.zeros(2)
```

### Step 4

Laguerre weight scaling factor

Goal
----
Return the factor that replaces the standard exponential weight of the Gauss-Laguerre rule by one matched to the integrand at hand. The factor combines a correlation-dependent prefactor with a bracket holding a term linear in maturity and a term carrying the fractional order through a Gamma function. It depends on the model and the maturity only, not on the strike, the damping or the discretization level.

```python
import numpy as np


def laguerre_scaling(T, alpha, gam, nu, rho, theta, V0):
    """Scaling factor of the scaled Gauss-Laguerre weight.

    The Fourier integrand is integrated against the weight exp(-sigma*u) on
    (0, inf) instead of the standard exp(-u).  The factor is chosen to match
    the estimated large-frequency exponential decay rate of the rough Heston
    characteristic function along the integration contour, using the decay
    rate the source paper adopts for this purpose.  The payoff transform
    contributes only an algebraic factor and does not enter.

    Args:
        T (float): maturity.
        alpha (float): fractional order H + 1/2.
        gam, nu, rho, theta, V0 (float): rough Heston parameters.

    Expected return:
        float: the scaling factor sigma, strictly positive for |rho| < 1.
        Raises:
        ValueError: if abs(rho) >= 1, if T <= 0, or if gam*nu == 0.
    """
    return 0.0
```

### Step 5

Scaled Gauss-Laguerre quadrature rule

Goal
----
Return the nodes and weights of the N-point Gauss-Laguerre rule associated with a scaled exponential weight. The rule is the generalized Gauss-Laguerre rule with zero algebraic exponent, and it is obtained from the standard rule by a single change of scale that acts on nodes and weights alike. Nodes come back in increasing order so that callers can index the smallest and largest without sorting.

```python
import numpy as np


def scaled_laguerre_rule(N, sigma):
    """Nodes and weights of the scaled Gauss-Laguerre quadrature rule.

    The rule is exact for polynomials of degree up to 2N-1 against the weight
    exp(-sigma*u) on (0, inf); it is the generalized Gauss-Laguerre rule with
    zero algebraic exponent.  Its nodes and weights follow from those of the
    standard rule, which corresponds to sigma = 1.  Increasing sigma
    concentrates the nodes near the origin, decreasing it spreads them
    towards larger u.

    Args:
        N (int): number of quadrature points, at least 1.
        sigma (float): positive scaling factor of the exponential weight.

    Expected return:
        np.ndarray of shape (2*N,) packed as
        [u_1, ..., u_N, w_1, ..., w_N], nodes in increasing order.
        Raises:
        ValueError: if N < 1 or if sigma <= 0.
    """
    return np.zeros(2 * int(N))
```

### Step 6

Fourier integrand values

Goal
----
Combine the characteristic exponent with the payoff transform along the damped contour and return the real integrand of the Fourier valuation formula at every node. The exponent arrives already evaluated, as two real arrays, so this step does no model work and is a pure assembly of the pricing formula. The discount factor and the constant of the inverse transform are included here rather than left to the caller.

```python
import numpy as np


def fourier_integrand(u, G_re, G_im, R, K, r, T):
    """Fourier pricing integrand of a European call at the damped contour.

    With xi = u + 1j*R and Phi(xi) = exp(G(xi)), the integrand of the Fourier
    valuation formula is

        g(u) = exp(-r*T) / (2*pi) * Re[ Phi(xi) * Phat(xi) ],

    where Phat is the Fourier transform of the call payoff,

        Phat(xi) = -K**(1 - 1j*xi) / (xi**2 + 1j*xi).

    Payoff admissibility requires R < -1; the value R = -1 is the pole of Phat
    at xi = -1j.

    Args:
        u: (n,) array of non-negative Fourier variables.
        G_re, G_im: (n,) arrays holding the characteristic exponent at
            xi = u + 1j*R, as produced by the exponent step.
        R (float): damping parameter, strictly less than -1.
        K (float): strike.
        r (float): risk-free rate.
        T (float): maturity.

    Expected return:
        np.ndarray of shape (n,): the integrand values g(u).
        Raises:
        ValueError: if u, G_re and G_im differ in length, if R >= -1,
        or if K <= 0.
    """
    return np.zeros(np.asarray(u, dtype=float).size)
```

### Step 7

Select the discretization level

Goal
----
Choose the finest level of the time-step hierarchy from the first level difference and the discretization share of the tolerance. The rule extrapolates the observed convergence rate from a single difference between two consecutive levels, converts it into a predicted error at any level, and returns the smallest level at which the prediction meets the target. The level is never returned below one, since that is where the indicator is defined.

```python
import numpy as np


def select_discretization_level(D1, eps_disc, p):
    """Finest discretization level of the time-step hierarchy.

    The hierarchy is dt_l = 2**(-l) * dt_0.  The discretization error is not
    computable directly, so the level is chosen from the asymptotic
    Richardson-type indicator the source paper derives, which extrapolates the
    observed rate from the first level difference

        D1 = | V_{Nbar, 1} - V_{Nbar, 0} |

    computed with a fixed number of quadrature points.  The rule returns the
    smallest admissible level whose indicator does not exceed the
    discretization part of the tolerance, and it never returns a level below
    the one at which the indicator is defined.

    Args:
        D1 (float): magnitude of the first level difference, positive.
        eps_disc (float): discretization part of the prescribed tolerance.
        p (float): convergence order of the Fourier integrand under refinement
            of the fractional Riccati time discretization.

    Expected return:
        float: the selected level L, an integer value returned as a float.
        Raises:
        ValueError: if D1 <= 0, if eps_disc <= 0, or if p <= 0.
    """
    return 0.0
```

### Step 8

Allocate quadrature points across levels

Goal
----
Distribute the quadrature points over the hierarchy so that the total work is smallest among all allocations meeting the quadrature tolerance. Half the tolerance is reserved for the level-zero term and half for the level differences taken together, and each relaxed optimum is rounded up. The per-node cost of a level difference is the sum of the costs at the two levels it spans, because one evaluation needs a solve at each.

```python
import numpy as np


def allocate_quadrature_points(A, s0, s, eps_quad, W):
    """Number of quadrature points to place at each level of the hierarchy.

    The multilevel estimator applies the scaled Gauss-Laguerre rule with N_0
    points to the level-zero integrand and with N_l points to each level
    difference, l = 1..L.  Evaluating a level difference at one node costs one
    Riccati solve at level l and one at level l-1, so the total work is

        W_ML = W_0*N_0 + sum_{l=1..L} (W_l + W_{l-1}) * N_l.

    The algebraic quadrature error model bounds the level-zero remainder by
    A_0 * N_0**(-s0/2) and the level-l remainder by A_l * N_l**(-s/2) with a
    common correction smoothness index s.  The counts minimise the work
    subject to the quadrature part of the tolerance being met, using the
    allocation the source paper derives and the split of eps_quad it
    prescribes between the level-zero term and the level differences.  Each
    relaxed optimum is rounded up to the next integer.

    Args:
        A: (L+1,) array of algebraic quadrature constants, A[0] for the
            level-zero integrand and A[l] for the l-th level difference.
        s0 (float): smoothness index of the level-zero integrand.
        s (float): common smoothness index of the level differences.
        eps_quad (float): quadrature part of the prescribed tolerance.
        W: (L+1,) array of per-node solver costs, W[l] at level l.

    Expected return:
        np.ndarray of shape (L+1,) holding the integer counts N_0..N_L as
        floats, each at least 1.
        Raises:
        ValueError: if A and W differ in length, if A is empty, if
        eps_quad <= 0, if s0 <= 0 or s <= 0, or if any entry of A is not
        positive.
    """
    return np.zeros(np.asarray(A, dtype=float).size)
```

### Step 9

Run the multilevel Fourier pricer

Goal
----
Chain the eight earlier steps into the whole calculation and return the value the method produces. Fix the scaling factor, compute the diagnostic level difference, select the level, allocate the points, then evaluate the level-zero term and each level difference with its own point count and add them. This is the orchestrator step: it chains the eight earlier public functions rather than reproducing any of them inline.

```python
import numpy as np


def run_multilevel_pricer(model, contract, numerics):
    """Chain the eight earlier steps and return the multilevel option value.

    Fixes the Laguerre scaling factor, computes the first level difference
    with the diagnostic number of quadrature points to obtain D1, selects the
    finest level L from it, allocates the quadrature points across the
    hierarchy, and evaluates the telescoping multilevel estimator

        V = Q_{N_0}[g_0] + sum_{l=1..L} Q_{N_l}[g_l - g_{l-1}],

    where Q_N is the scaled Gauss-Laguerre rule applied to the transformed
    integrand and already carries the factor two of the even-integrand
    reduction.  The same scaling factor is used at every level; only the
    number of points changes.  The per-node solver cost model is
    W_l = dt_l**(-beta), the level-zero algebraic constant and smoothness
    index are supplied, and the level-difference constants follow
    A_l = CA * dt_l**p with the common correction index s.

    This is the orchestrator step: chain the eight earlier public functions rather than reproducing any of them inline.

    Args:
        model: (6,) array [alpha, gam, nu, rho, V0, theta].
        contract: (5,) array [S0, K, T, r, R].
        numerics: (8,) array [dt0, eps, nbar, A0, s0, CA, s, beta].

    Expected return:
        float: the multilevel Fourier price V.
        Raises:
        ValueError: if model, contract and numerics do not have
        lengths 6, 5 and 8, or if S0 <= 0.
    """
    return 0.0
```
