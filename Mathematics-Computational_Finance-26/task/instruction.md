# Mathematics-Computational_Finance-26

## Background

Every option price you see quoted is the answer to a question asked backward in time. You know what the contract pays on its last day, that part is written into the contract, and the price today is what you get by working that known ending backward through a model of how the share price wanders. Mathematically this is a diffusion run in reverse, and reverse diffusion is the comfortable direction: it smooths. Small errors in the payoff shrink as you integrate back, so the answer today is stable no matter how jagged the thing you started from was.
Turn the question around and the comfort disappears. Suppose you are handed the price profile as it stands today, across a whole range of share prices, and you want to know what profile the same model implies on the expiry date. Now you are running the diffusion forward into the past's role, against the smoothing direction, and the arithmetic reverses with it. Any wobble in the data you were handed, and real data always wobbles, stops shrinking and starts growing, faster the finer the wobble. Two profiles that look identical to the eye today can imply wildly different things at expiry. This is what mathematicians call an ill-posed problem, and it cannot be fixed by computing more carefully; the instability is in the question, not the arithmetic.
The usual response is regularisation: refuse to consider the wildest answers. You stop asking for the profile that fits the data best and start asking for the profile that fits the data well while also being reasonably smooth in time, and you tune how much you care about each. That trade-off is a knob, and turning it too far either way gives you either noise amplified into nonsense or a bland curve that has forgotten the data.
The contribution here adds a second, different idea before the regularisation is applied. Rather than tracking the price at every one of a hundred grid points, it describes the whole profile by a handful of coefficients against a fixed family of polynomials, the Legendre family, and follows those coefficients through time instead. That does two useful things at once. It throws away the fine wiggles by construction, because a short polynomial expansion simply cannot represent them, which is exactly the content that would have exploded. And it quietly repairs a second problem: the equation carries a factor of the share price squared, which collapses to nothing at the left-hand edge where the share price is zero, and the polynomial description does not notice that edge the way a grid does.
What is left after the reduction is a small system of ordinary differential equations for the coefficients, and the regularisation is applied to that instead of to the original equation. The reconstruction is then the coefficient trajectory that best balances obeying those equations, matching the observed profile at today's date, and staying smooth across the horizon. The paper proves this has a unique answer that depends continuously on the data, which is the property the original question lacked, and shows that the reduction itself, not just the regularisation, is doing real stabilising work.

## Problem

Option pricing normally runs backward: the payoff is prescribed at maturity and today's price follows by integrating the Black-Scholes equation back in time. Reversing the question, prescribing the price profile observed today and asking what profile the same equation implies at expiry, turns a well-posed parabolic problem into an unstable one, because the evolution now runs against the smoothing direction of the operator and any high-frequency content in the observed data is amplified without bound. A recent contribution stabilises this by projecting the asset-price variable onto a finite basis of shifted Legendre polynomials before regularising, which acts as a spectral cutoff and at the same time relaxes the degeneracy that the factor $S^2$ creates at the zero-price boundary; the reduced coefficient trajectory is then recovered as the minimiser of a Tikhonov functional in time, at a regularisation weight the method selects for itself rather than one supplied to it. Your task is to run that reconstruction on one small, fully specified instance of the source paper's butterfly-spread experiment and report the recovered price at a single asset value.

Here is the exact setup to use:

- Domain and model: $S \in [0, S_{\max}]$ with $S_{\max} = 10$, risk-free rate $r = 0.05$, horizon $T = 1.5$. The volatility is the state-dependent smile of the source paper's numerical experiments, $\sigma(t,S) = \sigma_0\sqrt{1 + \eta\,e^{-t/T}\left((S - S_{\mathrm{ref}})/S_{\mathrm{ref}}\right)^2}$, with $\sigma_0 = 0.2$, $\eta = 0.25$ and $S_{\mathrm{ref}} = S_{\max}/2$.
- Observed profile: the clean observation at $t = 0$ is synthetic, and no closed-form price is involved. Impose the butterfly payoff $\Phi(S) = (S-K_1)^+ - 2(S-K_2)^+ + (S-K_3)^+$ with $K_1 = 3$, $K_2 = 5$, $K_3 = 7$ at $t = T$, and march the terminal-value Black-Scholes problem with the volatility above back to $t = 0$ on the uniform grid $S_i = i\,\Delta S$, $i = 0,\dots,100$, $\Delta S = 0.1$, using the explicit finite-difference scheme of the source paper's data-generation section together with that section's rule for the number of time steps and its treatment of the two endpoint values.
- Noise: corrupt that profile at noise level $\delta = 0.10$ using the noise model of the source paper's data-generation section, with the multipliers drawn once, in index order, as `np.random.default_rng(2026).uniform(-1.0, 1.0, size=101)`.
- Basis and reduction: use the shifted Legendre basis on $[0, S_{\max}]$ that the source paper defines, with the paper's per-mode scaling, truncated at $N = 8$, so nine modes $n = 0,\dots,8$. Project the forward-time equation onto that basis and assemble the reduced coefficient matrix $C(t)$ exactly as the paper derives it from the diffusion and convection integrals; the reduced system is $\mathbf{u}'(t) = C(t)\mathbf{u}(t)$.
- Quadrature: every entry of the two reduced matrices is computed by Gauss-Legendre quadrature with $40$ nodes mapped affinely onto $[0, S_{\max}]$. The Legendre coefficients of the noisy observation are computed instead by the composite trapezoidal rule on the $101$-point grid above.
- Time grid for the reconstruction: uniform, $t_k = k\,\Delta t$, $k = 0,\dots,40$, with $\Delta t = T/40$. The unknown is the stacked coefficient trajectory $\mathbf{v}(t_k) \in \mathbb{R}^{9}$.
- Discretisation of the functional: write the minimisation as one overdetermined least-squares system. The ODE-residual term contributes one block per interval, evaluated at the $40$ interval midpoints, with the forward difference quotient for $\mathbf{v}'$ and the arithmetic average of the two endpoint states for $\mathbf{v}$, each block scaled by $\sqrt{\Delta t}$. The data-misfit term contributes a single identity block of weight $1$. Every regularisation term contributes one block per grid point at which it is defined, scaled by $\sqrt{\alpha\,\Delta t}$, using the forward difference quotient for a first time derivative and the standard three-point central quotient for a second one. The regularisation term is taken in the norm the source paper specifies in its Tikhonov functional. Row order is immaterial to the least-squares solution. Solve each assembled system with `numpy.linalg.lstsq(..., rcond=None)`.
- Choice of the regularisation weight: $\alpha$ is not given. Sweep the eleven candidates $\alpha_j = 10^{\,j-9}$, $j = 0,\dots,10$, in that order of increasing weight, solving the system once for each. For every candidate evaluate the two quantities that the source paper's parameter-choice section plots against each other, the residual quantity $R_j = \left(\int_0^T \lvert \mathbf{v}_j' - C\mathbf{v}_j\rvert^2\,dt + \lvert \mathbf{v}_j(0) - \mathbf{u}^0_\delta\rvert^2\right)^{1/2}$ and the regularisation norm $Q_j = \lVert \mathbf{v}_j \rVert_{H^2(0,T;\mathbb{R}^{9})}$, each discretised with exactly the midpoint, difference and $\sqrt{\Delta t}$ rules used for the corresponding rows of the assembled system. Put $x_j = \ln R_j$ on the horizontal axis and $y_j = \ln Q_j$ on the vertical axis, and select the interior candidate that maximises the signed curvature $\kappa_j = (x_j' y_j'' - x_j'' y_j')/(x_j'^2 + y_j'^2)^{3/2}$, where the first and second derivatives are central differences in $j$ with unit spacing. Compare the signed values of $\kappa_j$, not their absolute values, and break ties towards the lowest index. The reported reconstruction is the one belonging to the selected candidate.

Run the reconstruction and report $u_{\alpha}^{N}(T, S^{\star})$, the recovered price at $S^{\star} = 6.25$, obtained by expanding the terminal coefficient vector of the selected candidate back into the basis. The answer is graded to an absolute tolerance of $10^{-6}$. In your reasoning, report the clean synthetic observation at $S = 5$ before any noise is applied, the first two Legendre coefficients of the noisy observation, the Euclidean norm of the full nine-component coefficient vector, the $(2,2)$ entry of the reduced diffusion matrix at the first midpoint, the selected candidate index and the weight it corresponds to, $R_j$ and $Q_j$ at that candidate, the first two coefficients of the recovered terminal state, the least-squares residual norm at the minimiser, the reconstructed value at $S = 5$, and the value the same pipeline returns at $S^{\star}$ when the sweep is skipped and $\alpha$ is fixed at $3.2\times 10^{-5}$ instead.

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

Shifted_legendre_basis_table

Goal
----
Evaluate the shifted, normalised Legendre basis and its first two derivatives at a given set of asset prices, and return the three tables packed into one flat array. Every later step consumes this table: the projection of the observed profile needs the functions themselves, the convection integral needs the first derivatives, the diffusion integral needs the second, and the final expansion back to a price needs the functions again. Building it once, from a stable recurrence, is what keeps the whole pipeline consistent.

Evaluate the basis that the source paper's dimension reduction is built on, together with its first two derivatives in the asset price, at a given set of points, and return the three tables packed into one flat array. Every later step consumes this table: the projection of the observed profile needs the functions themselves, the convection integral needs the first derivatives, the diffusion integral needs the second, and the final expansion back to a price needs the functions again. Building it once, from a stable recurrence, is what keeps the whole pipeline consistent.

The basis is the Legendre family carried from its reference interval onto the price interval and rescaled. The rescaling is not free to choose: the paper fixes a particular constant for each mode, and that constant is what makes the reduction behave the way the rest of the method assumes. Every reduced matrix entry downstream carries the product of two of these constants, so a basis that is merely orthogonal rather than scaled the paper's way produces matrices that are wrong by a different factor in every row and column, and no later step recovers.

The change of variable onto the price interval is affine, so each differentiation contributes a constant factor from the chain rule. Those factors are easy to drop, and dropping them leaves a table that looks plausible mode by mode while being wrong by a fixed power throughout. The second-derivative table is where the damage shows first, because the reduced diffusion operator is built from it.

Evaluate the polynomials by the three-term recurrence rather than by forming explicit coefficients, and differentiate the recurrence itself to get the two derivative tables. Building monomial coefficients and differentiating those loses several digits by the eighth mode.

```python
import numpy as np


def shifted_legendre_basis_table(S, N, smax):
    """Reduction basis and its first two derivatives on the price interval.

    Evaluates the Legendre-type basis that the source paper's dimension
    reduction uses, at the given asset prices, together with dell/dS and
    d2ell/dS2, and packs the three tables into a single flat array.  The
    per-mode scaling constant is the paper's.

    Args:
        S (np.ndarray): asset prices, each in [0, smax].
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.

    Expected return:
        np.ndarray of shape (3*(N+1)*len(S),).  The first (N+1)*len(S)
        entries are the basis values in row-major order over (mode, point),
        the next block the first derivatives and the last block the second
        derivatives, in the same layout.

    Raises:
        ValueError: if N < 0, if smax <= 0, if S is empty, or if any entry
        of S lies outside [0, smax].
    """
    return np.zeros(3 * (N + 1) * np.asarray(S).size)
```

### Step 2

observed_price_profile

Goal
----
Build the observation the reconstruction starts from: the synthetic option-price profile at today's date on a uniform price grid, and the corrupted copy that stands in for market data. Return both, packed end to end, so that later steps and tests can compare them.

The clean profile is not a closed-form price. It is manufactured the way the source paper manufactures the data for its numerical experiments: a butterfly-spread payoff is imposed at the horizon, and the ordinary terminal-value pricing problem, carrying the same state-dependent volatility that the reconstruction uses later, is marched back to today with the explicit finite-difference scheme of the paper's data-generation section. The stencil, the instant at which the coefficients are frozen within a step, the rule that fixes the number of time steps and the way the two endpoint values are filled are all the paper's, and none of them is supplied here. Each one moves the profile by more than the downstream tolerance, so a profile produced by a more accurate solver of the same equation is not a better answer to this step, it is a different one.

The corruption follows the rule the source paper uses to generate noisy data in its numerical experiments. How the noise level enters, and what it scales against, is the paper's convention and it is load bearing: option prices across a strike range span orders of magnitude, so a rule that treats the whole grid on one footing and a rule that treats each point on its own give perturbations that differ by orders of magnitude in the wings. Where the rule is applied matters as much as its form, because the profile is projected onto a small basis immediately afterwards and a perturbation applied before that projection is not the same object as one applied after it.

The draws are consumed once, in grid order, from a single seeded generator. Drawing them in a different order, or drawing more than the grid needs and slicing, gives a different perturbation and therefore a different reconstruction, so the order is part of the specification rather than an implementation detail.

```python
import numpy as np


def observed_price_profile(ns, smax, strikes, T, sigma0, eta, sref, r, delta,
                           seed):
    """Synthetic and noise-corrupted option-price profile on a uniform grid.

    The clean profile is the value at t = 0 of a butterfly spread whose
    payoff at t = T is (S - K1)^+ - 2 (S - K2)^+ + (S - K3)^+, computed on the
    grid S_i = i*smax/ns, i = 0..ns, by the explicit finite-difference
    data-generation scheme of the source paper, with that scheme's time-step
    rule and its treatment of the two endpoint values, under the local
    volatility sigma(t,S) = sigma0*sqrt(1 + eta*exp(-t/T)*((S - sref)/sref)**2)
    and rate r.  The noisy profile applies the paper's data-generation noise
    rule at level delta, using the draws of a single seeded generator consumed
    in grid order.

    Args:
        ns (int): number of grid intervals, so ns+1 grid points.
        smax (float): right end of the price grid.
        strikes (np.ndarray): shape (3,), the strikes K1 < K2 < K3.
        T (float): horizon at which the payoff is imposed.
        sigma0 (float): base volatility level.
        eta (float): smile curvature parameter.
        sref (float): reference price of the smile.
        r (float): risk-free rate.
        delta (float): noise level of the paper's data-generation rule.
        seed (int): seed of the generator supplying the draws.

    Expected return:
        np.ndarray of shape (2*(ns+1),).  The first ns+1 entries are the
        clean profile at t = 0, the next ns+1 the corrupted profile.

    Raises:
        ValueError: if ns < 2, if smax <= 0, if strikes is not three strictly
        increasing positive values, if T <= 0, if sigma0 <= 0, if sref <= 0,
        if eta < 0, or if delta < 0.
    """
    return np.zeros(2 * (ns + 1))
```

### Step 3

project_profile_onto_basis

Goal
----
Project a profile sampled on a price grid onto the reduced basis, returning its Legendre coefficients. This is the step that turns a hundred-and-one grid values into nine numbers, and it is where the noise the previous step injected stops being pointwise and becomes a perturbation of the reduced state.

The coefficient of a mode is the integral of the profile against that mode over the price interval, with no weight function in the integrand and nothing to invert afterwards. Whether the vector that comes out has the same length as the function it represents depends on how the basis was scaled upstream, which is settled in the step that builds it and not here; this step simply integrates against whatever table it is given.

The integral is evaluated by the composite trapezoidal rule on the grid that carries the data, not by a high-order rule. This is deliberate and it is not a compromise: the data are only known at those points, so a rule that samples elsewhere would be integrating an interpolant rather than the data, and the interpolation error would enter the reconstruction as an unmodelled perturbation on top of the noise the experiment prescribes.

The projection is where the reduction does its stabilising work. High-frequency content in the noisy profile has small inner products against the first few modes and is simply not represented in the output, which is exactly the content that the unstable forward evolution would otherwise amplify.

```python
import numpy as np


def project_profile_onto_basis(S, profile, N, smax):
    """Legendre coefficients of a profile sampled on a price grid.

    Coefficient m is the integral of profile(S) * ell_m(S) over the price
    interval, evaluated by the composite trapezoidal rule on the supplied
    grid, with ell_m the reduction basis of the source paper.

    Args:
        S (np.ndarray): grid points, increasing, spanning the interval.
        profile (np.ndarray): profile values at those points.
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.

    Expected return:
        np.ndarray of shape (N+1,), the Legendre coefficients.

    Raises:
        ValueError: if S and profile have different lengths, if fewer than
        two grid points are supplied, if N < 0, or if smax <= 0.
    """
    return np.zeros(N + 1)
```

### Step 4

reduced_convection_matrix

Goal
----
Assemble the reduced convection matrix, the projection of the first-order drift term of the pricing equation onto the basis, in the form the source paper's reduction uses. Entry (m, n) pairs mode m with mode n; which of the two is differentiated, and whether a derivative is moved across by parts, is the paper's convention and is not free to choose here.

Unlike its diffusion counterpart the matrix carries no time dependence and no volatility, so it is built once and reused at every time level. It is also strongly structured, and the structure is worth predicting before computing anything: the drift term does not raise polynomial degree, and the basis is graded by degree, so most of the matrix is forced to vanish and part of what survives is fixed by leading coefficients alone. An implementation that produces a dense or symmetric matrix has the pairing or the index order wrong.

The integrand is a polynomial of bounded degree, so a Gauss-Legendre rule with enough nodes evaluates it exactly rather than approximately. Once the node count clears that bound the answer stops changing, which means the node count is not a tuning parameter and the matrix is determined to machine precision by the basis alone.

The nodes and weights supplied by a standard Gauss-Legendre routine live on the reference interval and must be mapped affinely onto the price interval, the weights picking up the same scaling factor as the interval length. Forgetting the weight scaling leaves every entry short by a constant factor.

```python
import numpy as np


def reduced_convection_matrix(N, smax, nq):
    """Projection of the first-order drift term onto the reduced basis.

    Entry (m, n) is the projection of the first-order drift term onto the
    reduction basis, in the pairing the source paper's reduction specifies,
    evaluated by nq-node Gauss-Legendre quadrature mapped affinely onto
    [0, smax].

    Args:
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.
        nq (int): number of Gauss-Legendre nodes.

    Expected return:
        np.ndarray of shape ((N+1)**2,), the matrix flattened row-major,
        entry (m, n) at index m*(N+1)+n.

    Raises:
        ValueError: if N < 0, if smax <= 0, or if nq < 1.
    """
    return np.zeros((N + 1) ** 2)
```

### Step 5

reduced_diffusion_matrix

Goal
----
Assemble the reduced diffusion matrix at a given time, the projection of the second-order term of the pricing equation onto the basis, in the form the source paper's reduction uses. Entry (m, n) weights the pairing of modes m and n by the squared volatility and the squared asset price. Which mode carries the second derivative, and whether the expression is left as written or transformed before integrating, is the paper's convention.

That choice is the single most consequential thing about this step. The two natural readings differ by a boundary term that does not vanish here, because the coefficient of the second-order term degenerates at one end of the interval and not at the other, and they produce matrices with different symmetry. Both give a well-behaved computation; only one gives the paper's method.

The time dependence enters only through the volatility, which is a smile in the asset price whose curvature decays over the horizon. Because the squared volatility is polynomial in the asset price, the whole integrand is polynomial too, of degree bounded by the truncation level, so a Gauss-Legendre rule with enough nodes is exact and the entry is determined to machine precision. Evaluating the smile at the wrong instant, at a grid node instead of the interval midpoint the caller asks for, is a more likely error than any quadrature issue.

Two columns of the matrix vanish identically whatever the volatility, for reasons that follow from the lowest two modes alone. Those are the quickest check that the derivative table feeding this step is correct.

```python
import numpy as np


def reduced_diffusion_matrix(t, N, smax, nq, sigma0, eta, sref, T):
    """Projection of the second-order term onto the reduced basis at time t.

    Entry (m, n) is the projection of the second-order term onto the
    reduction basis at time t, in the form the source paper's reduction
    specifies, evaluated by nq-node Gauss-Legendre quadrature mapped affinely
    onto [0, smax].  The volatility is
    sigma(t,S) = sigma0 * sqrt(1 + eta * exp(-t/T) * ((S - sref)/sref)**2).

    Args:
        t (float): time at which the volatility is evaluated.
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.
        nq (int): number of Gauss-Legendre nodes.
        sigma0 (float): base volatility level.
        eta (float): smile curvature parameter.
        sref (float): reference price of the smile.
        T (float): horizon setting the decay of the smile.

    Expected return:
        np.ndarray of shape ((N+1)**2,), the matrix flattened row-major,
        entry (m, n) at index m*(N+1)+n.

    Raises:
        ValueError: if N < 0, if smax <= 0, if sref <= 0, if T <= 0, if
        nq < 1, or if eta < 0.
    """
    return np.zeros((N + 1) ** 2)
```

### Step 6

reduced_coefficient_matrix

Goal
----
Combine the two reduced operator blocks into the coefficient matrix that drives the reduced evolution. The combination is fixed by the source paper's reduction, and it is the single place in the pipeline where the direction of time is encoded.

The pricing equation has three terms, and the reduction turns each into a contribution to this matrix: the second-order term through the diffusion block, the first-order term through the convection block, and the zeroth-order discounting term through something simpler, because the projection of multiplication by a constant onto the basis is not an integral at all. The coefficients that multiply the three contributions, and the signs they carry, come from writing the equation in the direction the reconstruction runs rather than the direction pricing normally runs. Every sign reverses between the two, and the third contribution is the one most often dropped, because in the ordinary direction it reads as bookkeeping rather than as part of the operator.

The result is a square matrix of the same size as its two inputs, non-symmetric, and time-dependent through the diffusion block alone. One of its columns collapses to something trivial for reasons that involve only the lowest mode, which is the cheapest available check on the whole assembly chain: if that column comes out dense, the error is upstream in one of the two operator blocks or in an index order, not here.

The step takes both blocks as flattened square arrays and returns the combination in the same layout, so it can be applied to whichever time level the caller needs without rebuilding anything.

```python
import numpy as np


def reduced_coefficient_matrix(a_flat, b_flat, r):
    """Coefficient matrix of the reduced forward-time evolution.

    Combines the reduced diffusion block A and the reduced convection block B
    into the matrix C that satisfies the reduced system u'(t) = C(t) u(t) for
    the forward-time pricing equation, in the combination the source paper's
    reduction gives.

    Args:
        a_flat (np.ndarray): reduced diffusion matrix at the required time,
            flattened row-major, square.
        b_flat (np.ndarray): reduced convection matrix, flattened row-major,
            square and of the same size.
        r (float): risk-free rate.

    Expected return:
        np.ndarray of shape ((N+1)**2,), the coefficient matrix flattened
        row-major in the same layout as the inputs.

    Raises:
        ValueError: if either input is not a flattened square matrix, or if
        the two do not have the same shape.
    """
    return np.zeros(np.asarray(a_flat).size)
```

### Step 7

assemble_tikhonov_system

Goal
----
Turn the Tikhonov minimisation into one overdetermined linear least-squares problem and return the assembled matrix and right-hand side packed into a single flat array. This is where the regularisation is actually chosen, because the rows that implement the penalty are the only record of which norm it is taken in, and that norm is the source paper's.

Minimising a sum of squared quadratic terms is the same as stacking the discretised operator of each term into one tall matrix and solving in the least-squares sense, provided each block carries the square root of its own weight. A term that is an integral over the horizon and a term that is a single point evaluation do not scale the same way with the time step, and a term carrying a regularisation weight picks that up as well. Getting one of those square roots wrong does not produce a wrong-looking system, it produces a system that is regularised by the wrong amount.

The residual term contributes one block per time interval, evaluated at the midpoint, with the difference quotient standing for the derivative and the average of the two endpoint states standing for the state. The data term is a single identity block and is the only source of a non-zero right-hand side. The regularisation term contributes one block per grid point at which each of its constituents is defined, which is why those blocks do not all have the same count: an undifferentiated quantity is defined everywhere, a first difference loses one point and a second difference loses two. How many such groups there are is determined by the norm the paper specifies, so the height of the assembled system is not something the caller dictates.

The number of unknowns is the number of grid points times the number of modes. Row ordering has no effect on the solution, since permuting the rows permutes the terms of a sum.

```python
import numpy as np


def assemble_tikhonov_system(u0d, c_stack, T, alpha):
    """Assemble the Tikhonov minimisation as one least-squares system.

    Builds the augmented matrix and right-hand side whose least-squares
    solution minimises the discretised Tikhonov functional over the stacked
    coefficient trajectory v(t_0), ..., v(t_nt) on a uniform grid of nt
    intervals.  The residual blocks are evaluated at the nt interval
    midpoints with the forward difference quotient for the derivative and the
    average of the two endpoint states for the state, scaled by sqrt(dt); the
    data block is a single identity of weight one; the penalty blocks are
    scaled by sqrt(alpha*dt), using the forward difference quotient wherever
    a first time derivative appears in the paper's norm and the three-point
    central quotient wherever a second one does.

    Args:
        u0d (np.ndarray): shape (N+1,), Legendre coefficients of the data.
        c_stack (np.ndarray): the nt coefficient matrices at the interval
            midpoints, each (N+1) by (N+1), flattened row-major and
            concatenated in time order.
        T (float): horizon.
        alpha (float): regularisation weight.

    Expected return:
        np.ndarray of shape (nrow*ncol + nrow,) with ncol = (nt+1)*(N+1).
        The first nrow*ncol entries are the augmented matrix flattened
        row-major, the remaining nrow the right-hand side.  The row count
        nrow follows from the terms of the functional.

    Raises:
        ValueError: if u0d is empty, if T <= 0, if alpha <= 0, if the length
        of c_stack is not a multiple of (N+1)**2, or if it implies fewer
        than two time intervals.
    """
    return np.zeros(1)
```

### Step 8

solve_tikhonov_system

Goal
----
Solve the assembled least-squares system and return the coefficient trajectory that minimises the discretised Tikhonov functional. The input is the packed matrix and right-hand side produced by the previous step; the output is the stacked trajectory, one block of coefficients per time level.

The system is tall and thin, because the residual, data and regularisation terms each contribute their own rows while the unknowns are only the trajectory itself. It has no exact solution and is not meant to, since the data are noisy and the penalty actively pulls the answer away from a perfect fit. The residual at the minimiser is therefore small but non-zero, and a computation that reports a residual at the level of machine precision has solved a square system somewhere instead of the overdetermined one.

Solve by least squares rather than by forming and inverting the normal equations. The regularisation makes the system well conditioned enough that both routes agree far inside the required tolerance here, but the normal equations square the condition number, and the margin that buys is worth keeping for the harder configurations the same code has to serve.

Only the number of columns is passed in. The number of rows is whatever the assembly produced, and it is recoverable from the length of the packed input, since the packing is a matrix of that width followed by one right-hand-side entry per row. A length that is not consistent with the stated width means an inconsistent call rather than a recoverable situation.

```python
import numpy as np


def solve_tikhonov_system(sys_flat, ncol):
    """Least-squares solution of the assembled Tikhonov system.

    Unpacks the augmented matrix and right-hand side from the packed array.
    The matrix has ncol columns; its row count is recovered from the length
    of sys_flat, which holds nrow*ncol matrix entries followed by nrow
    right-hand-side entries.

    Args:
        sys_flat (np.ndarray): the packed system, matrix flattened row-major
            followed by the right-hand side.
        ncol (int): number of columns, that is (nt+1)*(N+1).

    Expected return:
        np.ndarray of shape (ncol,), the minimising stacked trajectory;
        entries k*(N+1) to (k+1)*(N+1) are the coefficient vector at time
        level k.

    Raises:
        ValueError: if ncol < 1, if the length of sys_flat is not a multiple
        of ncol+1, or if the implied system is not overdetermined.
    """
    return np.zeros(ncol)
```

### Step 9

lcurve_corner_index

Goal
----
Select the regularisation weight from a family of candidate solutions by locating the corner of the L-curve. The step receives two arrays of the same length, one entry per candidate in order of increasing weight, and returns the index of the candidate the criterion selects. It performs no solves of its own: the caller has already produced the family and evaluated both quantities for every member, and this step is the decision rule alone.

A regularised problem has no single right answer until the weight is fixed, and the weight cannot be read off the data because the noise level alone does not determine it. The standard device is to plot two competing quantities of the family against each other on logarithmic axes and take the point where the curve turns most sharply. The first argument is the residual quantity of each candidate and is placed on the horizontal axis; the second is the regularisation norm and is placed on the vertical axis. What enters each of the two quantities is fixed by the source paper's parameter-choice section and is the caller's responsibility; this step only needs them in that order.

The turn is measured by the signed curvature of the plane curve through the points (log of the first quantity, log of the second), traversed in the order the candidates are supplied: the product of the first derivative of the horizontal coordinate with the second derivative of the vertical one, minus the product of the second derivative of the horizontal coordinate with the first derivative of the vertical one, divided by the sum of the squared first derivatives raised to the power three halves. Take every derivative by central differences in the candidate index with unit spacing, consider only interior candidates, compare the signed values rather than their magnitudes, and break ties towards the lowest index. Swapping the axes or reversing the order of traversal flips the sign of every curvature, so either slip turns the most concave point into the selection.

The curvature is scale-invariant in a useful way: multiplying either quantity by a positive constant shifts its logarithm by a constant and leaves every difference unchanged, so the selected candidate does not depend on the units either quantity is measured in, nor on the base of the logarithm.

```python
import numpy as np


def lcurve_corner_index(residual_quantity, regularisation_norm):
    """Index of the L-curve corner over a family of candidate solutions.

    Both arguments hold one entry per candidate, ordered by increasing
    regularisation weight.  With x_j = log(residual_quantity[j]) on the
    horizontal axis and y_j = log(regularisation_norm[j]) on the vertical
    axis, the corner is the interior candidate that maximises the signed
    curvature
        kappa_j = (x'_j y''_j - x''_j y'_j) / (x'_j**2 + y'_j**2)**1.5,
    with x', y' the central first differences and x'', y'' the central
    second differences in the index j (unit spacing).  The signed value is
    compared, not its magnitude, and ties go to the lowest index.

    Args:
        residual_quantity (np.ndarray): shape (J,), strictly positive.
        regularisation_norm (np.ndarray): shape (J,), strictly positive.

    Expected return:
        float: the selected candidate index, strictly between 0 and J-1.

    Raises:
        ValueError: if the two arrays have different lengths, if fewer than
        three candidates are supplied, or if either array has a non-positive
        entry.
    """
    return 0.0
```

### Step 10

run_legendre_tikhonov_reconstruction

Goal
----
Chain the nine earlier steps into the whole reconstruction and return the recovered price at the requested asset value. Build the synthetic observation and corrupt it, project it onto the basis, assemble the convection block once and the diffusion block at each interval midpoint, combine them into the coefficient matrices, then sweep the candidate regularisation weights: for each one assemble and solve the least-squares system and evaluate the two L-curve quantities the selection rule compares. Hand those to the selection step, take the candidate it returns, and expand that solution's terminal coefficients back into a price. This is the orchestrator step: it chains the nine earlier public functions rather than reproducing any of them inline.

The regularisation weight is not supplied. It is chosen from the candidate ladder by the source paper's parameter-choice criterion, which is what makes the answer a property of the method rather than of a number somebody picked. The two quantities are the ones that section defines, a residual quantity and a regularisation norm, and they are evaluated on the solution with the same quadrature and the same difference rules that the assembled system uses for the corresponding terms of the functional, so that each is exactly what the least-squares rows measure rather than a separately discretised approximation of it. A pipeline that skips the sweep and uses a single plausible weight produces a number that looks entirely reasonable and is wrong, because the selected candidate is a discrete choice and the reconstruction jumps between neighbouring candidates rather than drifting.

What the method returns is not the true payoff and is not meant to be. It is the output of a regularised reconstruction at the weight the criterion selects, and every choice in the procedure leaves a fingerprint on the number: how the data were manufactured, how the basis is scaled, which direction the reduced generator runs in, which norm the penalty is taken in, where the noise is applied, and how the corner of the L-curve is located. Asking for the exact payoff would erase all of that, because every reasonable variant converges to the payoff as the noise and the weight go to zero.

The arrangement is free of adaptivity beyond that one selection. The truncation level, the grid counts, the quadrature size, the candidate ladder and the noise seed are all supplied, so nothing depends on a convergence test that could terminate differently on a different machine. The coefficient matrix is evaluated at interval midpoints, not at grid nodes, and the same midpoint value is used for both endpoint contributions of its block.

```python
import numpy as np


def run_legendre_tikhonov_reconstruction(market, numerics):
    """Chain the nine earlier steps and return the reconstructed price.

    Builds the synthetic observation and corrupts it, projects it onto the
    reduction basis, assembles the reduced convection block once and the
    reduced diffusion block at each of the nt interval midpoints, combines
    them into the reduced coefficient matrices, then for each candidate
    weight alpha_j = 10**(lo + j), j = 0..nalpha-1, assembles and solves the
    Tikhonov system and evaluates the two L-curve quantities of the source
    paper's parameter-choice section for that candidate, discretised with the
    same quadrature and difference rules as the assembled system.  The corner
    of the resulting L-curve selects the candidate, and the terminal
    coefficient vector of that candidate's solution is evaluated against the
    basis at sstar.

    This is the orchestrator step: chain the nine earlier public functions
    rather than reproducing any of them inline.

    Args:
        market: (9,) array [smax, r, sigma0, eta, sref, T, K1, K2, K3].
        numerics: (9,) array
            [N, nt, delta, seed, ns, nq, sstar, lo, nalpha], with the
            candidate ladder alpha_j = 10**(lo + j) for j = 0..nalpha-1.

    Expected return:
        float: the reconstructed price at sstar.

    Raises:
        ValueError: if market and numerics do not both have length 9, if
        smax <= 0, if sstar lies outside [0, smax], or if nalpha < 3.
    """
    return 0.0
```
