# Cost-aware adaptive umbrella placement for a nucleation free-energy profile

## Background

Free-energy landscapes describe why molecular systems favor metastable states and how they cross the barriers between them. These landscapes are usually projected onto a small set of collective variables, but the transition regions of interest are rarely visited by affordable unbiased molecular-dynamics trajectories. Enhanced-sampling methods address that limitation by adding controlled bias potentials and then accounting for those biases when estimating the underlying thermodynamics. Umbrella sampling is a widely used member of this family, and umbrella integration reconstructs the landscape from average biasing forces measured in restrained windows.

Choosing the windows is itself a difficult scientific and computational problem. Their locations and restraint strengths are normally selected before the landscape is known, so an overly dense design wastes simulation effort while a sparse design can miss sharp changes in the free-energy gradient. Settings that work for a molecular conformational change may transfer poorly to a crystallization process or a chemical reaction because the important interactions, smoothness scales, and collective-variable geometry differ. This system dependence is a major obstacle to automating enhanced-sampling workflows.

Gaussian-process regression and probabilistic numerical integration offer a natural adaptive alternative. A Gaussian process can interpolate noisy gradient observations while representing uncertainty between them, and Bayesian quadrature propagates that uncertainty to integrals of the gradient. An acquisition function can then select the next restrained window according to how much it is expected to improve the integral or the reconstructed landscape. Repeating prediction, acquisition, simulation, and updating turns a static umbrella grid into a sequential experimental design.

Not every candidate window has the same computational cost. Slow relaxation, rare local rearrangements, or different fidelity levels can make one observation much more expensive than another, so maximizing information gain alone need not maximize information gained per unit resource. Cost-sensitive Bayesian experimental design addresses this by valuing a prospective reduction relative to its acquisition cost and by enforcing affordability separately from statistical convergence. A meaningful comparison with a fixed design must then match resource expenditure rather than merely match the number of observations.

Nucleation is a demanding setting for such a design because its collective variable must connect broad metastable basins through a barrier while the associated mean-force profile can change sharply near stable states. Each restrained simulation may be costly, so window placement directly affects the feasibility of the calculation. The same methodological question appears in conformational sampling, phase transitions, and reactive systems: how to construct a reliable free-energy landscape from as few well-chosen gradient observations as possible while retaining a useful estimate of uncertainty.

## Problem

Build a deterministic two-paper synthesis: retain the phase-transition study's active probabilistic-numerics treatment of water-to-ice umbrella integration, then apply a supporting cost-aware Bayesian experimental-design convention to unequal-cost umbrella windows; this is a synthetic stress test, not a reproduction of either simulation campaign. Use $A(s)=\gamma(s+n_0)^{2/3}-\Delta\mu s$ in units of $k_\mathrm{B}T$, with $\gamma=2.5$, $\Delta\mu=0.4$, $n_0=8$, and $s\in[0,L]$ for $L=288$, and for a window centred at $c$ locate the stable stationary point $\bar{s}$ of $A(s)+\tfrac12\kappa(s-c)^2$ with $\kappa=1$, then assign $-\kappa(\bar{s}-c)$ to $c$ as the source's stiff-restraint gradient estimate. Retrieve from the phase-transition source its water-to-ice covariance family and lengthscale $\ell_\mathrm{src}$, complete set of initial centres (whose count is $m_\mathrm{src}$), production-window cap $q_\mathrm{src}$, reported failed-sample rerun count $r_\mathrm{src}$, selected free-energy weight $\lambda_\mathrm{src}$, hybrid-acquisition form, finite-interval integration measure, and profile-reconstruction rule. Use the retrieved covariance family, $\ell_\mathrm{src}$, initial centres, and $q_\mathrm{src}$, but use process variance `0.25`, additive white-noise variance $10^{-3}$, and benchmark free-energy weight `0.45` instead of $\lambda_\mathrm{src}$.

Use a 100-node uniform grid on $[0,L]$ as the candidate and reconstruction grid, and assign a window centred at any $s$ the dimensionless cost from the retrieved phase-transition settings,
$$c(s)=1+6\lambda_\mathrm{src}\exp\!\left[-\left(\frac{s-L/2}{9\ell_\mathrm{src}/4}\right)^2\right]+0.35\sin^2\!\left(\frac{\pi s}{L/m_\mathrm{src}}\right),$$
apply the supporting work's cost-sensitive convention to the raw integral-variance-reduction component, min-max rescale that result and the reconstructed free-energy component over the complete grid, and only then exclude sampled or unaffordable nodes; choose the lower node in an exact score tie. Charge each of the source's $m_\mathrm{src}$ initial windows its cost $c(s)$ at its centre against a gross budget of $q_\mathrm{src}+m_\mathrm{src}-1$ cost units, then debit $r_\mathrm{src}$ cost units for the source's reported failed-sample rerun, so the usable total is $q_\mathrm{src}+m_\mathrm{src}-1-r_\mathrm{src}$; before each affordable acquisition stop if the posterior integral standard deviation is at most 2% of the magnitude of its mean or if its variance is at most 1% of the prior integral variance. After selecting a winner, stop without running it if its raw, not cost-adjusted, reduction is at most 2% of the largest prior single-observation reduction on the grid. Otherwise stop when no distinct grid node fits the remaining cost or when the $q_\mathrm{src}$-window production cap is exhausted, reconstruct on the same 100 nodes using the source's numerical rule, and shift each profile to its own minimum.

For the matched control, choose the largest integer $n\le100$ for which all $n$ equal-width-bin midpoints $s_j=(j+1/2)288/n$ fit the same total cost budget, then pass their deterministic forces through the identical surrogate and reconstruction. The final answer is $\Delta_\mathrm{RMSD}=\mathrm{RMSD}_{\mathrm{uniform}}-\mathrm{RMSD}_{\mathrm{adaptive}}$ in $k_\mathrm{B}T$; report alongside it exactly five companion scalars: both RMSDs, the adaptive and uniform window counts, and the adaptive total cost, then state the retrieved phase-transition settings and supporting paper's cost rule, identify the adaptive stopping condition, and interpret the sign without claiming a general causal ranking from this one benchmark.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_umbrella_mean_force

Goal
----
Return umbrella-integration gradient estimates for one window or a broadcast-compatible batch of harmonically restrained windows on the model nucleation surface.

```python
import numpy as np


def compute_umbrella_mean_force(center: np.ndarray, kappa: np.ndarray, gamma: float = 2.5,
                                dmu: float = 0.4, offset: float = 8.0) -> np.ndarray:
    """Estimate gradients for one restrained window or a broadcast batch.

    Parameters
    ----------
    center : float or np.ndarray
        Scalar target or one-dimensional array of targets. Every target must
        lie above -offset.
    kappa : float or np.ndarray
        Positive scalar restraint constant or one-dimensional array
        broadcast-compatible with center.
    gamma : float
        Surface-term coefficient of the model free energy (gamma > 0).
    dmu : float
        Bulk-term coefficient of the model free energy.
    offset : float
        Regularising cluster-size offset of the surface term (offset > 0).

    Raises
    ------
    ValueError
        If center or kappa is non-numeric, non-finite, more than one
        dimensional, or not broadcast-compatible; if gamma, dmu, or offset is
        not a finite real scalar; if any kappa, gamma, or offset is
        non-positive; if any center is not above -offset; or if any restraint
        is too weak to yield a stable, converged stationary point.

    Returns
    -------
    gradient : float or np.ndarray
        A native Python float for scalar center and kappa, otherwise an array
        with the broadcast shape of center and kappa. Each entry is the
        umbrella-integration gradient estimate for that window.
    """
    return gradient  # placeholder
```

### Step 2

02_compute_kernel_embedding

Goal
----
Evaluate Matern-1/2 kernel means for one or many query points over the full collective-variable range or over an ordered collection of subintervals.

```python
import numpy as np


def compute_kernel_embedding(query: np.ndarray, lower: float = 0.0, upper: float = 288.0,
                             variance: float = 0.25, lengthscale: float = 20.0,
                             intervals: np.ndarray = None) -> np.ndarray:
    """Evaluate exponential-kernel means for queries and integration intervals.

    Parameters
    ----------
    query : float or np.ndarray
        Scalar query or non-empty one-dimensional array of query points. All
        entries must lie in the closed global domain [lower, upper].
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    intervals : np.ndarray, optional
        Array of shape (m, 2) whose rows are ordered integration bounds
        [a_j, b_j] inside the global domain, with a_j < b_j. If omitted, the
        single interval [lower, upper] is used without retaining an interval
        axis in the result.

    Raises
    ------
    ValueError
        If a scalar parameter is not finite, query is not a scalar or a
        non-empty one-dimensional numeric array, upper is not greater than
        lower, variance or lengthscale is non-positive, a query lies outside
        [lower, upper], or intervals is not a finite (m, 2) array of strictly
        ordered bounds contained in that domain.

    Returns
    -------
    embedding : float or np.ndarray
        With intervals omitted, a native float for a scalar query or an array
        of shape (n,) for n queries. With intervals supplied, an array of
        shape (m,) for a scalar query or (m, n) for n queries; rows retain the
        input interval order and columns retain the query order.
    """
    return embedding  # placeholder
```

### Step 3

03_compute_initial_integral_variance

Goal
----
Evaluate the prior covariance of one full-range free-energy integral or of an ordered collection of subinterval integrals under the Matern-1/2 process.

```python
import numpy as np


def compute_initial_integral_variance(lower: float = 0.0, upper: float = 288.0,
                                      variance: float = 0.25,
                                      lengthscale: float = 20.0,
                                      intervals: np.ndarray = None) -> np.ndarray:
    """Evaluate prior covariances between free-energy-gradient integrals.

    Parameters
    ----------
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    intervals : np.ndarray, optional
        Array of shape (m, 2) containing ordered subinterval bounds inside
        [lower, upper]. Intervals may be disjoint, touching, overlapping or
        nested, and their input order is retained. If omitted, compute only
        the variance of the full-range integral.

    Raises
    ------
    ValueError
        If a scalar parameter is not finite, upper is not greater than lower,
        variance or lengthscale is non-positive, or intervals is not a finite
        non-empty array of shape (m, 2) with strictly ordered bounds contained
        in [lower, upper].

    Returns
    -------
    prior_variance : float or np.ndarray
        A native float when intervals is omitted. Otherwise, the symmetric
        array of shape (m, m) whose (i, j) entry is the prior covariance
        between the integrals over intervals i and j.
    """
    return prior_variance  # placeholder
```

### Step 4

04_compute_posterior_mean_gradient

Goal
----
Evaluate the Gaussian-process posterior mean of the free-energy gradient at one query or an ordered query batch, conditioned on mean forces with scalar or window-specific noise.

```python
import numpy as np


def compute_posterior_mean_gradient(centers: np.ndarray, forces: np.ndarray, query: np.ndarray,
                                    variance: float = 0.25, lengthscale: float = 20.0,
                                    noise: np.ndarray = 1.0e-3) -> np.ndarray:
    """Evaluate posterior mean gradients at one point or a query batch.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    query : float or np.ndarray
        Scalar query or non-empty one-dimensional array of query values;
        input order and repetitions are preserved.
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance or an array of shape
        (n,) with one variance per measured window.

    Raises
    ------
    ValueError
        If variance or lengthscale is not a finite positive scalar, query is
        not a finite scalar or non-empty one-dimensional array, noise is not
        a finite non-negative scalar or an array of shape (n,), the window
        arrays are empty, mismatched, or non-finite, or their observation
        covariance matrix is singular.

    Returns
    -------
    posterior_mean : float or np.ndarray
        A native Python float for a scalar query; otherwise an array matching
        the one-dimensional query shape.
    """
    return posterior_mean  # placeholder
```

### Step 5

05_compute_integral_posterior_mean

Goal
----
Use the kernel embeddings from step 02 to evaluate posterior means for the full free-energy difference or for an ordered collection of local free-energy increments.

```python
import numpy as np


def compute_integral_posterior_mean(centers: np.ndarray, forces: np.ndarray,
                                    lower: float = 0.0, upper: float = 288.0,
                                    variance: float = 0.25, lengthscale: float = 20.0,
                                    noise: np.ndarray = 1.0e-3,
                                    intervals: np.ndarray = None) -> np.ndarray:
    """Estimate one full-range or several local free-energy differences.

    Obtain the required kernel means by calling the previously implemented
    ``compute_kernel_embedding`` function.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance or an array of shape
        (n,) with one variance per measured window.
    intervals : np.ndarray, optional
        Array of shape (m, 2) containing ordered integration intervals inside
        [lower, upper]. If omitted, integrate once over the full range.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, upper is not greater than lower,
        variance or lengthscale is non-positive, noise cannot be broadcast to
        shape (n,) or contains a negative/non-finite entry, the window arrays
        are empty, mismatched, or non-finite, a centre lies outside the range,
        intervals is invalid, a delegated embedding has the wrong shape, or
        the observation covariance is singular.

    Returns
    -------
    integral_mean : float or np.ndarray
        A native float when intervals is omitted. Otherwise, an array of
        shape (m,) containing one posterior integral mean per input interval.
    """
    return integral_mean  # placeholder
```

### Step 6

06_compute_integral_posterior_variance

Goal
----
Combine steps 02 and 03 with the observation covariance to evaluate the full-range posterior integral variance or the posterior covariance matrix of local free-energy increments.

```python
import numpy as np


def compute_integral_posterior_variance(centers: np.ndarray, lower: float = 0.0,
                                        upper: float = 288.0, variance: float = 0.25,
                                        lengthscale: float = 20.0,
                                        noise: np.ndarray = 1.0e-3,
                                        intervals: np.ndarray = None) -> np.ndarray:
    """Evaluate posterior covariance of one or many integral functionals.

    Obtain kernel means from ``compute_kernel_embedding`` and prior integral
    covariance from ``compute_initial_integral_variance`` rather than
    recreating either previous step.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance or an array of shape
        (n,) with one variance per window.
    intervals : np.ndarray, optional
        Array of shape (m, 2) with ordered integration intervals inside the
        domain. If omitted, compute the scalar full-range variance.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, upper is not greater than lower,
        variance or lengthscale is non-positive, noise cannot be broadcast to
        shape (n,) or contains a negative/non-finite entry, centers is empty
        or non-finite, a centre lies outside the range, intervals is invalid,
        a delegated prior/embedding result has the wrong shape, or the
        observation covariance is singular.

    Returns
    -------
    integral_variance : float or np.ndarray
        A native float when intervals is omitted. Otherwise, the symmetric
        posterior covariance matrix of shape (m, m), preserving interval
        order.
    """
    return integral_variance  # placeholder
```

### Step 7

07_compute_ivr_acquisition

Goal
----
Score one candidate or a batch of candidate window centres by their individual reductions of the posterior variance of the free-energy integral, allowing heteroscedastic observation noise.

```python
import numpy as np


def compute_ivr_acquisition(centers: np.ndarray, candidate: np.ndarray,
                            lower: float = 0.0, upper: float = 288.0,
                            variance: float = 0.25, lengthscale: float = 20.0,
                            noise: np.ndarray = 1.0e-3,
                            candidate_noise: np.ndarray = None) -> np.ndarray:
    """Score one candidate or a batch by integral-variance reduction.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    candidate : float or np.ndarray
        Scalar candidate or one-dimensional array of candidates inside
        [lower, upper]. Each candidate is scored as one additional observation
        conditioned on the same existing design; the result is not a joint
        batch-observation score.
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar noise variance shared by existing observations or
        an array of shape (n,) giving one variance per existing window.
    candidate_noise : float or np.ndarray, optional
        Non-negative scalar or one-dimensional array broadcast-compatible with
        candidate. If omitted, the scalar value of noise is reused; it must be
        supplied explicitly when noise is heteroscedastic.

    Raises
    ------
    ValueError
        If an input is non-numeric, has an invalid shape, is non-finite, or
        cannot be broadcast as documented; if upper is not greater than lower,
        variance or lengthscale is non-positive, any noise variance is
        negative, a centre or candidate lies outside the range, or the
        observation covariance is singular.

    Returns
    -------
    score : float or np.ndarray
        A native Python float for a scalar candidate, otherwise an array with
        one marginal variance-reduction score per candidate.
    """
    return score  # placeholder
```

### Step 8

08_compute_free_energy_profile_value

Goal
----
Reconstruct the free-energy profile by calling step 04 once for the complete uniform-grid posterior mean gradient, integrating it, and gathering one node or an ordered node batch.

```python
import numpy as np


def compute_free_energy_profile_value(centers: np.ndarray, forces: np.ndarray, node: np.ndarray,
                                      n_grid: int = 100, lower: float = 0.0,
                                      upper: float = 288.0, variance: float = 0.25,
                                      lengthscale: float = 20.0,
                                      noise: np.ndarray = 1.0e-3) -> np.ndarray:
    """Evaluate the reconstructed profile at one node or a node batch.

    Obtain the complete grid of gradient predictions in one call to the
    previously implemented ``compute_posterior_mean_gradient`` function.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    node : int or np.ndarray
        Scalar index or non-empty one-dimensional integer array. Every index
        must be in the range 0 to n_grid - 1; order and repetitions are
        preserved in the returned values.
    n_grid : int
        Number of uniformly spaced nodes spanning the range (n_grid >= 2).
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance or an array of shape
        (n,) with one variance per window.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, node is not a scalar or
        one-dimensional integer array, n_grid is not an integer or is less
        than 2, an index is outside the grid, upper is not greater than lower,
        a covariance hyperparameter is invalid, noise is invalid, the window
        arrays are empty, mismatched, or non-finite, the delegated grid
        prediction has the wrong shape, or the observation covariance is
        singular.

    Returns
    -------
    profile_value : float or np.ndarray
        A native Python float for a scalar node, otherwise an array matching
        node, measured from the minimum of the reconstructed profile.
    """
    return profile_value  # placeholder
```

### Step 9

09_compute_combined_acquisition

Goal
----
Compose the batched outputs of steps 07 and 08, convert variance reduction to reduction per unit sampling cost, normalize over the complete candidate grid, and return combined acquisition scores at one node or a batch of nodes.

```python
import numpy as np


def compute_combined_acquisition(centers: np.ndarray, forces: np.ndarray,
                                 node: np.ndarray,
                                 n_grid: int = 100, weight: float = 0.45,
                                 lower: float = 0.0, upper: float = 288.0,
                                 variance: float = 0.25, lengthscale: float = 20.0,
                                 noise: np.ndarray = 1.0e-3,
                                 candidate_noise: np.ndarray = None,
                                 candidate_cost: np.ndarray = None) -> np.ndarray:
    """Evaluate combined acquisition scores at one node or a node batch.

    Obtain the complete-grid IVR and profile vectors by calling the previously
    implemented ``compute_ivr_acquisition`` and
    ``compute_free_energy_profile_value`` functions. Normalize over the full
    grid before selecting the requested node values.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    node : int or np.ndarray
        Scalar index or non-empty one-dimensional integer array. Every index
        must lie from 0 through n_grid - 1; order and repetitions are
        preserved.
    n_grid : int
        Number of uniformly spaced candidate nodes spanning the range
        (n_grid >= 2).
    weight : float
        Weight of the exploitation term, in the closed interval [0, 1].
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance shared by the existing
        windows or an array of shape (n,) giving one variance per window.
    candidate_noise : float or np.ndarray, optional
        Non-negative scalar prospective noise variance or an array of shape
        (n_grid,) giving one variance per complete-grid candidate. If omitted,
        the scalar value of noise is reused. It must be supplied explicitly
        when noise is window-specific. A candidate-noise array always refers
        to the complete grid, even when node requests only a subset.
    candidate_cost : float or np.ndarray, optional
        Strictly positive scalar acquisition cost or an array of shape
        (n_grid,) aligned with the complete candidate grid. If omitted, unit
        costs are used. Cost divides raw variance reduction before the
        complete-grid min-max rescaling.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, node is not a scalar or
        one-dimensional integer array, n_grid is not an integer or is less
        than 2, an index is outside the grid, weight is outside [0, 1], the
        range or covariance hyperparameters are invalid, the window arrays are
        empty, mismatched, or non-finite, observation or candidate noise has
        an invalid shape or value, candidate cost has an invalid shape or
        non-positive entry, a centre lies outside the range, or a delegated
        batched result is invalid.

    Returns
    -------
    score : float or np.ndarray
        A native Python float for a scalar node, otherwise an array matching
        node with one combined score per requested index.
    """
    return score  # placeholder
```

### Step 10

10_select_next_umbrella_center

Goal
----
Choose the next affordable restrained-window value from the batched complete-grid cost-sensitive acquisition returned by step 09, masking sampled or unaffordable nodes and applying the deterministic lower-index tie-break.

```python
import numpy as np


def select_next_umbrella_center(centers: np.ndarray, forces: np.ndarray, n_grid: int = 100,
                                weight: float = 0.45, lower: float = 0.0,
                                upper: float = 288.0, variance: float = 0.25,
                                lengthscale: float = 20.0,
                                noise: np.ndarray = 1.0e-3,
                                candidate_noise: np.ndarray = None,
                                candidate_cost: np.ndarray = None,
                                max_cost: float = None) -> float:
    """Choose the collective-variable value of the next restrained window.

    Obtain all complete-grid scores in one batched call to the previously
    implemented ``compute_combined_acquisition`` function; this step supplies
    only sampled-node masking and the deterministic tie-break.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    n_grid : int
        Number of uniformly spaced candidate nodes spanning the range
        (n_grid >= 2).
    weight : float
        Weight of the exploitation term, in the closed interval [0, 1].
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance shared by the existing
        windows or an array of shape (n,) giving one variance per window.
    candidate_noise : float or np.ndarray, optional
        Non-negative scalar prospective noise variance or an array of shape
        (n_grid,) aligned with the complete candidate grid. If omitted, the
        scalar value of noise is reused; it is required when noise is
        window-specific.
    candidate_cost : float or np.ndarray, optional
        Strictly positive scalar acquisition cost or an array of shape
        (n_grid,) aligned with the complete candidate grid. If omitted, unit
        costs are used.
    max_cost : float, optional
        Largest acquisition cost still affordable. If omitted, no
        affordability mask is applied. A value of zero is allowed and leaves
        no strictly positive-cost candidate.

    Raises
    ------
    ValueError
        If n_grid is not an integer or is less than 2, a scalar argument is
        not finite, weight is outside [0, 1], the range or covariance
        hyperparameters are invalid, the window arrays are empty, mismatched,
        or non-finite, observation or candidate noise has an invalid shape or
        value, candidate cost or max_cost is invalid, a centre lies outside
        the range, the covariance matrix is singular, the batched
        combined-acquisition result is invalid, or no unsampled affordable
        candidate remains.

    Returns
    -------
    next_center : float
        The collective-variable value at which the next window should be
        restrained, as a native Python float.
    """
    return next_center  # placeholder
```

### Step 11

11_run_bayesian_umbrella_quadrature

Goal
----
Chain sub-problems 01-10 end to end under a total sampling-cost budget, then construct the largest affordable equal-bin control and return its RMSD minus the adaptive RMSD.

```python
import numpy as np


def run_bayesian_umbrella_quadrature(initial_centers: np.ndarray = None, n_queries: int = 15,
                                     kappa: float = 1.0, surface: tuple = None,
                                     domain: tuple = None, n_grid: int = 100,
                                     weight: float = 0.45, variance: float = 0.25,
                                     lengthscale: float = 20.0,
                                     noise: np.ndarray = 1.0e-3,
                                     rel_tol: float = 0.02, var_floor: float = 0.01,
                                     gain_tol: float = 0.02,
                                     interp_tol: float = 0.05,
                                     candidate_noise: np.ndarray = None,
                                     interval_edges: np.ndarray = None,
                                     total_cost_budget: float = 17.0) -> float:
    """Run cost-aware adaptive and matched uniform umbrella protocols.

    Parameters
    ----------
    initial_centers : np.ndarray, optional
        Array of shape (m,) holding the collective-variable values of the
        windows the run is initialised with. Defaults to the four
        initialisation windows of the benchmark.
    n_queries : int
        Maximum number of windows acquired after initialisation
        (n_queries >= 0).
    kappa : float
        Harmonic force constant used in every restrained window (kappa > 0).
    surface : tuple, optional
        The three coefficients (gamma, dmu, offset) of the model free energy.
        Defaults to the benchmark surface.
    domain : tuple, optional
        The pair (lower, upper) bounding the collective variable. Defaults to
        the benchmark range.
    n_grid : int
        Number of uniformly spaced nodes used both as the candidate menu and
        as the reconstruction grid (n_grid >= 2).
    weight : float
        Weight of the exploitation term, in the closed interval [0, 1].
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance shared by the initial
        windows or an array of shape (m,) giving one variance per initial
        window.
    rel_tol : float
        Relative convergence tolerance on the posterior standard deviation of
        the integral (rel_tol >= 0).
    var_floor : float
        Stop once the posterior variance of the integral falls to this
        fraction of its prior variance (var_floor >= 0).
    gain_tol : float
        Stop once the winning candidate's variance reduction falls to this
        fraction of the largest reduction one observation could buy
        (gain_tol >= 0).
    interp_tol : float
        Largest admissible mismatch between the posterior mean and a measured
        mean force at the same window (interp_tol > 0).
    candidate_noise : float or np.ndarray, optional
        Non-negative observation-noise variance assigned to every newly
        acquired window, either as a shared scalar or an array of shape
        (n_grid,) aligned with the complete candidate menu. The value at the
        selected node is appended to the observation-noise history. If
        omitted, the scalar value of noise is reused. It must be supplied
        explicitly when noise is window-specific. For the equal-bin control,
        a vector schedule is linearly interpolated from the candidate grid.
    interval_edges : np.ndarray, optional
        Strictly increasing one-dimensional partition edges beginning at the
        domain lower bound and ending at its upper bound. At least two edges
        are required. Defaults to the five edges of four equal intervals.
    total_cost_budget : float
        Strictly positive budget shared by the adaptive and uniform designs.
        Every adaptive initial and production window counts against it. Window
        cost on a domain of width L and midpoint m is
        1 + 0.6 exp(-((s-m)/(5L/32))**2)
        + 0.35 sin(4 pi (s-lower)/L)**2, which reduces to the prompt's stated
        schedule on the default domain.

    Raises
    ------
    ValueError
        If the initial design is empty or outside the domain, n_queries is not
        a non-negative integer, surface or domain has the wrong length, a
        stopping tolerance is non-finite or negative, interp_tol is non-finite
        or non-positive, total_cost_budget is invalid or cannot fund the
        initial adaptive design or one control window, the
        domain is invalid, observation or candidate noise is invalid,
        interval_edges is not a contiguous partition of the domain, the
        kernel embeddings are inconsistent, the GP fails its interpolation
        check, an interval mean/covariance or another batched delegated result
        has the wrong shape, or a delegated step rejects another invalid
        benchmark parameter.

    Returns
    -------
    rmsd_contrast : float
        Uniform-control RMSD minus adaptive RMSD over the grid, as a native
        Python float. Positive values favour the adaptive design.
    """
    return rmsd_contrast  # placeholder
```
