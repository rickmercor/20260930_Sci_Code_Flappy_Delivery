# Biology-Ecology-28

## Background

Plotless distance sampling is a standard way to estimate the density of trees and other sessile organisms without enumerating fixed plots. In the point-centred quarter method, the area around each random sampling point is divided into equal-angle quarters and the distance to the nearest, or l-th nearest, individual is measured in each quarter. Under complete spatial randomness these distances follow a gamma-type law in the searched area, which yields closed-form Cottam-type, Pollard-type and likelihood estimators of density.

Real populations are often aggregated through dispersal limitation, habitat heterogeneity or clonal growth, and estimators built on complete spatial randomness then underestimate density. Modelling the number of individuals in a searched sector as negative binomial, with an aggregation parameter k that recovers the Poisson case as k grows, gives a closed-form distance distribution. Its reciprocal, first and second moments lead to moment estimators of density and aggregation that do not require spatial randomness.

Field protocols usually cap the search at a maximum radius, so quarters without enough individuals inside it are right-censored. The classical corrections of Warde and Petranka and of Dahdouh-Guebas and Koedam assume complete spatial randomness. A recent unified framework instead inflates the Poisson moments using the censored proportion and replaces censored quarters in the negative-binomial moment sums by complete-spatial-randomness tail moments evaluated at an initial density. It also writes censored Poisson and negative-binomial likelihoods, and it analyses the large-sample limits of these adjusted moments to explain the remaining bias.

This benchmark tests the censored negative-binomial estimators on a survey of a clustered stand, including a joint likelihood fit and the large-sample bias of the moment-based correction for the same design. Simulating point processes, Morisita-type estimators and comparisons across mapped forest plots are not part of the task.

## Problem

The point-centred quarter method estimates plant population density from the distances between random sampling points and the nearest individuals in q equal-angle quarters around each point. Field crews usually stop searching at a maximum radius C, so a quarter with too few individuals inside C is right-censored, and the classical censoring corrections assume complete spatial randomness, which fails in clustered stands. A recent unified framework extends the moment-based and likelihood-based density estimators to right-censored quarter data under both the Poisson model and a negative binomial model, in which the number of individuals within radius r of a sampling point in one quarter is negative binomial with mean pi lambda r^2 / q and aggregation parameter k.

A survey of a clustered stand used n = 20 sampling points and q = 4 quarters and recorded, in each quarter, the distance in metres to the second-nearest tree (l = 2), searching no farther than C = 10 m; c marks a censored quarter. The four quarter distances at each point are:
Point 1: 1.58, c, 9.52, 5.80
Point 2: 8.54, 8.05, 3.38, c
Point 3: 4.52, 5.25, 5.84, c
Point 4: 5.94, 5.91, 4.41, 3.55
Point 5: 6.80, 6.42, 9.80, c
Point 6: 9.90, c, c, 5.99
Point 7: 5.13, c, 9.00, 5.61
Point 8: c, c, c, 4.40
Point 9: 6.04, c, c, 7.72
Point 10: c, 5.46, 3.63, 7.53
Point 11: c, c, c, 8.98
Point 12: 7.54, c, c, 6.22
Point 13: 7.32, 8.22, c, 7.21
Point 14: c, 4.22, 3.92, c
Point 15: 5.15, 8.01, 2.68, 6.38
Point 16: 5.86, 2.47, 3.94, 5.81
Point 17: 7.85, c, 8.32, 9.35
Point 18: c, 9.86, 5.47, 5.14
Point 19: 4.84, 2.45, 7.37, 3.03
Point 20: 7.26, 8.51, c, 6.77

Work with densities in trees per m^2 and distances in m. First compute the censored Pollard-type density estimate and use it as the initial density of the censored negative-binomial moment (Shen-type) density estimator; also compute the aggregation parameter implied by the same adjusted moments, defined as the k at which the exact negative-binomial moment ratio E[R^-1] E[R^2] / E[R] equals the corresponding ratio of the adjusted moments. Next fit lambda and k jointly by maximizing the censored negative-binomial likelihood (natural logarithm, all normalizing constants included). Finally, treat the fitted (lambda, k) as the true population and evaluate the large-sample limit of the censored Shen-type density estimator for this same design (q = 4, l = 2, C = 10 m): as the number of sampling points grows without bound, the censored fraction equals the model censoring probability, the per-quarter uncensored moment sums converge to the negative-binomial partial moments inside C, the censored Pollard-type initial density converges to its own limit, and censored quarters are still imputed with complete-spatial-randomness tail moments at that limiting initial density. Report the asymptotic relative bias of the censored Shen-type estimator, defined as the limiting Shen-type density divided by the fitted density minus one, to at least 8 significant digits.

In the reasoning, briefly describe the estimators and limits you used, then report these values to at least 8 significant digits: the number of censored quarters and the uncensored fraction; the mean of the recorded distances; the estimated expected count of individuals within the search radius; the censored Pollard-type density; the censored Shen-type density and the aggregation parameter implied by its adjusted moments; the censored negative-binomial maximum likelihood estimates of lambda and k and the maximized log-likelihood; the limiting censored Pollard-type initial density; and the limiting censored Shen-type density.

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

regularized_gamma

Goal
----
Evaluate the regularized lower and upper incomplete gamma functions P(a, x) and Q(a, x) = 1 - P(a, x).

```python
def regularized_gamma(a: float, x: float) -> list:
    """Return the regularized incomplete gamma functions [P(a, x), Q(a, x)].
 
    P(a, x) = gamma(a, x) / Gamma(a) with gamma(a, x) = integral_0^x t**(a - 1) exp(-t) dt,
    and Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x).
 
    Parameters
    ----------
    a : float
        Shape parameter, finite and strictly positive (need not be an integer).
    x : float
        Argument, finite and non-negative.
 
    Returns
    -------
    values : list of float
        [P, Q] as native Python floats, each accurate to at least 12 significant digits
        relative to the smaller of the two tails' natural scale (P(a, 0) = 0, Q(a, 0) = 1).
 
    Raises
    ------
    ValueError
        If a or x is not a finite real number, if a <= 0, if x < 0, or if the series or
        continued-fraction evaluation fails to converge.
    """
    return values
```

### Step 2

regularized_beta

Goal
----
Evaluate the regularized incomplete beta function I_x(a, b) together with its complement 1 - I_x(a, b).

```python
def regularized_beta(a: float, b: float, x: float) -> list:
    """Return [I_x(a, b), 1 - I_x(a, b)] for the regularized incomplete beta function.
 
    I_x(a, b) = B(x; a, b) / B(a, b) with B(x; a, b) = integral_0^x t**(a - 1) (1 - t)**(b - 1) dt.
 
    Parameters
    ----------
    a : float
        First shape parameter, finite and strictly positive.
    b : float
        Second shape parameter, finite and strictly positive.
    x : float
        Upper integration limit, finite with 0 <= x <= 1.
 
    Returns
    -------
    values : list of float
        [I, 1 - I] as native Python floats. The complement must be computed without
        cancellation, so that it stays accurate to at least 12 significant digits when I is
        close to one (and vice versa). I_0 = 0 and I_1 = 1.
 
    Raises
    ------
    ValueError
        If a, b or x is not a finite real number, if a <= 0 or b <= 0, if x is outside
        [0, 1], or if the continued-fraction evaluation fails to converge.
    """
    return values
```

### Step 3

poisson_adjusted_moments

Goal
----
From a right-censored point-centred quarter survey, compute the estimated censoring quantile m_hat_C and the censoring-adjusted first and second distance moments of the Poisson framework.

```python
def poisson_adjusted_moments(distances: list, C: float, ell: int) -> list:
    """Return [m_hat_C, M_1, M_2] for a right-censored point-centred quarter survey.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q equal-angle sectors (columns). Each entry is the distance
        from the sampling point to the ell-th nearest individual in that sector, a finite
        positive real, or None when the sector is censored. A recorded distance larger than C
        is also treated as censored. Every row must have the same length q >= 1.
    C : float
        Maximum search radius, finite and positive, in the same length unit as the distances.
    ell : int
        Nearest-neighbour order recorded in every sector, an integer >= 1.
 
    Returns
    -------
    values : list of float
        [m_hat_C, M_1, M_2] as native Python floats. m_hat_C solves
        gamma(ell, m) / Gamma(ell) = 1 - n0 / (n q), where n0 is the number of censored
        sectors; M_u (u = 1, 2) is the Poisson censoring-adjusted estimate of E[R^u], built from
        the sum of r^u over uncensored sectors divided by n q. With no censored sector,
        m_hat_C = inf and M_u is the plain sample moment.
 
    Raises
    ------
    ValueError
        If C is not a finite positive real, if ell is not an integer >= 1, if distances is not a
        non-empty rectangular sequence of rows with at least one sector, if an entry is neither
        None nor a finite positive real, or if every sector is censored.
    """
    return values
```

### Step 4

censored_poisson_densities

Goal
----
Compute the censored Cottam-type and censored Pollard-type density estimates from a right-censored point-centred quarter survey.

```python
def censored_poisson_densities(distances: list, C: float, ell: int) -> list:
    """Return [lambda_C, lambda_P], the censored Cottam-type and Pollard-type density estimates.

    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector (same convention as
        poisson_adjusted_moments).
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.

    Returns
    -------
    densities : list of float
        [lambda_C, lambda_P] in individuals per squared length unit, as native Python floats:
        lambda_C = q * ell / (4 * M_1**2) and lambda_P = (n * q * ell - 1) / (pi * n * M_2),
        with M_1 and M_2 the Poisson censoring-adjusted moments.

    Raises
    ------
    ValueError
        Under the same conditions as poisson_adjusted_moments (invalid C, ell or survey, or
        every sector censored).
    """
    return densities
```

### Step 5

csr_tail_moment

Goal
----
Compute the conditional moment E[R^u | R > C] of the l-th nearest-neighbour sector distance under complete spatial randomness with a given density.

```python
def csr_tail_moment(u: float, ell: int, q: int, lam: float, C: float) -> float:
    """Return E[R^u | R > C] for the ell-th nearest-neighbour distance in one of q sectors under CSR.
 
    Parameters
    ----------
    u : float
        Moment order, a finite real with ell + u / 2 > 0 (negative orders allowed).
    ell : int
        Nearest-neighbour order, an integer >= 1.
    q : int
        Number of equal-angle sectors around a sampling point, an integer >= 1.
    lam : float
        Population density used for the Poisson model, finite and positive.
    C : float
        Maximum search radius, finite and positive.
 
    Returns
    -------
    moment : float
        Native Python float. It must stay finite and accurate when the expected number of
        individuals inside the radius is very large (hundreds or more), where it approaches C**u.
 
    Raises
    ------
    ValueError
        If u, lam or C is not a finite real, if ell or q is not an integer >= 1, if lam <= 0 or
        C <= 0, if ell + u / 2 <= 0, if the incomplete gamma evaluation fails to converge, or if
        the moment is not representable as a finite float.
    """
    return moment
```

### Step 6

censored_shen_estimates

Goal
----
Compute the censored negative-binomial moment estimates of population density and aggregation parameter from a right-censored point-centred quarter survey.

```python
def censored_shen_estimates(distances: list, C: float, ell: int) -> list:
    """Return [lambda_n, k_n], the censored NBD moment estimates of density and aggregation.
 
    For u in (-1, 1, 2) the adjusted moment is
        E_u = (sum of r^u over uncensored sectors + n0 * E_CSR[R^u | R > C]) / (n q),
    where n0 is the number of censored sectors and the conditional moment uses the censored
    Pollard-type density lambda_P as the initial density. Then
        lambda_n = q (2 ell - 1) E_{-1} / (pi E_1) - q ell / (pi E_2),
    and k_n is the aggregation parameter k at which the exact negative-binomial moment ratio
    E[R^-1] E[R^2] / E[R] equals the ratio E_{-1} E_2 / E_1 of the adjusted moments.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
 
    Returns
    -------
    estimates : list of float
        [lambda_n, k_n] as native Python floats. With no censored sector the adjusted moments are
        the plain sample moments.
 
    Raises
    ------
    ValueError
        Under the conditions of poisson_adjusted_moments, if the censored Pollard-type density is
        not positive, if a conditional tail moment is invalid (see csr_tail_moment), or if the
        moment ratio makes k_n undefined (division by zero).
    """
    return estimates
```

### Step 7

nbd_censored_log_likelihood

Goal
----
Evaluate the right-censored negative-binomial log-likelihood of a point-centred quarter survey at a given density and aggregation parameter.

```python
def nbd_censored_log_likelihood(distances: list, C: float, ell: int, lam: float, k: float) -> float:
    """Return the censored NBD log-likelihood of the survey at (lam, k).
 
    The log-likelihood is the sum over uncensored sectors of log g(r; lam, k) plus n0 times
    log P(R > C; lam, k), where g is the negative-binomial density of the ell-th nearest-neighbour
    distance in one of q sectors and n0 is the number of censored sectors.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector. Rows must share one length q >= 1.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
    lam : float
        Population density (individuals per squared length unit), finite and positive.
    k : float
        Negative-binomial aggregation parameter, finite and positive.
 
    Returns
    -------
    log_likelihood : float
        Natural-log likelihood as a native Python float (including all normalizing constants).
 
    Raises
    ------
    ValueError
        If C, lam or k is not a finite positive real, if ell is not an integer >= 1, if distances is
        not a non-empty rectangular sequence of rows whose entries are None or finite positive reals,
        if every sector is censored, or if the censoring probability underflows to zero while some
        sector is censored.
    """
    return log_likelihood
```

### Step 8

nbd_censored_mle

Goal
----
Find the joint maximum likelihood estimates of population density and aggregation parameter under the right-censored negative-binomial distance model, together with the maximized log-likelihood.

```python
def nbd_censored_mle(distances: list, C: float, ell: int) -> list:
    """Return [lam_hat, k_hat, max_log_likelihood] for the censored NBD model.
 
    (lam_hat, k_hat) is the interior maximizer over lam > 0 and k > 0 of the log-likelihood
    defined in nbd_censored_log_likelihood, and max_log_likelihood is its value there.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
 
    Returns
    -------
    estimates : list of float
        [lam_hat, k_hat, max_log_likelihood] as native Python floats. lam_hat and k_hat must be
        converged to at least 10 significant digits (the score equations solved to near machine
        precision, e.g. by Newton refinement), not merely to a default optimizer tolerance.
 
    Raises
    ------
    ValueError
        Under the conditions of nbd_censored_log_likelihood, or if the log-likelihood has no
        interior maximum with k <= 1e8 (for example when it keeps increasing as k grows), or if
        the maximization fails to converge.
    """
    return estimates
```

### Step 9

nbd_truncated_moment

Goal
----
Compute the partial moment E[R^u; R <= C] of the l-th nearest-neighbour sector distance under the negative-binomial model, together with the censoring probability P(R > C).

```python
def nbd_truncated_moment(u: float, lam: float, k: float, q: int, ell: int, C: float) -> list:
    """Return [E[R^u; R <= C], P(R > C)] for the ell-th nearest-neighbour distance under the NBD model.
 
    E[R^u; R <= C] denotes the integral of r^u g(r; lam, k) over 0 < r <= C, where g is the
    negative-binomial density of the ell-th nearest-neighbour distance in one of q sectors.
 
    Parameters
    ----------
    u : float
        Moment order, a finite real with ell + u / 2 > 0 and k - u / 2 > 0.
    lam : float
        Population density, finite and positive.
    k : float
        Aggregation parameter, finite and positive.
    q : int
        Number of sectors, an integer >= 1.
    ell : int
        Nearest-neighbour order, an integer >= 1.
    C : float
        Maximum search radius, finite and positive.
 
    Returns
    -------
    values : list of float
        [partial_moment, censoring_probability] as native Python floats.
 
    Raises
    ------
    ValueError
        If u, lam, k or C is not a finite real, if lam, k or C is not positive, if q or ell is not
        an integer >= 1, if ell + u / 2 <= 0 or k - u / 2 <= 0, if an incomplete beta evaluation
        fails, or if the moment is not representable as a finite float.
    """
    return values
```

### Step 10

asymptotic_shen_bias

Goal
----
For a negative-binomial population with known density and aggregation, compute the large-sample limits of the censored Pollard-type initial density and of the censored negative-binomial moment density estimate, and the resulting asymptotic relative bias.

```python
def asymptotic_shen_bias(lam: float, k: float, q: int, ell: int, C: float) -> list:
    """Return [lambda_init_inf, lambda_n_inf, relative_bias] for the censored NBD moment estimator.
 
    In the limit n -> infinity with the censored fraction equal to p0 = P(R > C; lam, k):
      * m_inf solves gamma(ell, m) / Gamma(ell) = 1 - p0;
      * M_2_inf = E[R^2; R <= C] * Gamma(ell + 1) / gamma(ell + 1, m_inf) is the limit of the
        Poisson censoring-adjusted second moment, and lambda_init_inf = ell * q / (pi * M_2_inf)
        is the limit of the censored Pollard-type density;
      * for u in (-1, 1, 2), E_u_inf = E[R^u; R <= C] + p0 * E_CSR[R^u | R > C] with the
        conditional CSR moment evaluated at lambda_init_inf;
      * lambda_n_inf = q (2 ell - 1) E_{-1,inf} / (pi E_{1,inf}) - q ell / (pi E_{2,inf}), and
        relative_bias = lambda_n_inf / lam - 1.
    If p0 underflows to zero, all censoring corrections vanish (m_inf = inf).
 
    Parameters
    ----------
    lam : float
        True population density, finite and positive.
    k : float
        True aggregation parameter, finite with k > 1 (so that E[R^2] exists).
    q : int
        Number of sectors, an integer >= 1.
    ell : int
        Nearest-neighbour order, an integer >= 1.
    C : float
        Maximum search radius, finite and positive.
 
    Returns
    -------
    values : list of float
        [lambda_init_inf, lambda_n_inf, relative_bias] as native Python floats.
 
    Raises
    ------
    ValueError
        If lam or C is not a finite positive real, if k is not a finite real > 1, if q or ell is not
        an integer >= 1, if the limiting uncensored fraction 1 - p0 underflows to zero, or if a
        special-function evaluation fails (see nbd_truncated_moment and csr_tail_moment).
    """
    return values
```

### Step 11

censored_design_bias

Goal
----
Fit the censored negative-binomial model to a point-centred quarter survey and report the asymptotic relative bias of the censored negative-binomial moment density estimator for that fitted population and the same design.

```python
def censored_design_bias(distances: list, C: float, ell: int) -> float:
    """Return the asymptotic relative bias of the censored NBD moment estimator at the fitted population.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
 
    Returns
    -------
    relative_bias : float
        lambda_n_inf / lam_hat - 1 as a native Python float, where (lam_hat, k_hat) is the censored
        NBD maximum likelihood fit and lambda_n_inf is the large-sample limit of the censored NBD
        moment density estimate for q sectors, order ell and radius C at that population.
 
    Raises
    ------
    ValueError
        Under the conditions of nbd_censored_mle, if k_hat <= 1, or under the conditions of
        asymptotic_shen_bias.
    """
    return relative_bias
```
