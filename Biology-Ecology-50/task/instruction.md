# Estimate mosquito barrier permeability from mother and mixed-stage full-sibling kin

## Background

Adult mosquitoes may disperse daily among habitat sites, while their eggs and larvae remain at natal sites. Sampled mother–larva and full-sibling pairs therefore carry information about the locations and timing of maternal egg-laying, adult survival, and movement across a possible barrier.

A sampled larva’s laying day is unobserved. Its sibling may emerge as an adult, move before sampling, and have been laid either before or after the reference. The two laying orders require different spatial conditioning. At demographic equilibrium, surviving output from all females supplies separate denominators for sampled larvae and sampled adults. Combining the two kinship classes in a log pseudo-likelihood allows simultaneous inference on daily staying probability and barrier permeability.

## Problem

Adult mosquitoes move among sites, whereas larvae remain at their egg-laying sites. Spatial close-kin mark-recapture uses where and when related mosquitoes are sampled to estimate dispersal across a landscape barrier. Fit the crossing factor \(\delta\) using mother–larva and larva–adult full-sibling records, treating the daily staying probability \(p_0\) as an unknown nuisance parameter. The four site locations, unique kin records and fixed life-history parameters are the inputs; the requested output is the fitted \(\delta\).

Use the paper's zero-inflated, distance-dependent adult dispersal kernel and apply the factor \(\delta\) to movements that cross the vertical line \(x=10\,\mathrm{m}\). Nodes on this line belong to its right side. The inverse distance scale conditional on leaving a site is \(\lambda_c=1/30\,\mathrm{m}^{-1}\). For the mixed-stage full siblings, account for the unknown laying dates, maternal survival and movement in either laying order, survival through egg, larva, pupa and adult stages, and the target adult's movement after emergence. Combine their count contributions with the mother–larva contributions in the paper's log pseudo-likelihood, omitting binomial coefficients independent of \(\delta\) and \(p_0\).

Use inclusive integer days and the source's inclusive adult transition convention. Assume a stationary site population and equal prior weights for possible earlier maternal locations before conditioning on a known later location. At every site let \(N_F=25\), \(\beta=20\), \(T_E=2\), \(T_L=5\), \(T_P=1\), \(T_A=8\), and \((\mu_A,\mu_E,\mu_L,\mu_P)=(0.09,0.175,0.554,0.175)\). The mother records list a sampled-female stratum and target larvae; the sibling records list one reference larva and target adults, and repeated site/day strata refer to distinct references. Use each listed record once. An adult target cannot be the same individual as a reference larva, even at the same site and day. Globally maximize the joint score over \(0.1\leq\delta\leq0.9\) and \(0.05\leq p_0\leq0.95\), including boundaries. Report \(\delta\) within \(10^{-6}\); in the reasoning, show the fitted \(p_0\), first movement row, the larval and adult denominators, the first mother and mixed-stage sibling probabilities, and both category scores.

## Sites

| Index | \(x\) (m) | \(y\) (m) |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 20 |
| 2 | 20 | 0 |
| 3 | 20 | 20 |

## Mother–larva records

Each row is \((x_1,t_1,x_2,t_2,n_{F,s},n_{L,s},k)\): female stratum, target larval stratum, number of sampled females, number of target larvae, and number of matched larvae.

| Row | \(x_1\) | \(t_1\) | \(x_2\) | \(t_2\) | \(n_{F,s}\) | \(n_{L,s}\) | \(k\) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 3 | 1 | 4 | 5 | 100 | 4 |
| 2 | 2 | 3 | 3 | 4 | 5 | 100 | 7 |
| 3 | 0 | 4 | 2 | 5 | 5 | 100 | 1 |
| 4 | 2 | 4 | 0 | 5 | 5 | 100 | 5 |
| 5 | 0 | 5 | 3 | 6 | 5 | 100 | 1 |
| 6 | 2 | 5 | 1 | 6 | 5 | 100 | 2 |
| 7 | 0 | 6 | 0 | 7 | 5 | 100 | 11 |
| 8 | 2 | 6 | 2 | 7 | 5 | 100 | 8 |
| 9 | 0 | 7 | 1 | 8 | 5 | 100 | 4 |
| 10 | 2 | 7 | 3 | 8 | 5 | 100 | 1 |
| 11 | 0 | 8 | 2 | 9 | 5 | 100 | 3 |
| 12 | 2 | 8 | 0 | 9 | 5 | 100 | 2 |

## Larva–adult full-sibling records

Each row is \((x_1,t_1,x_2,t_2,n_{A,s},k)\): one reference larva, a target adult stratum, the number of sampled target adults, and the number of matched full siblings. Each row represents one reference and is counted once.

| Row | \(x_1\) | \(t_1\) | \(x_2\) | \(t_2\) | \(n_{A,s}\) | \(k\) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 3 | 0 | 11 | 40 | 0 |
| 2 | 0 | 3 | 1 | 11 | 40 | 0 |
| 3 | 2 | 3 | 2 | 11 | 40 | 0 |
| 4 | 0 | 3 | 3 | 11 | 40 | 0 |
| 5 | 0 | 3 | 0 | 14 | 40 | 0 |
| 6 | 2 | 3 | 1 | 14 | 40 | 1 |
| 7 | 0 | 3 | 2 | 14 | 40 | 0 |
| 8 | 2 | 3 | 3 | 14 | 40 | 0 |
| 9 | 0 | 4 | 0 | 12 | 40 | 0 |
| 10 | 2 | 4 | 1 | 12 | 40 | 1 |
| 11 | 0 | 4 | 2 | 12 | 40 | 0 |
| 12 | 2 | 4 | 3 | 12 | 40 | 0 |
| 13 | 2 | 4 | 0 | 15 | 40 | 0 |
| 14 | 0 | 4 | 1 | 15 | 40 | 0 |
| 15 | 2 | 4 | 2 | 15 | 40 | 0 |
| 16 | 0 | 4 | 3 | 15 | 40 | 0 |
| 17 | 2 | 5 | 0 | 13 | 40 | 0 |
| 18 | 0 | 5 | 1 | 13 | 40 | 0 |
| 19 | 2 | 5 | 2 | 13 | 40 | 0 |
| 20 | 0 | 5 | 3 | 13 | 40 | 0 |
| 21 | 0 | 5 | 0 | 16 | 40 | 0 |
| 22 | 2 | 5 | 1 | 16 | 40 | 0 |
| 23 | 0 | 5 | 2 | 16 | 40 | 0 |
| 24 | 2 | 5 | 3 | 16 | 40 | 0 |
| 25 | 0 | 6 | 0 | 14 | 40 | 0 |
| 26 | 2 | 6 | 1 | 14 | 40 | 0 |
| 27 | 0 | 6 | 2 | 14 | 40 | 2 |
| 28 | 2 | 6 | 3 | 14 | 40 | 1 |
| 29 | 2 | 6 | 0 | 17 | 40 | 0 |
| 30 | 0 | 6 | 1 | 17 | 40 | 0 |
| 31 | 2 | 6 | 2 | 17 | 40 | 0 |
| 32 | 0 | 6 | 3 | 17 | 40 | 0 |

## Output format requirements

Emit `<final_answer>` immediately followed by `<reasoning>`. Put exactly one finite decimal barrier factor inside `<final_answer>...</final_answer>`, with no units or prose. Explain the calculation and requested intermediate checks inside `<reasoning>...</reasoning>`.

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

01_movement_kernel.py

Goal
----
Construct a zero-inflated, barrier-adjusted daily movement matrix.

```python
def movement_kernel(coords: "np.ndarray", lam: float, barrier_x: float, delta: float, p0: float) -> "np.ndarray":
    """Construct a zero-inflated, barrier-adjusted daily movement matrix.

Parameters
----------
coords : float array of shape (N,2), at least two distinct locations
lam : positive inverse-distance scale conditional on leaving a site
barrier_x : vertical barrier x coordinate
delta : crossing factor in (0,1]
p0 : daily probability of staying at the origin, in [0,1)

Returns
-------
(N,N) nonnegative row-stochastic array, with diagonal entries equal to p0.

Notes
-----
For j != i, use distance-dependent exponential weights with a factor delta on cross-barrier movements, normalize only over j != i, then allocate total probability 1-p0 to those destinations. A node on the barrier lies on the right. This is the zero-inflated kernel in article Eq. 2 together with the barrier adjustment; p0 is exactly the diagonal probability, so do not renormalize the diagonal with off-diagonal weights.
"""
    return None
```

### Step 2

02_adult_transition.py

Goal
----
Propagate adult locations using the inclusive daily movement convention.

```python
def adult_transition(M: "np.ndarray", t0: int, t1: int) -> "np.ndarray":
    """Propagate adult locations using the inclusive daily movement convention.

    Parameters
    ----------
    M : row-stochastic daily movement matrix
    t0 : integer origin day
    t1 : integer destination day with t1 >= t0

    Returns
    -------
    (N,N) float array, adult transition probabilities

    Notes
    -----
    Return the paper's forward movement probability for the interval: the daily movement matrix raised to 1+t1-t0. The same-day transition therefore includes one movement opportunity. Raise ValueError if t1<t0.
    For all ordered nodes i,j, the transition from day t0 to t1 is (M ** (1+t1-t0))[i,j], where ** denotes matrix power.

    Raises
    ------
    ValueError : if the destination day precedes the origin day.
    """
    return None
```

### Step 3

03_reverse_origin.py

Goal
----
Infer the earlier location given an adult location at sampling.

```python
def reverse_origin(M: "np.ndarray", current: int, t0: int, t1: int) -> "np.ndarray":
    """Infer the earlier location given an adult location at sampling.

    Parameters
    ----------
    M : row-stochastic daily movement matrix
    current : integer sampled node index
    t0 : integer earlier day
    t1 : integer sampling day

    Returns
    -------
    (N,) float array, conditional probabilities for each earlier node

    Notes
    -----
    Condition on the sampled destination by normalizing the destination column of the forward transition matrix over possible origin nodes. This uses equal origin weights as in the source expression.
    For an earlier node i and sampled current node j, return (M ** (1+t1-t0))[i,j] / sum_h (M ** (1+t1-t0))[h,j]. Do not normalize an origin row.
    """
    return None
```

### Step 4

04_larval_denominator.py

Goal
----
Calculate equilibrium surviving larval output at a sampled node.

```python
def larval_denominator(nf: float, beta: float, te: int, tl: int, mu_e: float, mu_l: float) -> float:
    """Calculate equilibrium surviving larval output at a sampled node.

    Parameters
    ----------
    nf : equilibrium adult female count at node
    beta : eggs per female per day
    te : egg-stage duration in days
    tl : larval-stage duration in days
    mu_e : daily egg mortality
    mu_l : daily larval mortality

    Returns
    -------
    float, expected surviving larval offspring at the site

    Notes
    -----
    Return the stationary expected surviving larval output over all TL admissible egg-laying days using the paper's egg and larval survival factors.
    E_L = nf * beta * (1-mu_e)**te * sum((1-mu_l)**a for a=0,...,tl-1).
    """
    return None
```

### Step 5

05_mother_larva.py

Goal
----
Calculate spatial mother to larva kinship probability.

```python
def mother_larva(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, ta: int, mu_a: float, mu_e: float, mu_l: float) -> float:
    """Calculate spatial mother to larva kinship probability.

    Parameters
    ----------
    M : adult daily movement matrix
    x1,t1 : sampled mother node and day
    x2,t2 : sampled larva node and day
    nf,beta,te,tl,ta,mu_a,mu_e,mu_l : demographic and stage parameters

    Returns
    -------
    float, probability a sampled female is mother of the sampled larva

    Notes
    -----
    Marginalize integer egg-laying days consistent with larval age and possible maternal age. Apply adult survival, backward maternal origin probability, egg and larval survival, then divide by stationary larval output.
    For each integer y from t2-te-(tl-1) through t2-te, keep t1-ta < y <= t1; add (1-mu_a)**(t1-y) * reverse_origin(M,x1,y,t1)[x2] * beta * (1-mu_e)**te * (1-mu_l)**(t2-y-te), then divide the sum by E_L.
    """
    return None
```

### Step 6

06_adult_denominator.py

Goal
----
Compute the equilibrium surviving adult offspring output at one site.

```python
def adult_denominator(nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    """Compute the equilibrium surviving adult offspring output at one site.

Parameters
----------
nf : equilibrium adult female count at the site
beta : eggs laid per female per day
te,tl,tp : fixed durations of egg, larva and pupa stages in days
ta : maximum adult age in days, covering ages 0 through ta-1
mu_a,mu_e,mu_l,mu_p : corresponding daily mortality probabilities

Returns
-------
float, expected adult output from all females at one site at a fixed sampling day.

Notes
-----
Under the paper's stationary population assumption, each possible adult age contributes surviving output. Let s_q=1-mu_q for q in {A,E,L,P}. Calculate nf*beta*s_E**te*s_L**tl*s_P**tp*sum(s_A**a for a=0,...,ta-1). The sum includes age zero and is zero only for an empty adult-age range."""
    return None
```

### Step 7

07_larva_adult_movement.py

Goal
----
Calculate the composite spatial factor for a larva and its adult sibling.

```python
def larva_adult_movement(M: "np.ndarray", x1: int, x2: int, y1: int, y2: int, t2: int, te: int, tl: int, tp: int) -> float:
    """Calculate the composite spatial factor for a larva and its adult sibling.

Parameters
----------
M : square row-stochastic adult daily movement matrix
x1 : known laying site of reference larva 1
x2 : sampled location of adult sibling 2
y1,y2 : egg-laying days of siblings 1 and 2
t2 : sampling day of adult sibling 2
te,tl,tp : subadult stage durations

Returns
-------
float, conditional spatial probability of adult 2 at x2, given larva 1 at x1 and the two laying dates.

Notes
-----
Valid supplied histories have t2 >= y2+te+tl+tp. When y2 >= y1, maternal movement after laying egg 1 and movement of adult offspring 2 combine as the inclusive adult transition from x1 on day y1+te+tl+tp to x2 on day t2. When y2 < y1, condition the earlier maternal node on the known later node x1 using the normalized destination column of the inclusive transition from y2 to y1. Average the offspring transition from each earlier node on day y2+te+tl+tp to sampled node x2 at t2 using those conditional origin probabilities. This is the two-order movement law in S1 Text Eqs. 26-27."""
    return None
```

### Step 8

08_sibling_larva_adult.py

Goal
----
Compute larva-adult full-sibling probability across sites and days.

```python
def sibling_larva_adult(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    """Compute larva-adult full-sibling probability across sites and days.

Parameters
----------
M : adult daily movement matrix
x1,t1 : reference larva site and sampling day
x2,t2 : target adult site and sampling day
nf,beta : equilibrium females per site and female daily fecundity
te,tl,tp,ta : fixed stage durations and maximum adult age
mu_a,mu_e,mu_l,mu_p : daily stage mortalities

Returns
-------
float, probability that a target adult is a full sibling of the reference larva.

Notes
-----
Marginalize reference egg day y1 in the inclusive range t1-te-(tl-1) through t1-te, with p_L(a)=(1-mu_l)**a/sum_{r=0}^{tl-1}(1-mu_l)**r and a=t1-y1-te. For each y1 consider y2 between y1-(ta-1) and y1+(ta-1), intersected with the adult sampling support t2-te-tl-tp-(ta-1) through t2-te-tl-tp. Multiply each supported term by one half, adult maternal survival (1-mu_a)**abs(y2-y1), larva_adult_movement for these days and sites, and beta*(1-mu_e)**te*(1-mu_l)**tl*(1-mu_p)**tp*(1-mu_a)**(t2-y2-te-tl-tp). Divide the total by adult_denominator. Both offspring orders and adult age zero are included; return zero if no histories fit."""
    return None
```

### Step 9

09_joint_kin_score.py

Goal
----
Combine mother-larva and larva-adult full-sibling count scores.

```python
def joint_kin_score(M: "np.ndarray", mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    """Combine mother-larva and larva-adult full-sibling count scores.

Parameters
----------
M : daily adult movement matrix
mo_rows : records (x1,t1,x2,t2,n_f_sample,n_l_sample,k_mother)
la_rows : records (x1,t1,x2,t2,n_a_sample,k_sibling)
nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p : demographic parameters

Returns
-------
float, combined log pseudo-likelihood, excluding binomial coefficients.

Notes
-----
For each mother record evaluate individual mother_larva probability P, then use the group probability 1-(1-P)**n_f_sample in k*log(p)+(n_l_sample-k)*log(1-p). For each mixed-stage sibling record use sibling_larva_adult directly in k*log(P)+(n_a_sample-k)*log(1-P). Every listed record is used once, and the larval reference cannot coincide with a target adult, even at matching site and day. Define 0*log(0)=0 by continuity; other impossible observed counts produce negative infinity. The count coefficients independent of dispersal parameters are omitted."""
    return None
```

### Step 10

10_fit_barrier.py

Goal
----
Estimate barrier crossing while accounting for an unknown staying rate.

```python
def fit_barrier(coords: "np.ndarray", lam: float, barrier_x: float, mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float, bounds: tuple, p0_bounds: tuple) -> float:
    """Estimate barrier crossing while accounting for an unknown staying rate.

Parameters
----------
coords : N by 2 site coordinates
lam,barrier_x : conditional dispersal scale and vertical barrier position
mo_rows,la_rows : observed mother and mixed-stage sibling strata
nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p : fixed demographic parameters
bounds,p0_bounds : closed intervals for barrier crossing delta and staying p0

Returns
-------
float, globally maximizing barrier crossing factor delta.

Notes
-----
For each candidate (delta,p0) in the rectangular closed bounds, construct movement_kernel and evaluate joint_kin_score. Maximize that score jointly over both parameters, including endpoints, and return delta alone. A correct global solution within 1e-6 absolute error is accepted for an interior unique optimum; the choice of numerical optimizer is open. This joint fit follows the paper's pseudo-likelihood inference framework."""
    return None
```
