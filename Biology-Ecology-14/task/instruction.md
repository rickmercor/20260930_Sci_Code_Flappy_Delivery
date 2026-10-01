# Biology-Ecology-14

## Background

Niche breadth and niche overlap summarize how species divide foods, habitats or other resource states. Information-theoretic versions give different importance to resource states according to how strongly they distinguish species. The modified weighting method used here keeps every sampled state in the calculation, including an empty state, and removes a focal species or pair before estimating the weights so their own observations do not determine the scale on which they are judged.

Competitive coexistence is a separate question. Positive equilibrium densities, global stability and protection above a management reserve are not interchangeable. A plan can pass the stability boundary yet fail finite-community feasibility, and a feasible stable plan can still be fragile when the same correlated environmental disturbance acts across all conditions. The final score therefore joins niche structure to the interaction model and then asks for the weakest reserve margin across the full community.

## Problem

A field team must choose one refuge plan for a six-species community exposed to three environmental conditions. The choice uses two recent ecological methods in sequence. First, infer noncircular niche breadths and overlaps from the resource-use table using a recent modified information-theoretic weighting method. Then use those niche values to condition the competition and self-regulation data before applying a recent finite-community coexistence and stability treatment. Finally, measure reserve protection against a shared correlated climate disturbance, repeat the entire calculation under six resource-state stress scenarios, and choose a plan that survives the stress audit rather than merely winning the nominal calculation.

All inputs below are synthetic. Species order is Sp1 through Sp6, resource-state order is R1 through R7, condition order is E1 through E3, and plan order is P01 through P06. Use natural logarithms and the convention 0 ln 0 = 0.

## Resource observations

R7 was sampled but no individual used it.

| Species | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sp1 | 7 | 39 | 11 | 9 | 0 | 6 | 0 |
| Sp2 | 39 | 0 | 25 | 0 | 31 | 4 | 0 |
| Sp3 | 0 | 0 | 0 | 21 | 37 | 10 | 0 |
| Sp4 | 0 | 0 | 0 | 0 | 14 | 0 | 0 |
| Sp5 | 0 | 0 | 0 | 37 | 0 | 0 | 0 |
| Sp6 | 0 | 38 | 0 | 0 | 0 | 0 | 0 |

Use the source's exponential modified factor ed_j in Eq. (33), not either of its other candidate factors, together with its adjusted probabilities, standardized Shannon-type breadth, and information-theoretic overlap. Treat delta_j as the contribution of state j: its numerator sums over species i only; the stray outer sum over j in the displayed Eq. (16) would erase the state index and is not used. Set the matrix-expansion constant to k = 10000. Apply the source's noncircular rule throughout: the factor for a species breadth comes from the matrix without that species, and the factor for a pair overlap comes from the matrix without both species. The occupied-state count in every modified factor is the count from the complete table, not the reduced table.

Pack the niche geometry in a symmetric matrix G: G_ii is the noncircular breadth of species i and G_ij is the noncircular overlap of species i and j.

## Measured competition

Rows are affected species and columns are competitors. Diagonal entries are zero.

### Condition E1

|  | Sp1 | Sp2 | Sp3 | Sp4 | Sp5 | Sp6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sp1 | 0 | 0.515 | 0.858 | 1.099 | 0.880 | 0.244 |
| Sp2 | 0.218 | 0 | 0.792 | 0.794 | 0.891 | 0.134 |
| Sp3 | 1.020 | 0.121 | 0 | 0.869 | 0.803 | 0.199 |
| Sp4 | 0.396 | 0.345 | 0.452 | 0 | 0.859 | 0.543 |
| Sp5 | 0.876 | 0.526 | 0.397 | 0.614 | 0 | 0.387 |
| Sp6 | 0.398 | 0.691 | 0.511 | 0.179 | 0.331 | 0 |

### Condition E2

|  | Sp1 | Sp2 | Sp3 | Sp4 | Sp5 | Sp6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sp1 | 0 | 0.685 | 1.068 | 1.013 | 0.914 | 0.300 |
| Sp2 | 0.205 | 0 | 0.669 | 0.870 | 0.757 | 0.150 |
| Sp3 | 1.302 | 0.093 | 0 | 0.783 | 0.697 | 0.250 |
| Sp4 | 0.292 | 0.416 | 0.471 | 0 | 0.952 | 0.720 |
| Sp5 | 0.828 | 0.540 | 0.443 | 0.606 | 0 | 0.406 |
| Sp6 | 0.372 | 0.636 | 0.487 | 0.133 | 0.421 | 0 |

### Condition E3

|  | Sp1 | Sp2 | Sp3 | Sp4 | Sp5 | Sp6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sp1 | 0 | 0.628 | 1.112 | 1.418 | 1.106 | 0.223 |
| Sp2 | 0.226 | 0 | 1.133 | 0.992 | 1.014 | 0.177 |
| Sp3 | 1.352 | 0.158 | 0 | 1.025 | 0.930 | 0.232 |
| Sp4 | 0.337 | 0.409 | 0.347 | 0 | 1.136 | 0.366 |
| Sp5 | 1.115 | 0.593 | 0.438 | 0.666 | 0 | 0.469 |
| Sp6 | 0.465 | 0.774 | 0.403 | 0.219 | 0.304 | 0 |

Let beta_i = G_ii and let gamma_bar be the mean of all ordered off-diagonal G_ij values. The effective competition coefficient is

B_sij = B0_sij exp[u_s(G_ij - gamma_bar) + v_s((beta_j - beta_bar) - (beta_i - beta_bar))]

for i not equal to j, with a zero diagonal. The condition coefficients are:

| Condition | u_s | v_s |
| --- | ---: | ---: |
| E1 | 0.80 | -0.50 |
| E2 | -0.35 | 0.70 |
| E3 | 1.10 | 0.45 |

## Refuge plans

The table gives measured self-regulation slopes before the niche correction.

| Plan | Sp1 | Sp2 | Sp3 | Sp4 | Sp5 | Sp6 | q_c | Maximum effort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01 | 0.6140 | 1.1170 | 0.8170 | 1.1640 | 1.2540 | 0.9730 | -1.10 | 8.0 |
| P02 | 1.6049 | 0.9328 | 1.1209 | 0.9394 | 1.4058 | 0.7073 | 0.80 | 7.6 |
| P03 | 0.5950 | 1.4360 | 0.8550 | 0.8270 | 0.9000 | 0.7810 | -0.60 | 8.4 |
| P04 | 0.9220 | 1.0440 | 0.8500 | 1.1740 | 1.4720 | 0.9220 | 1.25 | 7.9 |
| P05 | 1.2220 | 1.2840 | 0.9280 | 0.6300 | 1.2550 | 0.5530 | -0.90 | 8.2 |
| P06 | 1.0440 | 1.4040 | 0.9660 | 0.5790 | 0.6650 | 0.7160 | 0.55 | 7.7 |

For plan c, use the effective slope d_ci = d0_ci exp[q_c(beta_i - beta_bar)]. At effort a in condition s, define A_sc(a) = a diag(d_c) + B_s. The undisturbed equilibrium is A_sc(a)^(-1) 1.

## Climate disturbance and reserve rule

The two climate-factor loadings are:

| Condition | Species | Factor 1 | Factor 2 |
| --- | --- | ---: | ---: |
| E1 | Sp1 | 0.306 | 0.292 |
| E1 | Sp2 | -0.313 | 0.230 |
| E1 | Sp3 | -0.395 | 0.320 |
| E1 | Sp4 | 0.382 | 0.177 |
| E1 | Sp5 | 0.361 | 0.427 |
| E1 | Sp6 | -0.416 | -0.493 |
| E2 | Sp1 | -0.198 | 0.303 |
| E2 | Sp2 | -0.129 | 0.568 |
| E2 | Sp3 | -0.205 | -0.096 |
| E2 | Sp4 | 0.252 | 0.326 |
| E2 | Sp5 | -0.297 | 0.106 |
| E2 | Sp6 | -0.056 | -0.095 |
| E3 | Sp1 | -0.008 | 0.031 |
| E3 | Sp2 | -0.332 | 0.017 |
| E3 | Sp3 | 0.395 | 0.785 |
| E3 | Sp4 | 0.069 | 0.690 |
| E3 | Sp5 | 0.083 | -0.445 |
| E3 | Sp6 | -0.045 | -0.223 |

The same disturbance vector z acts in every condition, and x_sc(a,z) = A_sc(a)^(-1)(1 + U_s z). Its uncertainty set is z^T Sigma^(-1) z <= rho^2, with

|  | Factor 1 | Factor 2 |
| --- | ---: | ---: |
| Factor 1 | 1.0000000000 | -0.3577708764 |
| Factor 2 | -0.3577708764 | 0.8000000000 |

Reserve densities in species order are [0.032, 0.045, 0.036, 0.027, 0.041, 0.034]. For fixed plan and effort, rho is the largest radius for which every species remains at or above its reserve in every condition. Use the exact ellipsoidal distance to each reserve boundary, not separate marginal climate ranges.

For every plan and condition, move to the diagonal regulation coordinates in which the self-regulation term becomes a times the identity and the equilibrium equation keeps a right-hand side of ones. In those coordinates, use the second source's canonical column-centering construction and calculate the finite positive-feasibility threshold F. Also calculate G in the same coordinates, the boundary obtained from the smallest eigenvalue of the symmetric part of the transformed interaction matrix. The allowed effort interval for plan c is

[max(0.2, 1.8 max_s(F_sc, G_sc)), maximum effort_c].

A plan is unavailable in a calculation if its interval is empty or if its best reserve radius is negative. Otherwise maximize rho_c(a)/a over the complete closed interval. Interior changes in the limiting species or condition must be included. Values within 1e-10 tie in favour of the lower plan label.

The nominal calculation is only the first screen. For each of the six occupied resource states R1 through R6, make one stress table by multiplying that complete resource column by 0.35 and leaving every other entry unchanged. Recompute the whole chain from the modified factors onward; do not reuse nominal breadths, overlaps, competition, regulation, certificates, optima or limiting constraints. Within each stress table, a plan receives one win when its optimized score is within 1e-10 of that table's largest finite score. Its regret is the largest finite score in that table minus its own score. The tail regret is the mean of its two largest stress regrets. A plan is eligible only if it wins at least four of the six stress tables. Its decision score is its nominal optimized score minus 0.40 times its tail regret. Select the eligible plan with the largest finite decision score, with values within 1e-10 tied in favour of the lower plan label. Return zero if no plan is eligible.

Report the selected plan's decision score.

In the reasoning, identify both papers and the governing conventions. Give the complete-matrix standardized resource heterogeneity and R7 modified factor; the six niche breadths and the Sp1-Sp2 overlap; the three effective-competition audit entries E1/Sp1<-Sp2, E2/Sp3<-Sp4 and E3/Sp6<-Sp1; the effective slope vector for P04; P04's three F values, three G values and lower effort; all six nominal optimized plan scores; P04's nominal optimum effort, reserve radius and limiting condition/species; the six stress winners in R1 through R6 order; the stress-win counts, tail regrets and decision scores for all six plans; the eligible set and selected plan; and the winner and score produced by each of these four wrong shortcuts: using the nominal winner without the stress audit, attenuating all six occupied columns together instead of one at a time, ranking the plans by their number of stress wins without applying the regret penalty, and allowing plans with fewer than four stress wins into the decision-score comparison. Report every requested numerical value, including the final answer, to at least seven significant figures.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Include the requested deterministic intermediate values.
Do not include code, full input tables, or full matrices in the response.

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

01_use_probabilities.py

Goal
----
Return the conditional resource-use probabilities of the source's Eq. (6) for a resource matrix: resource_matrix[i, j] is the abundance (or resource-use value) of species i in resource state j, and the result holds, for every species, the probability that one of its individuals is associated with each resource state. A species whose row is entirely zero receives a row of zeros.

```python
def use_probabilities(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Return the conditional resource-use probabilities of the source's Eq. (6) for a resource matrix: resource_matrix[i, j] is the abundance (or resource-use value) of species i in resource state j, and the result holds, for every species, the probability that one of its individuals is associated with each resource state. A species whose row is entirely zero receives a row of zeros.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances: s species (rows) by r resource states (columns).

    Returns
    -------
    use_probabilities : numpy.ndarray
        Array of shape (s, r); each row sums to 1 (or is all zeros for an absent species) (float64).

    Raises
    ------
    ValueError
        If resource_matrix is not two-dimensional, contains a negative or non-finite entry, or has no positive entry.
    """
    return use_probabilities
```

### Step 2

02_resource_entropies.py

Goal
----
Return the four information quantities of the source's Eqs. (10)-(13) for a resource matrix, as the array [H(X), H(Y), H(XY), H_Y(X)]: the entropy of the resource states, the entropy of the species, their joint entropy, and the conditional entropy of the resource states given the species, all in nats, computed from the pooled-community probabilities of the source's Eqs. (5), (7) and (9).

```python
def resource_entropies(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Return the four information quantities of the source's Eqs. (10)-(13) for a resource matrix, as the array [H(X), H(Y), H(XY), H_Y(X)]: the entropy of the resource states, the entropy of the species, their joint entropy, and the conditional entropy of the resource states given the species, all in nats, computed from the pooled-community probabilities of the source's Eqs. (5), (7) and (9).

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances: s species by r resource states.

    Returns
    -------
    entropies : numpy.ndarray
        Array of shape (4,): [H(X), H(Y), H(XY), H_Y(X)] in nats (float64).

    Raises
    ------
    ValueError
        If resource_matrix is not two-dimensional, contains a negative or non-finite entry, or has no positive entry.
    """
    return entropies
```

### Step 3

03_state_contributions.py

Goal
----
Return the contribution of each resource state to the source's standardized resource heterogeneity M(X) (its Eq. (16), the quantity the source denotes delta_j), given the resource matrix, the use probabilities of step 01 and the entropies of step 02. The contributions sum to M(X) of the source's Eq. (15); a state used by no species, or used in identical proportions by every species, contributes zero.

```python
def state_contributions(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                        entropies: "numpy.ndarray") -> "numpy.ndarray":
    """Return the contribution of each resource state to the source's standardized resource heterogeneity M(X) (its Eq. (16), the quantity the source denotes delta_j), given the resource matrix, the use probabilities of step 01 and the entropies of step 02. The contributions sum to M(X) of the source's Eq. (15); a state used by no species, or used in identical proportions by every species, contributes zero.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances.
    use_probs : numpy.ndarray
        Array of shape (s, r) from step 01.
    entropies : numpy.ndarray
        Array of shape (4,) from step 02: [H(X), H(Y), H(XY), H_Y(X)].

    Returns
    -------
    state_contributions : numpy.ndarray
        Array of shape (r,): the contribution delta_j of every resource state (float64).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an input is negative or non-finite, or H(X) is not positive (fewer than two occupied resource states).
    """
    return state_contributions
```

### Step 4

04_modified_weights.py

Goal
----
Return the source's modified weighting factor of each resource state (its Eq. (33), denoted ed_j) from the state contributions of step 03 and n_occupied, the number of resource states that are used by at least one species in the complete resource matrix the analysis refers to (the source's r'). The factors are strictly positive for every state, including states used by no species, and sum to 1.

```python
def modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """Return the modified resource-state weights.

    Parameters
    ----------
    contributions : numpy.ndarray
        One-dimensional array from step 03.
    n_occupied : int
        Number of occupied resource states.

    Returns
    -------
    weights : numpy.ndarray
        Strictly positive float64 weights summing to 1.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return weights
```

### Step 5

05_noncircular_niche_geometry.py

Goal
----
Build the complete noncircular niche geometry for a community

```python
def noncircular_niche_geometry(resource_matrix: "numpy.ndarray", k: float) -> "numpy.ndarray":
    """Return the square noncircular niche-geometry matrix.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Species-by-resource matrix of finite nonnegative counts, with at
        least three species and two resource states.
    k : float
        Matrix-expansion constant, strictly greater than one.

    Returns
    -------
    geometry : numpy.ndarray
        Square float64 array. geometry[i,i] is species i's noncircular
        standardized breadth and geometry[i,h] is the noncircular modified-
        factor Horn overlap of species i and h. With ed_j the modified
        factors of the reduced matrix (without species i for a breadth,
        without species i and h for an overlap; step 04, using the complete
        matrix's occupied-state count), the adjusted use of a focal species
        is p*_ij = N_ij / sum_l(ed_l * k * N_il). The breadth is
        -(k / ln k) * sum_j ed_j * p*_ij * ln(p*_ij), and the overlap is
        -(1 / (2 ln 2)) * sum_j ed_j * k * [I(p*_ij) + I(p*_hj)
        - I(p*_ij + p*_hj)], with I(x) = x ln x and 0 ln 0 = 0.

    Raises
    ------
    ValueError
        If resource_matrix is not a finite nonnegative two-dimensional
        matrix with at least three nonempty species and two resource states,
        fewer than two resource states are occupied, or k is not finite and
        strictly greater than one.
    """
    return geometry
```

### Step 6

06_effective_competition.py

Goal
----
Convert measured competition into the niche-conditioned condition panel.

```python
def effective_competition(base_competition: "numpy.ndarray", geometry: "numpy.ndarray",
                          coupling: "numpy.ndarray") -> "numpy.ndarray":
    """Apply the declared directed ecological bridge.

    For each condition s and off-diagonal pair i,j, multiply the measured
    coefficient by exp(coupling[s,0]*(overlap_ij-mean_overlap) +
    coupling[s,1]*(breadth_j-breadth_i)). The overlap mean is over all
    ordered off-diagonal entries. Diagonal entries remain zero.

    Raises
    ------
    ValueError
        If the arrays are nonfinite or dimensionally misaligned, the
        competition panel is not square and nonnegative with a zero
        diagonal, the geometry matrix is not symmetric and aligned, or
        coupling does not contain one two-value row per condition.
    """
    return competition
```

### Step 7

07_effective_regulation.py

Goal
----
Convert plan-level regulation measurements into niche-conditioned slopes.

```python
def effective_regulation(base_designs: "numpy.ndarray", geometry: "numpy.ndarray",
                         breadth_response: "numpy.ndarray") -> "numpy.ndarray":
    """Multiply plan c, species i by exp(q_c*(beta_i-mean(beta))).

    Raises
    ------
    ValueError
        If the arrays are nonfinite or dimensionally misaligned, any
        regulation slope in base_designs is not strictly positive, the
        geometry matrix is not symmetric and aligned, or breadth_response
        does not contain one value per plan.
    """
    return regulation
```

### Step 8

08_coexistence_certificates.py

Goal
----
Compute the finite-community feasibility and global-stability boundaries.

```python
def coexistence_certificates(competition: "numpy.ndarray", regulation: "numpy.ndarray") -> "numpy.ndarray":
    """Return one certificate row per condition for one management plan.

    Work in the diagonal regulation coordinates that turn the self-regulation
    term into effort times the identity while keeping the right-hand side
    equal to one. For each condition, F is the last nonnegative effort at
    which the finite-community equilibrium branch reached from large effort
    is not strictly positive, including losses through every boundary face.
    G is the separate sufficient global-stability boundary from the smallest
    eigenvalue of the symmetric part in the same coordinates. Return
    [F, G, alpha_0, alpha_1, ..., alpha_N], where alpha_0 is the interior
    branch obstruction and alpha_i is the governing loss through face i;
    record a nongoverning or negative candidate as zero. A root or eigenvalue
    counts as real when |Im| <= 1e-8 max(1, |Re|), and candidates within
    1e-10 of the interior obstruction are nongoverning.

    Raises
    ------
    ValueError
        If competition is not a finite nonnegative panel of square matrices
        with zero diagonals, regulation is not an aligned finite vector, or
        any regulation value is not strictly positive.
    """
    return certificates
```

### Step 9

09_select_niche_refuge.py

Goal
----
Select the refuge plan that maximizes robust reserve protection per unit effort. Returns
-------
return score

```python
def select_niche_refuge(resource_matrix: "numpy.ndarray", k: float,
                        base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                        loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                        covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                        coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                        buffer: float, minimum_effort: float) -> float:
    """Return the largest eligible robust climate radius per unit effort.

    Construct the noncircular niche geometry, condition the competition and
    regulation panels, enforce both coexistence certificates with the buffer,
    and optimize each plan on its full closed effort interval. A single
    correlated ellipsoidal climate disturbance acts in all factors. Plans
    with an empty interval or negative best reserve radius are unavailable.
    Values within 1e-10 tie in favor of the earliest plan; return zero when no
    plan is available.

    Raises
    ------
    ValueError
        If the ecological arrays are nonfinite or dimensionally misaligned,
        plan slopes or effort bounds are not strictly positive, reserves are
        negative, buffer is not greater than one, minimum_effort is not
        positive, covariance is not symmetric positive definite, or the
        required population-response system is singular or has zero climate
        response.
    """
    return score
```

### Step 10

10_return_score.py

Goal
----
Select the refuge plan that survives the resource-state stress audit.

```python
def select_stress_tested_refuge(resource_matrix: "numpy.ndarray", k: float,
                                base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                                loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                                covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                                coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                                buffer: float, minimum_effort: float,
                                attenuation: float, minimum_wins: int,
                                regret_weight: float) -> float:
    """Return the largest eligible stress-adjusted refuge score.

    First compute every plan's nominal optimized reserve-radius-per-effort
    score. Then form one stress scenario for each occupied resource state by
    multiplying that complete resource column by attenuation and recomputing
    the entire niche, competition, regulation, certificate and optimization
    chain. A plan wins a stress scenario when its score is within 1e-10 of
    that scenario's largest finite score. Its scenario regret is the largest
    finite score minus its own score. The tail regret is the mean of its two
    largest stress regrets. A plan is eligible only when it wins at least
    minimum_wins stress scenarios. Its decision score is its nominal score
    minus regret_weight times its tail regret. Choose the largest finite
    decision score, with values within 1e-10 tied in favour of the earliest
    plan, and return zero when no plan is eligible.

    Raises
    ------
    ValueError
        If attenuation is not strictly between zero and one, minimum_wins is
        not an integer from one through the number of occupied states,
        regret_weight is negative or nonfinite, or any earlier ecological
        contract is violated.
    """
    return result
```
