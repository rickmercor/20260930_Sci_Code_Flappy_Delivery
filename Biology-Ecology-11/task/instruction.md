# Biology-Ecology-11

## Background

Metapopulation and metacommunity theory describe species that live in discrete habitat patches linked by dispersal. Classical measures of persistence, such as the metapopulation capacity of a fragmented landscape, start from a dispersal kernel that is assumed rather than derived, usually a function that decays with the distance between patches, and from extinction and colonisation rates that are uniform across the landscape. Real landscapes break both assumptions. In river networks, dispersal follows the directed structure of the drainage, downstream movement is easier than upstream movement, and patches differ widely in the habitat they offer.

A growing line of work in statistical ecology derives effective metapopulation and metacommunity models from individual-based dynamics instead. Settled individuals that do not move are distinguished from a mobile dispersal stage, such as seeds, larvae or spores, that explores the landscape along a network before it settles. Coarse-graining over the mobile stage turns an arbitrary dispersal topology into an effective colonisation kernel that encodes both the network and the species' traits. This makes it possible to ask exact questions about persistence in heterogeneous landscapes, about how several species competing for the same space share or partition a landscape, and about how the finite size of local habitat produces demographic fluctuations and eventual extinction. These questions matter for conservation and for interpreting occupancy and extinction records from fragmented systems, where observed variability reflects the structure of the dispersal network as much as local demography.

## Problem

Two species compete for the colonisable sites of a heterogeneous river landscape. Your task is to derive the effective colonisation dynamics of both species from their individual-based reactions on the stated landscape, to determine how they share it in the long run, and to quantify the demographic fluctuations of their settled populations that the finite numbers of sites produce.

Patch $i$ holds $M_i$ colonisable sites, each empty or held by one settled individual $S_{\alpha i}$ of species $\alpha \in \{1, 2\}$, and mobile explorers $X_{\alpha i}$ of either species may be present in any number. With $w_{ij}$ the weight of the dispersal link from patch $i$ to patch $j$ and $f_\alpha = D_\alpha / \lambda_\alpha$, the individual-based reactions of species $\alpha$ are: $S_{\alpha i} \to \emptyset_i$ at rate $e_{\alpha i}$, where $\emptyset_i$ is an empty site of patch $i$; $S_{\alpha i} \to S_{\alpha i} + X_{\alpha j}$ at rate $c_{\alpha i} C_{\alpha, ij}$ with $C_{\alpha, ij} = \xi_\alpha w_{ij} / (1 + 1/f_\alpha)$; $X_{\alpha i} \to X_{\alpha j}$ at rate $D_\alpha w_{ij}$; loss of each explorer at rate $\gamma_\alpha$; and $X_{\alpha i} + \emptyset_i \to S_{\alpha i}$ and $X_{\alpha i} + S_{\beta i} \to S_{\beta i}$ for either species $\beta$, each at rate $\lambda_\alpha / M_i$ per explorer-site pair. The process is the continuous-time Markov chain on the numbers of settled individuals and of explorers of both species defined by these reactions and rates.

Use the following landscape, with rates per year.

- thirteen patches numbered 0 to 12; patch 0 is the outlet, and patches 1 to 12 drain into patches 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 6 and 10 respectively
- every patch has a link of weight 1.0 to the patch it drains into and a link of weight 0.3 from that patch back to it; in addition there are one-way overland links of weight 0.6 from patch 7 to patch 12 and of weight 0.5 from patch 11 to patch 4, and there are no other links
- $M_i = 40 n_i$, where $n_i$ counts the patches whose downstream path passes through patch $i$, patch $i$ included
- $e_{1 i}$ for patches 0 to 12: 0.30, 0.22, 0.26, 0.18, 0.35, 0.20, 0.24, 0.15, 0.40, 0.28, 0.17, 0.33, 0.21
- $c_{1 i}$ for patches 0 to 12: 0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.1, 1.3, 0.7, 0.9, 1.2, 0.8, 1.0
- $e_{2 i}$ for patches 0 to 12: 0.25, 0.30, 0.20, 0.28, 0.22, 0.30, 0.18, 0.32, 0.26, 0.20, 0.24, 0.19, 0.30
- $c_{2 i}$ for patches 0 to 12: 1.0, 0.9, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.0, 1.2, 0.8, 1.1, 0.9
- $\lambda_1 = 1.0$, $D_1 = 1.6$, $\gamma_1 = 0.25$, $\xi_1 = 0.8$; $\lambda_2 = 1.0$, $D_2 = 3.2$, $\gamma_2 = 0.5$, $\xi_2 = 0.8$

For a species alone in the landscape, its generalised metapopulation capacity is the dimensionless landscape measure that exceeds one exactly when the extinct state of the deterministic occupancy dynamics is unstable, and that is divided by $\theta$ when every local extinction rate of that species is multiplied by $\theta$. From every initial condition in which both species are present, the deterministic occupancy dynamics of the pair reaches the same stable steady state. Near that state the finite-site process persists for a long time, fluctuating about that steady state, before the eventual extinction.

Report, as the final answer, the standard deviation of the total number of settled individuals of species 2 in the landscape in that long-lived state, to leading order in the inverse site numbers when all site numbers grow in their stated proportions, to two decimal places. The answer is graded within $0.5$.

Report also, inside the reasoning: the generalised metapopulation capacity of species 1 alone, to four decimal places; the value of a common multiplier of every $c_{2 i}$ at which species 2, introduced in small numbers into the landscape held by species 1 at its own deterministic steady state, changes from declining to increasing, to three decimal places; the total numbers of sites held by species 1 and by species 2 at the stable steady state of the pair, to one decimal place; the standard deviation of the total number of settled individuals of species 1 in the long-lived state, to the same order as the final answer, to two decimal places; and the mean total number of settled individuals of species 2 in the long-lived state minus its value at the stable steady state of the pair, at the lowest order in the inverse site numbers at which this difference is non-zero, to two decimal places.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words): state the values requested above and the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

step_01_dendritic_landscape

Goal
----
This stage assembles the weight matrix and the site numbers from the drainage map, and records the two structural facts the later stages rely on: the out-strength q_i = sum_j w_ij of every patch, which is the total rate factor with which an explorer leaves it, and whether the directed network is strongly connected. Persistence theory for the effective dynamics needs an irreducible colonisation kernel, and the kernel inherits irreducibility from strong connectivity of the dispersal network.

A drainage map is valid only if it has exactly one outlet and every downstream path reaches that outlet without revisiting a patch; a map with a cycle does not describe a river.

```python
def build_dispersal_landscape(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
) -> dict:
    """Assemble the directed dispersal network and the patch site numbers of a river landscape.

    Parameters
    ----------
    downstream : sequence of int
        Downstream neighbour of each patch, -1 for the outlet.
    downstream_weight : float
        Weight of each downstream link, above zero.
    upstream_weight : float
        Weight of each upstream link, not below zero.
    extra_links : sequence of (int, int, float)
        Additional one-way links (i, j, w).
    sites_per_patch_drained : float
        Colonisable sites per patch of drainage count, above zero.

    Returns
    -------
    dict
        Under the keys weights, sites, drainage, out_strength, n_links and strongly_connected; strongly_connected is the integer 1 if every patch reaches every other along directed links and 0 otherwise.

    Raises
    ------
    ValueError
        When the drainage map has other than one outlet, points outside the patch set, points a patch at itself or contains a cycle, when a weight or the site density fails to be finite with the required sign, or when an extra link is malformed, repeats an ordered pair or duplicates a river link.
    """
    return
```

### Step 2

step_02_explorer_resolvent

Goal
----
When explorers are fast compared with the turnover of settled individuals, their densities sit at the quasi-stationary balance F x = b / lambda for the current production b, so what the settled population experiences is the resolvent F^(-1). The operator L does not depend on f or g, so it is diagonalised once, L = V Omega V^(-1), and the resolvent for any exploration efficiency and mortality ratio follows from the same eigenbasis,

F^(-1) = V diag(1 / (1 + g + f omega_l)) V^(-1),

which is how the effective kernel is written as an explicit function of the dispersal dynamics and the topology. For a directed network the eigenvalues and eigenvectors are in general complex, the inverse eigenvector matrix is not the transpose of the eigenvector matrix, and the resolvent is real only after the conjugate pairs are summed. An operator whose eigenvector matrix is numerically singular cannot be inverted this way.

```python
def explorer_resolvent(
    weights: np.ndarray,
    sites: np.ndarray,
    exploration_efficiency: float,
    mortality_ratio: float,
) -> dict:
    """Assemble the size-weighted explorer operator and invert the quasi-stationary balance in its eigenbasis.

    Parameters
    ----------
    weights : np.ndarray
        Link weights w_ij, non-negative with a zero diagonal.
    sites : np.ndarray
        Site numbers M_i, above zero.
    exploration_efficiency : float
        f = D / lambda, not below zero.
    mortality_ratio : float
        g = gamma / lambda, not below zero.

    Returns
    -------
    dict
        Under the keys operator, eigenvalues_real, eigenvalues_imag, resolvent, conservation_residual, inversion_residual and discarded_imaginary; the eigenvalues are sorted by ascending real part and then by ascending imaginary part.

    Raises
    ------
    ValueError
        When the weights fail to be a finite, non-negative square array with a zero diagonal, when the site numbers fail to be finite, above zero and of matching length, when f or g fails to be finite and not below zero, or when the operator is not diagonalisable to working precision.
    """
    return
```

### Step 3

step_03_effective_colonisation_kernel

Goal
----
In densities, the production of explorers in patch j is b_j = sum_i c_i C_ij [S_i] / M_j = sum_i c_i (M_i / M_j) C_ij p_i, with p_i = [S_i] / M_i the fraction of occupied sites. With explorers at their quasi-stationary balance F x = b / lambda, the colonisation attempt rate per site is lambda x = F^(-1) b, and colonisation of patch i proceeds at lambda x_i (1 - p_i). The settled dynamics close on the occupancies alone,

dp_i/dt = -e_i p_i + (1 - p_i) sum_j K_ij c_j p_j,

with the effective colonisation kernel

K_ij = sum_k (F^(-1))_ik (M_j / M_k) C_jk.

The kernel is independent of the occupancies because an attempt on an occupied site also consumes the explorer, so explorer loss by attempts is lambda x_i whatever the occupancy. It depends on the dispersal dynamics through f and g and on the topology through the eigenbasis of the explorer operator. The ratio of site numbers sits inside the sum, at the patch k where the explorers are released, not outside it: the resolvent already carries the size weighting of movement, and a factor M_j / M_i outside the sum counts it twice.

The same kernel is often wanted in counts. At vanishing occupancy one settled individual of patch j generates new settled individuals in patch i at rate K_ij c_j M_i / M_j, which does not involve the site numbers at all. Every explorer ends either in a colonisation attempt or in death, so summing the count kernel over targets gives the colonisation budget sum_i K_ij M_i / M_j = xi q_j / [(1 + 1 / f)(1 + g)] of a source patch.

```python
def effective_colonisation_kernel(
    weights: np.ndarray,
    sites: np.ndarray,
    max_explorability: float,
    exploration_rate: float,
    colonisation_rate: float,
    explorer_death_rate: float,
) -> dict:
    """Eliminate the quasi-stationary explorers and return the effective colonisation kernel of the settled occupancies.

    Parameters
    ----------
    weights : np.ndarray
        Link weights w_ij.
    sites : np.ndarray
        Site numbers M_i.
    max_explorability : float
        Maximal explorability xi, above zero.
    exploration_rate : float
        Explorer movement rate D, above zero.
    colonisation_rate : float
        Colonisation attempt rate lambda, above zero.
    explorer_death_rate : float
        Explorer death rate gamma, not below zero.

    Returns
    -------
    dict
        Under the keys kernel, count_kernel, release_feasibility, colonisation_budget, exploration_efficiency, mortality_ratio and budget_residual.

    Raises
    ------
    ValueError
        When the network or the site numbers are invalid, when xi, D or lambda fails to be finite and above zero, or when gamma fails to be finite and not below zero.
    """
    return
```

### Step 4

step_04_generalised_capacity

Goal
----
This stage computes lambda_max, the normalised positive right and left Perron vectors, the leading eigenvalue of the kernel alone for comparison with the classical capacity, and whether the kernel is irreducible. Irreducibility is decided on the pattern of positive entries, since a directed walk must join every ordered pair of patches.

```python
def generalised_capacity(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
) -> dict:
    """Compute the generalised metapopulation capacity of a colonisation kernel with patch-dependent local rates.

    Parameters
    ----------
    kernel : np.ndarray
        Effective colonisation kernel K_ij, non-negative.
    fecundity : np.ndarray
        Patch fecundities c_i, above zero.
    extinction : np.ndarray
        Patch extinction rates e_i, above zero.

    Returns
    -------
    dict
        Under the keys capacity, kernel_radius, perron_right, perron_left, perron_residual and irreducible; each Perron vector is positive and normalised to unit sum.

    Raises
    ------
    ValueError
        When the kernel fails to be a finite, non-negative square array of at least two patches, when the fecundities or extinction rates fail to be finite, above zero and of matching length, or when the kernel is not irreducible, since the Perron eigenvalue then fails to decide persistence.
    """
    return
```

### Step 5

step_05_equilibrium_occupancy

Goal
----
The stage runs the fixed-point iteration from full occupancy, polishes the result with Newton steps on the residual -e_i p_i + (1 - p_i) s_i, and reports the stability of the result through the largest real part of the eigenvalues of the Jacobian J = -diag(e + s) + diag(1 - p) K C of the occupancy dynamics.

```python
def equilibrium_occupancy(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Solve for the stable steady state of the effective occupancy dynamics and its landscape total.

    Parameters
    ----------
    kernel : np.ndarray
        Effective colonisation kernel K_ij.
    fecundity : np.ndarray
        Patch fecundities c_i.
    extinction : np.ndarray
        Patch extinction rates e_i.
    sites : np.ndarray
        Site numbers M_i.

    Returns
    -------
    dict
        Under the keys occupancy, occupied_sites, mean_occupancy, residual, stability_abscissa and persistent; persistent is the integer 1 if the steady state is non-trivial and 0 otherwise.

    Raises
    ------
    ValueError
        When the kernel, the local rates or the site numbers are invalid, when the kernel is not irreducible, or when the iteration fails to converge.
    """
    return
```

### Step 6

step_06_establishment_multiplier

Goal
----
Introduce the invader in small numbers into the landscape held by the resident at its own deterministic steady state p_1*. Linearised in the invader, its occupancies grow as dp_2/dt = [diag(1 - p_1*) K_2 C_2 - E_2] p_2, while the resident is left unchanged at first order. The invader increases exactly when the Perron eigenvalue of diag(1 - p_1*) K_2 C_2 E_2^(-1) exceeds one: the generalised capacity of the kernel diag(1 - p_1*) K_2, the invader's kernel thinned by the space the resident holds. Multiplying every fecundity c_2i by a common factor scales that eigenvalue by the same factor, so the multiplier at which the invader changes from declining to increasing is its reciprocal. A multiplier below one means the invader can already establish at its stated fecundities.

The stage computes the resident steady state, the resident-conditioned invasion capacity, the invader's capacity alone for comparison, and the establishment multiplier.

```python
def establishment_multiplier(
    resident_kernel: np.ndarray,
    invader_kernel: np.ndarray,
    resident_fecundity: np.ndarray,
    resident_extinction: np.ndarray,
    invader_fecundity: np.ndarray,
    invader_extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Find the common fecundity multiplier at which an invader starts to increase in a landscape held by a resident.

    Parameters
    ----------
    resident_kernel : np.ndarray
        Effective colonisation kernel of the resident.
    invader_kernel : np.ndarray
        Effective colonisation kernel of the invader.
    resident_fecundity : np.ndarray
        Patch fecundities of the resident.
    resident_extinction : np.ndarray
        Patch extinction rates of the resident.
    invader_fecundity : np.ndarray
        Patch fecundities of the invader.
    invader_extinction : np.ndarray
        Patch extinction rates of the invader.
    sites : np.ndarray
        Site numbers M_i.

    Returns
    -------
    dict
        Under the keys resident_occupancy, resident_sites, resident_capacity, invader_capacity, invasion_capacity and multiplier.

    Raises
    ------
    ValueError
        When a kernel, a set of local rates or the site numbers are invalid, when the kernels differ in size, or when the resident cannot persist on its own.
    """
    return
```

### Step 7

step_07_coexistence_state

Goal
----
Their stable steady state is found here by relaxation: from several interior initial conditions the dynamics are integrated until they settle, and the result is polished by Newton steps on the steady-state equations. Agreement of every start, positivity of every species in every patch, and a Jacobian whose eigenvalues all have negative real parts establish that the pair coexists at a unique stable state. Relaxation is slow when the species are close competitors, because the exchange of space between them decays at a small rate, and the stage reports that rate.

The steady state of these occupancy equations is also the steady state of the full rate equations of settled individuals and explorers, whatever the relative speed of explorers, because the explorer balance holds exactly at a steady state; only the approach to it and the fluctuations about it depend on the explorer speed.

```python
def coexistence_state(
    kernels: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
    n_starts: int,
) -> dict:
    """Find the unique stable steady state at which several competing species share the landscape.

    Parameters
    ----------
    kernels : np.ndarray
        Effective colonisation kernels, one per species.
    fecundity : np.ndarray
        Fecundities, one row per species.
    extinction : np.ndarray
        Extinction rates, one row per species.
    sites : np.ndarray
        Site numbers M_i.
    n_starts : int
        Number of interior initial conditions.

    Returns
    -------
    dict
        Under the keys occupancy, held_sites, total_held, start_spread, residual, stability_abscissa and slowest_rate.

    Raises
    ------
    ValueError
        When the inputs are invalid, when the relaxation fails, when any start leaves a species extinct in some patch, when the starts reach different states, or when the state reached is not stable.
    """
    return
```

### Step 8

step_08_chain_linearisation

Goal
----
The deterministic state supplies the settled numbers; the explorer numbers follow from the exact explorer balance, which is linear in the explorers,

[diag(D_a q + gamma_a + lambda_a) - D_a W^T] X_a = C_a^T (c_a * S_a),

because every attempt consumes an explorer whether or not it succeeds. The state vector is ordered as the settled numbers of species 1 to S, patch by patch, followed by the explorer numbers in the same order.

```python
def chain_linearisation(
    weights: np.ndarray,
    sites: np.ndarray,
    settled: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
) -> dict:
    """Build the drift Jacobian and the jump covariance of the settled-and-explorer chain at a deterministic state.

    Parameters
    ----------
    weights : np.ndarray
        Link weights w_ij.
    sites : np.ndarray
        Site numbers M_i.
    settled : np.ndarray
        Deterministic settled numbers, one row per species.
    fecundity : np.ndarray
        Fecundities, one row per species.
    extinction : np.ndarray
        Extinction rates, one row per species.
    max_explorability : np.ndarray
        Maximal explorability of each species.
    exploration_rate : np.ndarray
        Explorer movement rate of each species.
    colonisation_rate : np.ndarray
        Colonisation attempt rate of each species.
    explorer_death_rate : np.ndarray
        Explorer death rate of each species.

    Returns
    -------
    dict
        Under the keys explorers, jacobian, jump_covariance and drift_residual.

    Raises
    ------
    ValueError
        When the network, the site numbers, the settled numbers or any species rate is invalid, or when the arrays disagree in size.
    """
    return
```

### Step 9

step_09_stationary_covariance

Goal
----
The quantities of ecological interest are sums over patches. The total number of settled individuals of species a is the sum of its N settled components, so its variance is the sum of the corresponding N by N block of Sigma, and the covariance between the totals of two species is the sum of their cross block. For competitors sharing sites that covariance is typically strongly negative, because space one species gains is space the other loses; and when the two species exchange space slowly, that slow mode dominates the variance of each total.

Everything here depends on J and B only. Eliminating the explorers first and adding noise to the settled numbers alone is a different linear process whenever explorers are not much faster than settled turnover, and gives a different covariance.

```python
def stationary_covariance(
    jacobian: np.ndarray,
    jump_covariance: np.ndarray,
    n_species: int,
    n_patches: int,
) -> dict:
    """Solve for the stationary covariance of the linearised chain and summarise the fluctuations of the settled totals.

    Parameters
    ----------
    jacobian : np.ndarray
        Drift Jacobian of the chain at a stable steady state.
    jump_covariance : np.ndarray
        Jump covariance of the chain at that state.
    n_species : int
        Number of species.
    n_patches : int
        Number of patches.

    Returns
    -------
    dict
        Under the keys covariance, total_sd, total_correlation, patch_sd and relaxation_rate.

    Raises
    ------
    ValueError
        When the matrices fail to be finite and square of size 2SN, when the jump covariance fails to be symmetric and positive semi-definite, or when the Jacobian has an eigenvalue with real part not below zero.
    """
    return
```

### Step 10

step_10_mean_shift

Goal
----
Writing <n> = n* + delta about the steady state n*, and keeping the leading order, J delta + h = 0 with h = (1/2) H : Sigma. The only non-zero second derivatives are those of the settled drift of species a in patch i with respect to X_ai and any S_bi in the same patch, each equal to -lambda_a / M_i, so that

h_{S_ai} = -(lambda_a / M_i) sum_b Sigma_{X_ai, S_bi},

and h vanishes in every explorer component. The explorer drift is linear, because every colonisation attempt removes the explorer whether or not it succeeds. The mean shift of the total of species a is the sum of its settled components of delta.

The sign follows the covariance between explorers and settled individuals in each patch: where more explorers coincide with more occupied sites, colonisation is less efficient on average than at the mean state, and the settled means fall below the deterministic values.

```python
def mean_shift(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    colonisation_rate: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Compute the leading-order shift of the mean numbers of individuals away from the deterministic steady state.

    Parameters
    ----------
    jacobian : np.ndarray
        Drift Jacobian of the chain at the steady state.
    covariance : np.ndarray
        Stationary covariance of the chain.
    colonisation_rate : np.ndarray
        Colonisation attempt rate of each species.
    sites : np.ndarray
        Site numbers M_i.

    Returns
    -------
    dict
        Under the keys shift, total_shift and forcing.

    Raises
    ------
    ValueError
        When the matrices fail to be finite and of size 2SN consistent with the rates and sites, when the covariance is not symmetric, or when the Jacobian is singular.
    """
    return
```

### Step 11

step_11_two_species_fluctuation_report

Goal
----
The graded quantity is the standard deviation of the total number of settled individuals of species 2. The capacity of species 1, the establishment multiplier, the sites held by each species, the standard deviation for species 1 and the mean shift of species 2 are returned alongside it, because each is a step the graded number depends on.

```python
def two_species_fluctuation_report(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
    n_starts: int,
) -> dict:
    """Analyse how two competitors share a river landscape and the finite-site fluctuations of their settled totals.

    Parameters
    ----------
    downstream : sequence of int
        Downstream neighbour of each patch, -1 for the outlet.
    downstream_weight : float
        Weight of each downstream link.
    upstream_weight : float
        Weight of each upstream link.
    extra_links : sequence of (int, int, float)
        One-way overland links.
    sites_per_patch_drained : float
        Sites per patch of drainage count.
    fecundity : np.ndarray
        Fecundities, one row per species.
    extinction : np.ndarray
        Extinction rates, one row per species.
    max_explorability : np.ndarray
        Maximal explorability of each species.
    exploration_rate : np.ndarray
        Explorer movement rate of each species.
    colonisation_rate : np.ndarray
        Colonisation attempt rate of each species.
    explorer_death_rate : np.ndarray
        Explorer death rate of each species.
    n_starts : int
        Number of interior initial conditions for the coexistence state.

    Returns
    -------
    dict
        Under the keys species2_total_sd, species1_capacity, establishment_multiplier, held_sites, species1_total_sd, species2_mean_shift, species1_mean_shift, total_correlation, relaxation_rate, explorer_totals, invader_capacity, invasion_capacity, resident_sites, invader_alone_sites, operator_max_imag and sites.

    Raises
    ------
    ValueError
        When any stage rejects its inputs, when the network is not strongly connected, when the rates are not given for exactly two species, or when the two species do not share the landscape at a unique stable state.
    """
    return
```
