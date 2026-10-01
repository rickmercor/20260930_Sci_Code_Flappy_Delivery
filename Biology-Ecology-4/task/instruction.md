# Biology-Ecology-4

## Background

Large herbivores and their predators are managed on the strength of population counts, yet the outcome of removing or protecting a predator propagates through body-mass structure: predators take the smallest individuals, grazers of different size compete for the same grass while being protected to different degrees by their bulk, and what a predator can sustain itself on depends on the mass distribution of its prey rather than on their numbers. Whether such a community persists is therefore not a question that can be read off any single population's growth rate. A grazer that is the poorer competitor for grass may nevertheless persist because the predator, sustained by the better competitor, keeps that competitor in check, while the same predator, once it can live on the larger grazer alone, may exclude the smaller grazer altogether through apparent competition. Culling, disease or protection of the predator moves its annual survival, and there is no reason to expect the community to survive that movement in only one direction: with too little predator survival the predator is lost, with too much one grazer is lost, each by a different mechanism acting at a different corner of the community.

Community-level questions of this kind are answered by permanence theory rather than by simulation. The community is robustly permanent when no boundary state that the dynamics can settle into (no sub-community with some species missing) can hold the missing species out, in a sense made precise by invasion growth rates, subject to a global condition on how the sub-communities can be assembled from one another. For structured populations this analysis has to be carried out on the annual projection operators themselves, at boundary equilibria that the full system never visits because they are unstable to the very species they exclude, and the census of those equilibria is itself a function of the management parameter: a predator-containing sub-community exists only where the predator can live on that resource base. The bioenergetic ecosystem projection framework used here provides exactly the objects needed for such an analysis: annual kernels for every species that respond to the state of the whole community, and unstructured pools that close the nutrient loop.

## Problem

A recent line of work builds a general bioenergetic integral projection model for whole ecosystems: several body-mass-structured populations at different trophic levels are coupled through fluxes of biomass, consumption is modelled as sequences of encounters with individual consumers, the biomass removed from a population is partitioned among the consumers responsible, the acquired resource is assimilated with population-level saturation and allocated to maintenance, reproduction and growth, and unstructured decomposer, nutrient and grass pools close the recycling loop. Reproduce that projection framework on the bespoke three-species configuration below, certify robust permanence of the community by an invasion analysis of its boundary, and return the length of the window of predator survival over which the community is robustly permanent.

Determine from the source, and state briefly in your reasoning: (i) how survival of the encounters with a consumer species is constructed from the per-encounter kill probability and the consumer's number of individuals, and how the biomass removed from a population is partitioned among the consumer species and turned into each consumer's acquired resource; (ii) how the acquired resource becomes assimilated mass per individual (the saturating population-level form and its consequences) and how the assimilated mass is allocated among maintenance, reproductive investment and somatic growth, including how the offspring number is fixed; (iii) the order of survival, growth and reproduction within a year and which individuals reproduce; (iv) how grazing on the unstructured grass pool is constructed and partitioned among grazers, and where in the grass balance it acts; (v) the invasion growth rate of an absent species at a boundary equilibrium, in particular how its own density-dependent terms and its acquisition are treated in the limit of a vanishing population and which resident quantities set the consumer pressure it experiences; (vi) which sub-communities form the boundary census, what must hold at each of them, and what global condition on the pattern of invasions is required for the criterion to certify robust permanence.

Configuration. Two grazers share one unstructured grass pool and are eaten by one predator that also preys on its own species; the grass is fed by a nutrient pool replenished by decomposers acting on a fixed organic-matter pool. Species indices are 0 (small grazer), 1 (large grazer) and 2 (predator), in that order everywhere. With $h(x,y)=xy/(x+y)$ and the organic-matter pool held at the constant $D$, the unstructured pools advance as

$$B_{t+1}=B_t+(1-\phi_{DP})\,h(\phi_{DB}D,\rho_B B_t)-\phi_{BD}B_t,$$

$$P_{t+1}=P_t+\phi_{DP}\,h(\phi_{DB}D,\rho_B B_t)-h(\phi_{PG}P_t,\rho_G G_t),$$

$$G_{t+1}=G_t+\gamma_3\,h(\phi_{PG}P_t,\rho_G G_t)-\phi_{GD}G_t-G^{h}_t,$$

where $G^{h}_t$ is the grass removed by the grazers during year $t$, constructed as in the source from the grazers' per-encounter capture probabilities $\beta^{(j)}_{d,h,0}$ and their numbers of individuals. Each structured species $i$ is projected by the source's annual kernel, with intrinsic survival $\beta^{(i)}_{s,n,0}\,\mathrm{sigmoid}\big((z-\beta^{(i)}_{s,n,1})\beta^{(i)}_{s,n,2}\big)$, reproduction probability $\beta^{(i)}_{b,0}\,\mathrm{sigmoid}\big((z-\beta^{(i)}_{b,1})\beta^{(i)}_{b,2}\big)$, reproductive investment $\nu^{(i)}_o$, offspring mass of mean $\mu^{(i)}_o$ and standard deviation $\sigma^{(i)}_o$, growth standard deviation $\sigma^{(i)}_g$, metabolic coefficient $\delta^{(i)}$, assimilation and conversion efficiencies $\alpha^{(i)}_1,\alpha^{(i)}_2$, per-capita consumption capacity $\gamma^{(i)}_1$ and consumable fraction $\gamma^{(i)}_2$; the per-encounter probability that one predator kills an individual of species $i$ of mass $z$ is $\beta^{(i,2)}_{d,p,0}\,\mathrm{sigmoid}\big((z-\beta^{(i,2)}_{d,p,1})\beta^{(i,2)}_{d,p,2}\big)$. Grazers acquire all the grass they remove ($\lambda_2=1$), the predator acquires all the biomass it kills ($\lambda_4=1$), and no species scavenges or takes up nutrients ($\lambda_1=\lambda_3=0$). All fluxes of a year are evaluated from the state at the start of that year. Parameter block (units kg, individuals, years; sigmoid$(u)=1/(1+e^{-u})$):

D = 2.0e8  # kg, held fixed
phi_BD = 0.25  # yr^-1
phi_DB = 0.2  # dimensionless
rho_B = 54.75  # kg kg^-1 yr^-1
phi_DP = 0.175  # dimensionless
phi_GD = 0.033  # yr^-1
phi_PG = 0.0032  # dimensionless
rho_G = 1.3  # kg kg^-1 yr^-1
gamma_3 = 13.3  # kg kg^-1
species 0 (small grazer): beta_sn0 = 0.96 yr^-1, beta_sn1 = 7.9 kg, beta_sn2 = 0.025 kg^-1; alpha_1 = 0.60, alpha_2 = 0.075; gamma_1 = 2500 kg yr^-1, gamma_2 = 0.5; sigma_g = 5 kg; beta_b0 = 0.91 yr^-1, beta_b1 = 180 kg, beta_b2 = 0.08 kg^-1; nu_o = 0.05; mu_o = 15 kg, sigma_o = 3 kg; delta = 1.0 kg kg^-3/4 yr^-1; beta_dh0 = 0.0002 per encounter
species 1 (large grazer): beta_sn0 = 0.95 yr^-1, beta_sn1 = -200 kg, beta_sn2 = 0.005 kg^-1; alpha_1 = 0.62, alpha_2 = 0.085; gamma_1 = 4850 kg yr^-1, gamma_2 = 0.5; sigma_g = 10 kg; beta_b0 = 0.8 yr^-1, beta_b1 = 400 kg, beta_b2 = 0.075 kg^-1; nu_o = 0.045; mu_o = 22.5 kg, sigma_o = 3 kg; delta = 1.25 kg kg^-3/4 yr^-1; beta_dh0 = 0.0003 per encounter
species 2 (predator): beta_sn0 = theta (the control parameter, yr^-1), beta_sn1 = -40 kg, beta_sn2 = 0.025 kg^-1; alpha_1 = 0.75, alpha_2 = 0.08; gamma_1 = 800 kg yr^-1, gamma_2 = 0.68; sigma_g = 3 kg; beta_b0 = 0.58 yr^-1, beta_b1 = 40 kg, beta_b2 = 0.2 kg^-1; nu_o = 0.15; mu_o = 2 kg, sigma_o = 1 kg; delta = 1.5 kg kg^-3/4 yr^-1; beta_dh0 = 0 (does not graze)
kill sigmoid of the predator on species 0: beta_dp0 = 0.006 per encounter, beta_dp1 = 200 kg, beta_dp2 = -0.025 kg^-1
kill sigmoid of the predator on species 1: beta_dp0 = 0.004 per encounter, beta_dp1 = 350 kg, beta_dp2 = -0.05 kg^-1
kill sigmoid of the predator on species 2 (intraspecific): beta_dp0 = 0.0012 per encounter, beta_dp1 = 16 kg, beta_dp2 = -0.1 kg^-1
theta_ref = 0.78  # reference value of the control parameter
theta in [0.60, 0.95]  # scanned bracket

BENCHMARK CONVENTION (not a paper parameter). Float64 throughout; no randomness. Each structured species lives on a mesh of m = 50 equal cells on [0, z_max] with z_max = 500, 1000 and 100 kg for species 0, 1 and 2; densities are carried at the cell midpoints, every integral (including the number of individuals of a species) is the midpoint-rule sum with the cell width, and the recruitment and growth densities are evaluated at the midpoints with each column rescaled to unit midpoint-rule mass (evaluate them so that no column underflows to zero). A boundary equilibrium is a fixed point of the annual map restricted to a sub-community, located to a componentwise relative residual $\max_k |F(x)_k-x_k|/(1+|x_k|)\le 10^{-12}$ by any convergent iteration from any starting point, whether or not it attracts within its face; a sub-community enters the boundary census exactly when it carries such a fixed point with every member strictly positive (the empty community always does). Invasion growth rates are per year: the natural logarithm of the spectral radius of the absent species' projection matrix at the fixed point in the exact limit of a vanishing population of that species (not evaluated at a small finite number of individuals). The margin used to certify permanence is evaluated with the census recomputed at every theta, and thresholds are located to $10^{-12}$ absolute in theta within the stated bracket. Within the bracket the set of theta at which the community is robustly permanent is a single open interval. Report all quantities to at least six significant figures.

Do not decide the existence of a boundary equilibrium from the predator-free system or assume a fixed census across the bracket; do not evaluate the consumer pressure on an invader from anything but the residents' numbers at the fixed point; do not approximate a fixed point by simulating for a fixed horizon; do not replace the vanishing-population limit by a small population; do not test a sub-community with only its most plausible invader when several species are absent; do not replace the located thresholds by interpolation on a coarse grid; do not use tolerances looser than those stated; do not hard-code which sub-community binds at either end of the window.

In your reasoning, report these quantities from your run alongside the final answer, all at theta_ref unless stated: (1) the closed-form empty-community equilibrium (B, P, G) and the annual nutrient input to the P pool; (2) the boundary census at theta_ref, at theta = 0.60 and at theta = 0.95 (which sub-communities carry a fixed point), and the theta at which each predator-containing sub-community first appears; (3) for every census vertex at theta_ref, the number of individuals of each resident and the grass pool; (4) the invasion growth rate of every absent species at every census vertex at theta_ref; (5) the per-capita acquired mass and the assimilated mass per individual of the rare small grazer at the large-grazer-and-predator vertex, and of the rare predator at the two-grazer vertex, at theta_ref; (6) the permanence margin, and the vertex and invader attaining it, at theta_ref, at theta = 0.60 and at theta = 0.95; (7) the lower threshold, the vertex and invader whose rate vanishes there, and the census at that threshold including the residents' numbers of the predator-containing vertex that exists there; (8) the upper threshold, the vertex and invader whose rate vanishes there, the residents' numbers and grass pool of that vertex at the threshold, and at the same threshold the large grazer's invasion rate at the small-grazer-and-predator vertex and the predator's invasion rate at the two-grazer vertex; (9) the residual achieved at each fixed point, the resident rates at their own vertices, the transverse multiplier of each absent species, and the leading in-face multiplier of each vertex at theta_ref. Also state how the fixed points were located, how the thresholds were bracketed and refined, and how the global condition on the pattern of invasions was checked.

Return the length (in units of theta) of the interval of theta within [0.60, 0.95] over which the three-species community is robustly permanent, produced by this deterministic benchmark.

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

s01_grass_fraction_removed

Goal
----
Grass removal by a grazer population under the encounter model of the source, expressed as a fraction of the standing grass biomass. The step builds the population-level capture probability of each grazer species from its per-encounter capture probability and its number of individuals, combines the grazers into the total fraction of grass captured, applies the source's rule that only grass surviving intrinsic mortality can be grazed, and partitions the total removal among the grazer species. It also returns the per-individual removal of a vanishingly rare grazer species in the presence of the resident grazers (the rare-invader limit), which is the quantity that enters an invasion analysis. It validates that the inputs are admissible (non-negative counts, capture probabilities in [0, 1), a grass mortality proportion in [0, 1)) and raises ValueError otherwise. It deliberately excludes the conversion of removed grass into assimilated biomass and any dynamics of the grass pool.

```python
def grass_fraction_removed(counts, capture_probs, phi_GD, j, per_capita_limit=False):
    """Fraction of the standing grass biomass removed per year by grazer species j.

    Parameters
    ----------
    counts : sequence of float
        Number of individuals of every structured species, in species order (0 for absent species).
    capture_probs : sequence of float
        Per-encounter capture probability beta of every species (0 for species that do not graze).
    phi_GD : float
        Proportion of the grass that dies of intrinsic causes each year; grazing acts on the survivors.
    j : int
        Index of the grazer species whose removal is requested.
    per_capita_limit : bool, optional
        If True, return the removal per individual of species j in the limit of a vanishing
        population of species j (its own entry in `counts` is ignored); the residents are the other
        species with positive counts.

    Returns
    -------
    float
        Fraction of the standing grass biomass removed by species j per year (population level),
        or per individual of a vanishingly rare species j if `per_capita_limit` is True.
    """
    return 0.0
```

### Step 2

s02_prey_biomass_killed

Goal
----
Biomass of a structured prey species removed by the predator population under the source's encounter model of consumption. The step discretises the prey's body-mass distribution on the benchmark mesh, evaluates the per-encounter kill probability as a decreasing sigmoid of prey mass with the frozen parameters of the prey-predator pair, raises the survival of one encounter to the number of predators, weights the removal by the prey's intrinsic survival (only individuals that survive intrinsic causes of death are exposed to consumers) and integrates mass times removal probability over the distribution. In the vanishing-predator limit it returns the removal per predator. It validates the prey distribution specification and raises ValueError on inadmissible inputs. It excludes acquisition efficiencies and assimilation.

```python
def prey_biomass_killed(prey, prey_dist, n_consumers, per_capita_limit=False, overrides=None):
    """Biomass (kg per year) of a prey species killed by the predator population.

    Parameters
    ----------
    prey : int
        Index of the prey species (0 small grazer, 1 large grazer, 2 predator itself for intraspecific predation).
    prey_dist : dict
        Either {"count": n, "mean": mu, "sd": sigma} (a Gaussian body-mass distribution discretised on the
        prey's benchmark mesh and rescaled to n individuals) or {"values": [...]} (densities per kg at the
        mesh midpoints of the prey).
    n_consumers : float
        Number of predator individuals.
    per_capita_limit : bool, optional
        If True, return the removal per predator in the limit of a vanishing predator population
        (`n_consumers` is ignored).
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Kilograms of prey biomass removed per year by the predator population, or per predator in the limit.
    """
    return 0.0
```

### Step 3

s03_assimilated_mass

Goal
----
Net assimilated mass per individual per year from the total mass of resources acquired by a population, under the source's saturating consumption-and-assimilation model, including its vanishing-density form for a rare population. The step validates the inputs (non-negative masses and counts, positive rate constants) and raises ValueError otherwise. It excludes how the acquired mass is obtained and how the assimilated mass is allocated.

```python
def assimilated_mass(Zc, N, alpha1, alpha2, gamma1, gamma2):
    """Mass assimilated per individual per year (kg) from the population's acquired resources.

    Parameters
    ----------
    Zc : float
        Total resource mass acquired by the population per year (kg). If N == 0, the per-capita
        acquired mass of a vanishingly rare population (kg per individual per year).
    N : float
        Population size (number of individuals); 0 selects the vanishing-density limit.
    alpha1, alpha2 : float
        Assimilation and conversion efficiencies.
    gamma1 : float
        Per-capita consumption capacity (kg per individual per year).
    gamma2 : float
        Consumable fraction of the acquired mass.

    Returns
    -------
    float
        Assimilated mass per individual per year (kg).
    """
    return 0.0
```

### Step 4

s04_asymptotic_growth_rate

Goal
----
Asymptotic annual growth rate (natural logarithm of the dominant eigenvalue) of one structured species in a fixed environment, from the source's projection kernel assembled on the benchmark mesh: intrinsic survival, survival of encounters with a given number of predators, reproduction probability and offspring number, bioenergetic allocation of a given per-capita assimilated mass to maintenance, reproduction and growth, and the column-normalised recruitment and growth kernels. It validates the species index and raises ValueError on an unknown species. It excludes the feedback from the species onto its environment.

```python
def asymptotic_growth_rate(species, Za, consumer_count, overrides=None):
    """Natural logarithm of the dominant eigenvalue of a species' annual projection matrix.

    Parameters
    ----------
    species : int
        Species index (0 small grazer, 1 large grazer, 2 predator).
    Za : float
        Assimilated mass per individual per year (kg) in the environment considered.
    consumer_count : float
        Number of predators acting on this species with the frozen kill parameters of the pair
        (for species 2 this is the number of conspecific cannibals).
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Asymptotic annual growth rate (per year).
    """
    return 0.0
```

### Step 5

s05_annual_step_factor

Goal
----
One year of the full ecosystem map of the benchmark configuration: the decomposer, nutrient and grass pools advanced by the source's difference equations with the organic-matter pool held fixed, and every structured species projected through its kernel evaluated at the current state (grass removal partitioned among grazers, prey biomass removed by the predator and by intraspecific predation, saturating assimilation, allocation, column-normalised kernels). The step reports the one-year multiplication factor of a chosen quantity (a species' number of individuals, or one of the three dynamic pools). It validates the state specification and the selector and raises ValueError on inadmissible input (including a zero starting value of the selected quantity). It excludes equilibrium location.

```python
def annual_step_factor(theta, state, which, overrides=None):
    """One-year multiplication factor of a species count or of a dynamic pool under the ecosystem map.

    Parameters
    ----------
    theta : float
        Maximum annual intrinsic survival probability of the predator (the control parameter).
    state : dict
        {"B": kg, "P": kg, "G": kg, "species": [spec0, spec1, spec2]} in species order (small grazer,
        large grazer, predator), each spec being [count, mean, sd] (a Gaussian body-mass density discretised
        on the species' mesh and rescaled to the count; count 0 means absent) or {"values": [...]} (densities
        per kg at the mesh midpoints).
    which : int or str
        Species index 0, 1 or 2 (factor of the number of individuals) or "B", "P", "G" (factor of the pool).
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Value of the selected quantity after one year divided by its value before.
    """
    return 0.0
```

### Step 6

s06_boundary_equilibrium

Goal
----
Location of the fixed point of the ecosystem map restricted to a sub-community (a face of the boundary), to the benchmark residual, and report of one of its coordinates. The step solves the reduced fixed-point system in which each resident's distribution is the Perron vector of its own kernel (dominant eigenvalue exactly one), the nutrient and grass balances close, and the intraspecific kill flux of the predator is self-consistent, then verifies the componentwise relative residual of the full map; consumer faces are seeded from the equilibrium of their resource sub-face. It returns 0.0 for a face that carries no feasible fixed point (a species absent from the face, or a consumer that cannot persist there). It validates the face and the selector and raises ValueError on inadmissible input. It excludes invasion analysis.

```python
def boundary_equilibrium(theta, face, which, overrides=None):
    """A coordinate of the fixed point of the ecosystem map restricted to a sub-community.

    Parameters
    ----------
    theta : float
        Maximum annual intrinsic survival probability of the predator.
    face : sequence of int
        Indices of the species present in the sub-community (empty for the empty community).
    which : int or str
        Species index (number of individuals at the fixed point) or "B", "P", "G" (pool in kg).
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        The requested coordinate of the fixed point located to componentwise relative residual <= 1e-12,
        or 0.0 when the face carries no feasible fixed point.
    """
    return 0.0
```

### Step 7

s07_invasion_growth_rate

Goal
----
Invasion growth rate of a species at the fixed point of a sub-community: the natural logarithm of the spectral radius of the species' projection kernel evaluated at the located boundary equilibrium in the limit of a vanishing population of that species (per-capita acquisition in the rare limit, consumers evaluated at their resident numbers, and no feedback of the rare species on the residents). For a species that belongs to the sub-community the same construction returns its resident rate, which vanishes at the fixed point. The step locates the equilibrium internally, validates the face, the species and the existence of the fixed point, and raises ValueError otherwise. It excludes the census and the permanence criterion.

```python
def invasion_growth_rate(theta, face, species, overrides=None):
    """Invasion growth rate (per year) of a species at a sub-community's fixed point.

    Parameters
    ----------
    theta : float
        Maximum annual intrinsic survival probability of the predator.
    face : sequence of int
        Indices of the species present in the sub-community.
    species : int
        Index of the species whose rate is requested (an absent species gives its invasion rate in the
        vanishing-density limit; a resident gives its resident rate, which is zero at the fixed point).
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Natural logarithm of the spectral radius of the species' kernel at the fixed point (per year).
    """
    return 0.0
```

### Step 8

s08_permanence_margin

Goal
----
Permanence margin of the three-species community at a given predator survival: the boundary census (every proper sub-community that carries a feasible fixed point, consumer faces being admitted only when a fixed point exists with all members positive), the invasion growth rates of all absent species at every census vertex, the invasion graph on the census and its acyclicity, and the margin defined as the minimum over vertices of the largest invasion rate among the species absent from that vertex. The step raises ValueError if the invasion graph is cyclic (the criterion then does not apply). It excludes root finding in the control parameter.

```python
def permanence_margin(theta, overrides=None):
    """Permanence margin of the three-species community at predator survival theta.

    Parameters
    ----------
    theta : float
        Maximum annual intrinsic survival probability of the predator.
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Minimum over boundary-census vertices of the largest invasion growth rate among the species
        absent from the vertex (per year); positive exactly when the community is robustly permanent.
    """
    return 0.0
```

### Step 9

s09_permanence_window_length

Goal
----
Orchestrator: length of the interval of predator survival values within the scanned bracket over which the three-species community is robustly permanent. The step brackets the sign changes of the permanence margin on a coarse grid, locates each threshold by Brent's method to 1e-12, identifies at each threshold the census vertex and the invader whose rate vanishes there, rebuilds that vanishing rate independently from the elementary steps (grass removal or prey removal in the rare limit, saturating assimilation, kernel growth rate) using the vertex coordinates, checks that the vertex is a fixed point of the annual map, and returns the measure of the set of theta values with positive margin. Every earlier step is called and its output consumed on the answer path; inconsistencies raise ValueError. A local re-trace of the equilibrium state is used only to obtain the residents' mass distributions, which no earlier step exposes.

```python
def permanence_window_length(theta_min=0.60, theta_max=0.95, overrides=None):
    """Length of the set of predator-survival values in [theta_min, theta_max] at which the community is robustly permanent.

    Parameters
    ----------
    theta_min, theta_max : float
        Ends of the scanned bracket of the predator's maximum annual intrinsic survival probability.
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Lebesgue measure of {theta in [theta_min, theta_max] : permanence margin(theta) > 0}.
    """
    return 0.0
```
