# Chemistry-Computational_Chemistry-33

## Background

Kinetic Monte Carlo (KMC) is the workhorse for simulating surface catalysis at the level of individual adsorbates: the state of the surface is a lattice occupation pattern, elementary events (adsorption, desorption, diffusion, surface reactions) fire stochastically with rates taken from transition-state theory and electronic-structure calculations, and the simulation reproduces the statistics of the underlying chemical master equation. The number such a simulation is run to produce is almost always a rate: the turnover, that is the number of reaction events the surface delivers per unit time, collected over a sampling window and reported as an average. This makes KMC the natural bridge between first-principles energetics and measurable kinetics, at length and time scales far beyond molecular dynamics.

A persistent practical obstacle is stiffness. Adsorbate diffusion and other quasi-equilibrated channels can fire many orders of magnitude more often than the rate-limiting reactions, so that nearly all simulation effort is consumed by events that merely reshuffle configurations within a quasi-equilibrated set without advancing the slow chemistry. Two families of acceleration schemes have been developed in response. Rate-rescaling methods detect quasi-equilibrated reaction channels (for instance from the balance of forward and reverse occurrence counts) and scale their rate constants down to a user-chosen time-scale window, on the assumption that as long as the fast events remain equilibrated the slow kinetics are unaffected. Absorbing-Markov-chain methods instead treat the set of configurations connected by fast events as a transient block, compute the exit time and exit-state distribution from that block analytically, and let the simulation jump directly between such blocks.

Anisotropy and driving are both common in the surfaces where this matters. A channelled surface under an applied field, a temperature gradient or an epitaxial strain gradient carries a preferred direction along the grooves, so that a hop one way along a channel is not as likely as the hop back; the diffusion that dominates the event count is then not a relaxation towards equilibrium but a driven transport process, and what it does to the arrangement of the adsorbates is not something the interaction energy alone decides. A missing-row reconstructed fcc(110) face is grooved: adsorbates move easily along a channel and cross between channels only over a much higher barrier, so a single surface can carry a fast and a slow diffusion channel at once, and a nearest-neighbour pair means something different depending on which way the bond runs. Lateral interactions between neighbouring adsorbates are the other standard ingredient, entering the hop rates through the energies of the configurations a hop connects, and they matter because the reaction consumes pairs and the interaction decides how often pairs occur.

The research setting behind this task views the acceleration schemes through the master equation itself. Configurations connected by fast events are grouped into superbasins, a binary projector maps the full probability vector onto superbasin probabilities, and the generator is written as a fast part divided by a small timescale-separation parameter plus a slow part. A singular-perturbation expansion of the probability vector in that parameter is then used to obtain reduced models. What the successive orders of that expansion mean, how they are closed, how fast the truncated models converge, and what the leading order corresponds to among the existing acceleration schemes are the questions the source paper settles. The approach is illustrated there on small reaction networks and on single-species lattice models with adsorption, desorption, diffusion and a two-site reaction, the same ingredients used in this task. The number of lattice configurations doubles with every site, so on a lattice only a little larger than those the full generator is far beyond dense linear algebra, while the number of superbasins stays small; that is where a reduction of this kind earns its keep.

## Problem

Lattice kinetic Monte Carlo simulations of heterogeneous catalysis are frequently stiff: adsorbate diffusion fires far more often than adsorption, desorption and reaction, so most of the computational effort is spent on quasi-equilibrated hops that carry no kinetic information. A recently proposed treatment works directly with the master equation that underlies such simulations: the configurations connected by the fast events are lumped into superbasins, the master equation is projected onto the superbasin probabilities, and the probability vector is expanded in the timescale-separation parameter, which produces a hierarchy of closed reduced models of successive orders. Your task is to carry that reduction through third order for one driven, channelled surface model and to report, as a fraction of what the quasi-equilibrium picture predicts, how far the finite hop rate moves the reaction rate the surface would be measured to deliver.

The surface is channelled, as on a missing-row reconstructed fcc(110) face, and is modelled as a periodic 3 × 5 square lattice (fifteen sites, no edges): three channels, each a ring of five sites, carrying a single adsorbate species, so a configuration is one of 32 768 occupation patterns and the master equation dp/dt = W p is written with W[j, i] the rate from configuration i to configuration j and zero column sums. The fifteen bonds joining two sites of the same row run along a channel and the fifteen bonds joining two sites of the same column run across channels, over the missing row, and the two orientations are not interchangeable. Adsorbates on neighbouring sites repel whichever way the bond runs: writing P(i) for the number of bonds of either orientation whose two sites are both occupied in configuration i, configuration i carries the energy J·P(i) in units of the thermal energy. Adsorption fires on every empty site with rate k_ads and desorption on every occupied site with rate k_des, both environment independent. A doubly occupied bond may react, which empties both of its sites; a pair sitting across channels is the further apart of the two, so it reacts at its own smaller rate k_rxn_x while a pair sitting along a channel reacts at k_rxn.

Diffusion is what separates the timescales, and along the channels it is driven. An adsorbate on a site with an empty nearest neighbour may hop there, at a prefactor times the factor exp(−(J/2)(P(j) − P(i))) for the hop carrying configuration i to configuration j. Across channels the prefactor is κ_x in either direction. Along a channel the barrier is far lower, and a bias imposed along the channel direction makes one sense of motion easier than the other: a hop to the neighbour at the next column index carries the prefactor κ(1 + b) and a hop to the neighbour at the previous column index the prefactor κ(1 − b). Split the generator as W = F/ε + S, with F holding only the in-channel hops built from the normalised prefactor κ, and S holding adsorption, desorption, both reactions and the cross-channel hops. Superbasins are the configuration classes the fast events cannot mix, and both the full and the superbasin probabilities are expanded in powers of ε and closed order by order up to third.

What is measured on such a surface is not a configuration but a rate: the quantity of interest is R(i), the total rate at which a reaction event of either orientation fires in configuration i, averaged over the probability distribution and then over a sampling window that opens after the run has started. Everything must be exact, with no time discretisation anywhere: no step size may enter the answer. Writing π^(k)(t) = Σ_{l ≤ k} ε^l p^(l)(t) for the k-th order truncation of the full probability vector and ⟨R⟩^(k) for the window average of Σ_i R(i) π^(k)_i(t) that it predicts, report how far the third-order prediction lies from the quasi-equilibrium one, as a fraction of the quasi-equilibrium one.

Use exactly this configuration:
- lattice: 3 × 5, periodic in both directions, sites indexed s = 5 r + c, so that row r is one channel
- k_ads = 8.5 (per empty site), k_des = 4.0 (per occupied site)
- k_rxn = 10.0 (per doubly occupied bond along a channel), k_rxn_x = 4.0 (per doubly occupied bond across channels)
- lateral interaction: J = 1.5 (repulsive, in units of the thermal energy, counted over bonds of both orientations)
- κ = 1.0 per directed in-channel hop, drive bias b = 0.5, and ε = 0.02, so the physical in-channel prefactor is 50(1 ± b)
- κ_x = 0.4 per directed cross-channel hop, which stays in the slow generator
- initial state: the empty lattice at t = 0
- sampling window: t ∈ [0.02, 0.08]
- all rate constants in the same inverse time unit

In support of that number, report the few scalars that determine it: the quasi-equilibrium window-averaged rate itself; the three successive order contributions to it; the quasi-equilibrium mean reaction propensity of the class holding one adsorbate in each of two channels, of the class holding one adsorbate in every channel, and of the classes holding two and three adsorbates in a single channel; the number of superbasins and how many configurations each holds; the spread of the leading-order weights inside the class holding one adsorbate in every channel; how the correction divides between pairs lying along a channel and pairs lying across channels; the same fraction recomputed with the drive switched off and every other constant left alone; and the same fraction obtained from the untruncated master equation, propagated over the same window from the same start.

Your final answer must be a single number: Δ = (⟨R⟩^(3) − ⟨R⟩^(0)) / ⟨R⟩^(0), a dimensionless ratio, not a percentage.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
- Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

lattice_generators

Goal
----
The setup is a lattice model with N sites. Each configuration is a binary occupation vector: a site is 1 if occupied and 0 if empty. Configuration i is encoded as the integer whose bit s is 1 when site s is occupied, so there are m = 2^N configurations and configuration i and the integer i are the same object. The probability vector follows the master equation dp/dt = W p in the column convention: W[j, i] is the rate from configuration i to configuration j for j != i, and W[i, i] is the negative sum of the outgoing rates, so every column of W sums to zero.

Lattice structure

- The surface is channelled, as on a missing-row reconstructed fcc(110) face.

- Sites form an nrow x ncol lattice with periodic boundaries in both directions, holding at most 15 sites.

- Each site is indexed as s = r * ncol + c.

- A bond joining two sites of the same row runs along a channel.

- A bond joining two sites of the same column runs across channels, over the missing row.

- Periodicity needs care on small lattices: wrapping can send a site to itself, which is not a bond at all, and it can produce the same unordered pair twice, which is still one bond. A lattice one row deep therefore has no cross-channel bonds, and a lattice one column wide has no in-channel bonds.

Energy

- P(i) is the number of doubly occupied bonds of either orientation in configuration i.

- Configuration i carries the lateral-interaction energy J * P(i).

- Both orientations enter P(i), so every rate that feels the energy feels bonds of both kinds.

Three distinctions, one per role

1. Interaction: both orientations enter P(i).

2. Reaction: both orientations react and both empty their two sites at once, but a pair across channels is the further apart of the two and carries its own rate constant. A reaction fires at its bare rate constant; the interaction factor acts on hops only.

3. Diffusion: only motion along a channel is fast; crossing between channels is slow and stays in the slow generator.

Hopping rules

- Every hop carries the symmetric-barrier factor exp(-(J / 2) * (P(j) - P(i))) for the move i -> j: the barrier is raised by half of whatever energy the hop must pay and lowered by half of whatever it releases.

- Along a channel the hop is also driven. Writing the two in-channel neighbours of a site as its downstream neighbour, the one at the next column index, and its upstream neighbour, the one at the previous column index, the downstream hop carries the prefactor kappa * (1 + drive_bias) and the upstream hop kappa * (1 - drive_bias). The drive multiplies the interaction factor; it does not replace it.

- A channel one site long has no in-channel neighbour at all.

- A channel two sites long is the case in which the downstream and the upstream neighbour are the same site: the two directed moves are then the same move and their prefactors add, so such a channel carries 2 * kappa times the interaction factor and no drive survives, as it cannot on a ring with no sense of direction.

- Cross-channel hops feel the same interaction factor with their own prefactor kappa_cross, and the drive along the channels does not act across them, so both of their directions carry the same prefactor.

- The two-site ring is treated differently across channels than it is along one. On a lattice two rows deep the neighbour above and the neighbour below a site are the same site, and that cross-channel hop is carried once at kappa_cross, not twice: the cross-channel move is taken from the deduplicated bond set rather than from a pair of directed moves. Only the in-channel prefactors add on a two-site ring.

Adsorption and desorption

- Both are environment independent: adsorption fires on every empty site at k_ads and desorption on every occupied site at k_des.

Generator split

- The generator is split as W = F / eps + S.

- F holds the driven in-channel hops only, built from the normalised prefactor kappa.

- S holds adsorption, desorption, both reactions and the cross-channel hops.

- The physical in-channel prefactor is kappa / eps, and the small parameter enters only later.

Size

- A lattice of 15 sites has m = 2^15 = 32768 configurations. A dense m x m array of float64 then takes 8.6 GB, while a column of F or S holds only a few dozen nonzero rates, because a configuration changes by one elementary event at a time.

- F and S are therefore assembled and returned as scipy.sparse matrices in CSR format, and no dense m x m array is formed at any point, whatever the lattice size.

Observable

- What an experiment on such a surface records is not a configuration but the rate at which product leaves it.

- For each configuration, the total reaction propensity is the rate at which a reaction event of either orientation fires in it.

- It is a per-configuration rate, not a probability, and it is the observable the rest of the pipeline averages.

Returns

tuple (F, S, event_rate):

- F: fast generator of driven in-channel hopping, a scipy.sparse CSR matrix of shape (2 ** (nrow * ncol), 2 ** (nrow * ncol)), column convention with zero column sums.

- S: slow generator holding adsorption, desorption, both reactions and the cross-channel hops, same format, shape and convention.

- event_rate: np.ndarray of shape (2 ** (nrow * ncol),), the total rate at which a reaction event of either orientation fires in each configuration.

```python
def lattice_generators(nrow: int, ncol: int, k_ads: float, k_des: float, k_rxn: float, k_rxn_cross: float, kappa: float, drive_bias: float, kappa_cross: float, interaction: float) -> tuple: """Build the fast and slow generators of a driven channelled lattice. Raises ValueError if dimensions are invalid, nrow*ncol exceeds 15, a rate or hopping prefactor is negative or non-finite, drive_bias is non-finite or outside [-1, 1], or interaction is non-finite."""; return F, S, event_rate
```

### Step 2

fast_projector

Goal
----
When the generator of a master equation splits as W = F / eps + S with a small eps, the fast part F acts on a far shorter timescale than S, and the reduction replaces the full probability vector p by a smaller vector of slow variables, p_tilde = Q p. The projector Q must be chosen so that the fast events cannot change p_tilde, Q F = 0: its rows span the left null space of F. That identity is what makes the singular-perturbation hierarchy of the later steps closable, because it removes the unknown higher-order term from every projected equation.

The left null space has one dimension for every recurrent class of the fast process, that is, every set of configurations that the fast events keep circulating among and never leave. When every configuration belongs to such a class, the fast events split the configurations into disjoint superbasins and Q is the 0/1 matrix that assigns each configuration to its own. A fast network can also contain configurations that the fast events leave for good and never re-enter. Such a configuration belongs to no single class, and its weight in Q is shared among the classes the fast process can carry it to.

A basis of the left null space is not unique, so it is returned in one canonical form. Every entry is non-negative, every column sums to one, and row r equals one on every configuration of the r-th recurrent class and zero on every configuration of every other recurrent class. Rows are ordered by the smallest configuration index belonging to their recurrent class, so that the reduced coordinates are canonical and the later steps can be chained without a relabelling. An off-diagonal entry of F counts as a fast transition when it exceeds 1e-12 times the larger of 1 and the largest absolute entry of F.

The fast generator can be large. On the task's lattice it has 32768 configurations, and a dense array with that many rows and columns would take 8.6 GB, so F may be given either as a dense np.ndarray or as a scipy.sparse matrix, and the construction has to work from its nonzero entries: it may not form any dense array with m x m entries. The projector itself is returned as a dense array, since it has only one row per recurrent class.

```python
def fast_projector(F: "np.ndarray | scipy.sparse.spmatrix") -> "np.ndarray": """Parameters: F is a square fast generator in the column convention. Returns: Q, the canonical recurrent-class projector satisfying Q @ F = 0. Raises: ValueError if F is not a non-empty square 2D array, holds a non-finite entry, has a negative off-diagonal rate, or has a column that does not sum to zero."""; return Q
```

### Step 3

driven_steady_state

Goal
----
Collecting the terms of order 1/eps in the expansion p = p^(0) + eps p^(1) + eps^2 p^(2) + ... of the master equation dp/dt = (F / eps + S) p leaves an algebraic condition on the leading term alone: the fast generator annihilates it, F p^(0) = 0.

The leading term is carried by a conditional probability matrix, p^(0) = K^(0) p_tilde^(0), with one column for every row of the projector Q built in the previous step. Column r is supported on the r-th recurrent class, where it is non-negative and sums to one, and it is zero on every other configuration. The two properties

    F K^(0) = 0,   Q K^(0) = I

fix it uniquely, whatever fractional entries Q carries on the configurations that belong to no recurrent class.

Nothing about the shape of that distribution may be assumed. It is whatever the supplied fast generator makes it, and the fast generator is the only place it can come from: the same set of configurations, given a different fast process on it, carries a different leading-order distribution even though the interaction energies of its configurations are unchanged.

F may be a scipy.sparse matrix as large as the lattice's 32768 configurations, and the leading-order map is needed for every class at once, so the construction may not form any dense array with m x m entries. It is returned as a dense array of shape (m, n).

```python
def driven_steady_state(F: "np.ndarray | scipy.sparse.spmatrix", Q: "np.ndarray") -> "np.ndarray": """Parameters: F is a fast generator and Q its projector. Returns: K0, the stationary within-class map satisfying F @ K0 = 0 and Q @ K0 = I. Raises: ValueError if F is not a non-empty square 2D array, holds a non-finite entry, has a negative off-diagonal rate or a column that does not sum to zero; if Q does not have shape (n, m) with n >= 1, holds a non-finite entry, is negative anywhere, or has a column that does not sum to one; if Q does not annihilate F; or if a row of Q does not carry exactly one stationary state of F."""; return K0
```

### Step 4

hierarchy_closure

Goal
----
Insert the expansion p = p^(0) + eps p^(1) + eps^2 p^(2) + ... into the master equation dp/dt = (F / eps + S) p and collect equal powers of eps. The construction is fixed by three conventions and nothing else:

1. the projection of the k-th full-space term is carried by the k-th reduced variable, Q p^(k) = p_tilde^(k);
2. every full-space term is linear in the reduced variables introduced so far, with matrix coefficients that do not depend on time;
3. the coefficient of p_tilde^(k) in p^(k) is the leading-order map K^(0), and every coefficient of a lower-order reduced variable is annihilated by Q.

Carried to order K this determines, in particular, the response matrix that multiplies p_tilde^(0) in each of p^(1), ..., p^(K), and it closes the K + 1 reduced equations into a single linear autonomous system

    dy/dt = L y,   y = (p_tilde^(0), p_tilde^(1), ..., p_tilde^(K)).

Both are needed downstream: L propagates the reduced coefficients, and the response matrices rebuild the full-space probability from them, which is what an observable that varies between the configurations of one class requires. The hierarchy itself contains no eps; the separation parameter enters only when the terms are recombined.

Nothing in the construction may assume that Q is a 0/1 matrix, and nothing asks the fast process to be reversible.

F and S may be scipy.sparse matrices as large as the lattice's 32768 configurations, so the closure may not form any dense array with m x m entries. The responses and L are returned as dense arrays.

```python
def hierarchy_closure(F: "np.ndarray | scipy.sparse.spmatrix", S: "np.ndarray | scipy.sparse.spmatrix", Q: "np.ndarray", K0: "np.ndarray", order: int) -> tuple: """Parameters: fast and slow generators F and S, projector Q, leading map K0, and integer order >= 1. Returns: response matrices resp and reduced hierarchy generator L. Raises: ValueError if F or S is not a non-empty square 2D array, holds a non-finite entry, has a negative off-diagonal rate or a column that does not sum to zero; if F and S differ in shape; if Q does not have shape (n, m) with n >= 1, holds a non-finite entry, is negative anywhere, has a column that does not sum to one, or does not annihilate F; if K0 is not a finite array of shape (m, n), Q @ K0 is not the identity, or F @ K0 does not vanish; if order is not an integer >= 1; or if the constrained system has no solution."""; return resp, L
```

### Step 5

window_averaged_state

Goal
----
The reduced hierarchy dy/dt = L y for y = (p_tilde^(0), p_tilde^(1), ..., p_tilde^(K)) is a linear autonomous system, so once its initial vector is fixed the whole trajectory is determined. The initial condition follows from the requirement that the expansion reproduce the prescribed reduced probabilities at t = 0: the zeroth-order reduced vector equals the initial reduced distribution p_tilde_0, and all higher-order reduced terms start at zero. The number of orders is read off the sizes: L has K + 1 blocks, each of the length of p_tilde_0, along each side.

What is wanted here is not the state at one instant. A rate measurement on a catalyst does not read the surface at a single moment; it collects product over a sampling window that opens after the run has started, and reports the mean over that window. The quantity the rest of the pipeline needs is therefore the time average of each reduced coefficient vector over the window [t_start, t_end],

    ybar = (1 / (t_end - t_start)) * integral from t_start to t_end of y(s) ds,

and it has to be exact. Nothing in this task may depend on a step size, so neither the trajectory nor the average may be produced by an explicit time-stepping scheme or by quadrature on samples of it: the average is a closed-form functional of L, of the initial vector and of the two window edges, and must be evaluated as one.

Two properties of L are worth having in mind. It is block lower triangular, with the reduced generator repeated along its diagonal, so the zeroth-order block evolves on its own and each higher block is driven by the ones below it. And its diagonal blocks are generators of a probability-conserving process, so L is singular.

The K + 1 rows of the result are the eps-independent coefficient vectors of the window average; the k-th order truncation is recombined from them afterwards as ybar^(0) + eps ybar^(1) + ... + eps^k ybar^(k).

```python
def window_averaged_state(L: "np.ndarray", ptilde0: "np.ndarray", t_start: float, t_end: float) -> "np.ndarray": """Parameters: hierarchy generator L, initial reduced distribution ptilde0, and a valid time window. Returns: exact window-averaged coefficient vectors. Raises: ValueError for invalid arrays, distribution, or time window."""; return coeffs
```

### Step 6

turnover_orders

Goal
----
The reduced hierarchy carries only the projections p_tilde = Q p of the probability, and that is enough for an observable that takes one value on all the configurations a reduced coordinate covers, such as the adsorbate count on a lattice whose classes fix it. The reaction propensity is not such an observable: two configurations of the same class hold the same adsorbates in different arrangements, so they offer the reaction a different number of pairs and turn over at different rates. Averaging it over the reduced coefficients alone amounts to assuming that the arrangement inside a class never departs from its leading-order form, which is precisely the assumption the expansion was set up to correct, so that route returns the zeroth-order answer dressed up as a corrected one.

The full-space probability therefore has to be rebuilt before the observable is averaged. Each term of the expansion is a linear combination of the reduced coefficient vectors of its own order and below, and the conventions of the closure step fix every coefficient in it: the leading-order map multiplies the vector of the term's own order, and because the hierarchy has the same form at every order, the responses the closure step produced are all the other coefficients the rebuild needs. They are the only place where the arrangement inside a class can move away from its leading-order form.

Contracting each rebuilt term with the reaction propensity gives the coefficients

    o_k = sum_i rate(i) * p^(k)_i,   k = 0, 1, ..., K,

from which the k-th order truncation of the expected reaction rate is recombined afterwards as sum over l <= k of eps^l o_l. The coefficients themselves carry no eps, and o_0 is exactly what a simulation that treats the fast channel as infinitely fast would report. The order K is read off the sizes of the inputs.

Whatever the reduced coefficient vectors mean, the same assembly applies: here they are the sampling-window averages produced by the previous step, so the o_k are the coefficients of the window-averaged reaction rate.

```python
def turnover_orders(event_rate: "np.ndarray", K0: "np.ndarray", resp: "np.ndarray", coeffs: "np.ndarray") -> "np.ndarray": """Parameters: reaction propensities, leading map, response matrices, and reduced coefficients. Returns: order-by-order reaction-rate coefficients. Raises: ValueError if K0, resp or coeffs is not a 2D array; if event_rate, K0 and resp do not share the configuration count; if resp does not have shape (m, K n) for K >= 1; if coeffs does not have shape (K + 1, n); or if any input holds a non-finite entry."""; return orders
```

### Step 7

rate_rescaling_bias

Goal
----
Final orchestrator. It chains the earlier steps to answer one question about an accelerated simulation of a driven, channelled catalytic surface: if the fast in-channel diffusion is treated as infinitely fast, which is what a scheme that rescales the rate constants of quasi-equilibrated events does once its equilibration test passes, by what fraction is the reaction rate it reports wrong?

The reaction propensity of a configuration is k_rxn times its doubly occupied bonds along a channel plus k_rxn_cross times those across channels. With o_0, ..., o_K the order-by-order coefficients of the window-averaged reaction rate, its k-th order truncation is <R>^(k) = sum over l <= k of eps^l o_l. Its zeroth order is the quasi-equilibrium prediction, the one an infinitely fast diffusion assumption produces, and the requested order is the corrected answer at this separation; the benchmark asks for the third. The number returned is the fractional bias of the quasi-equilibrium prediction,

    Delta = (<R>^(K) - <R>^(0)) / <R>^(0),

a dimensionless ratio, positive when the quasi-equilibrium picture understates the rate. It is referred to the quasi-equilibrium prediction because that is the number an accelerated run actually reports.

The initial reduced distribution is the column of Q that belongs to the configuration initial_config. For the empty surface that column selects a class holding one configuration, and the higher-order reduced terms therefore start at zero.

```python
def rate_rescaling_bias(nrow: int = 3, ncol: int = 5, k_ads: float = 8.5, k_des: float = 4.0, k_rxn: float = 10.0, k_rxn_cross: float = 4.0, kappa: float = 1.0, drive_bias: float = 0.5, kappa_cross: float = 0.4, interaction: float = 1.5, eps: float = 0.02, t_start: float = 0.02, t_end: float = 0.08, initial_config: int = 0, order: int = 3) -> float: """Returns the fractional finite-rate bias of the quasi-equilibrium reaction rate. Raises: ValueError if nrow, ncol, initial_config or order is not an integer; if the lattice does not have between 1 and 15 sites; if initial_config does not index a configuration of the lattice; if order is below 1; if eps is not positive and finite; if a rate constant or hopping prefactor is negative or non-finite, or drive_bias lies outside [-1, 1]; if t_start is not finite and non-negative or t_end is not finite and greater than t_start; or if the quasi-equilibrium rate vanishes, which leaves the fractional bias undefined."""; return bias
```
