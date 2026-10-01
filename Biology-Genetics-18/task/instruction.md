# Biology-Genetics-18

## Background

Coalescent theory describes how the ancestral lineages of sampled genomes merge as they are traced back in time. The rate at which they merge depends on population sizes, on migration between subpopulations, and on events such as population splits and admixture. Methods that infer ancestral recombination graphs now supply estimated coalescence times for many genomic segments, so summaries of these times can be fitted directly to demographic models rather than being integrated out through allele-frequency statistics.
 
A standard summary is the instantaneous coalescence rate: the rate at which a coalescence occurs at time t in the past, given that none has occurred more recently. For a pair of lineages in a single panmictic population this rate is inversely proportional to the effective population size. Under population structure it also depends on where the lineages were sampled and on how migration and admixture move them between subpopulations. Larger samples coalesce sooner and so are more sensitive to recent history. Coalescences between lineages sampled from different populations, called cross-coalescences, are especially sensitive to recent gene flow.
 
Whether a particular summary can resolve a demographic parameter can be quantified locally by the Fisher information that the distribution of an observed coalescence time carries about that parameter. Comparing this information across summaries and sampling designs indicates which observations are most useful for detecting a given demographic event.

## Problem

Reconstructed genealogies make coalescence times directly observable, and the time to the first coalescence among several sampled lineages, or to the first coalescence between lineages sampled from two different populations, carries information about recent structured demographic history. Recent work computes the exact hazards of these first-event times, the first-coalescence rate $\mathrm{ICR}_k$ and the cross-coalescence rate $\mathrm{CCR}$, for arbitrary sampling configurations by following the numbers of ancestral lineages in each deme backward in time through migration, admixture pulses and population splits, and it uses the resulting densities for local identifiability analysis based on Fisher information. Your task is to use that framework to decide which of two first-event times computed from the same six genomes is more informative about a recent admixture pulse: the inputs are a two-deme demography and a sampling design, and the output is a ratio of Fisher informations.
 
Measure time $t$ in generations before the present. Each pair of lineages in a deme of diploid size $N$ coalesces at rate $$1/(2N)$$ per generation, and migration is given by backward-time rates per lineage per generation, where $$M_{ij}$$ is the rate at which a lineage in deme $i$ moves to deme $j$. There are two present-day demes, pop1 (index 0) and pop2 (index 1). On $0 \le t < 150$ the diploid sizes are $N_0 = 4000$ and $N_1 = 1000$, with $M_{01} = 10^{-4}$ and $M_{10} = 2\times10^{-4}$. At $t = 150$ an admixture pulse occurs: forward in time, a fraction $\alpha = 0.25$ of pop1 is replaced by migrants from pop2, so pop2 is the source and pop1 is the destination. On $150 \le t < 1000$ the sizes are $N_0 = 4000$ and $N_1 = 2500$, with the same migration rates. At $t = 1000$ pop1 and pop2 merge, backward in time, into one panmictic ancestral population of diploid size $5000$ that persists for all older times. Sample three lineages from pop1 and three from pop2. Let $T_k$ be the time of the first coalescence among all six lineages. Colour the pop1 lineages red and the pop2 lineages blue, and let $T_\times$ be the time of the first coalescence between a red and a blue lineage; red–red and blue–blue coalescences may occur earlier and do not end this process.
 
For each of $T_k$ and $T_\times$, compute the exact survival function, hazard and density, and then the Fisher information about $\alpha$ carried by one observation of that time, $I_k(\alpha)$ and $I_\times(\alpha)$, evaluated at $\alpha = 0.25$ with every other demographic parameter held fixed. The information integral runs over all $t \ge 0$, including the ancestral epoch. Report the ratio $R = I_\times(\alpha)/I_k(\alpha)$ to at least eight significant figures. In the reasoning, first state three source-method relations: (1) the equation you propagate for the survival-weighted lineage-configuration probabilities of each first-event time, including how same-colour coalescences enter the cross-coalescence calculation and how the hazard is obtained from these probabilities; (2) the backward-time effect of the pulse on individual lineages; and (3) the Fisher-information integral for a single first-event time. Then report, each to at least six significant figures: $\Pr(T_k > 150)$; $\mathrm{ICR}_k$ immediately before and immediately after the pulse; $\mathrm{ICR}_k$ for $t \ge 1000$; $\mathrm{CCR}$ immediately before and immediately after the pulse; $\Pr(T_\times < 1000)$; the contribution of first-event times earlier than $t = 150$ to each information; $\partial \log \Pr(T_k > 1000)/\partial\alpha$ and the contribution of times $t \ge 1000$ to $I_k$; the contribution of times $t \ge 1000$ to $I_\times$; $I_k$; $I_\times$; and $R$. End with one sentence interpreting $R$ for this sampling design.
 
Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_occupancy_rate_matrix.py

Goal
----
Build the survival-weighted rate matrix of the exact first-coalescence calculation for m lineages in d demes.

```python
def occupancy_rate_matrix(m: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    """Return Q_mig - D_lambda on the occupancy states of m lineages in d demes.
 
    Parameters
    ----------
    m : int
        Number of distinct lineages, an integer m >= 0 (``bool`` is invalid).
    sizes : np.ndarray
        Finite, strictly positive diploid deme sizes N_i with shape (d,), d >= 1.
    migration : np.ndarray
        Finite (d, d) matrix of backward-time migration rates; entry (i, j),
        i != j, is the rate per lineage per generation at which a lineage in
        deme i moves to deme j. Off-diagonal entries must be nonnegative;
        diagonal entries are ignored.
 
    Returns
    -------
    rate_matrix : np.ndarray
        Float array of shape (S, S), S = C(m + d - 1, d - 1). States are all
        integer vectors x with nonnegative entries summing to m, ordered in
        ascending lexicographic order of the tuple (x_1, ..., x_d); for d = 2
        and m = 2 the order is (0, 2), (1, 1), (2, 0). Entry [x, y] for y != x
        is the migration rate from x to y, and entry [x, x] equals minus the
        total outgoing migration rate minus lambda(x).
 
    Raises
    ------
    ValueError
        If m is not an integer >= 0, if sizes is not a finite strictly positive
        one-dimensional array, or if migration does not have shape (d, d), is
        not finite, or has a negative off-diagonal entry.
    """
    return rate_matrix
```

### Step 2

02_pulse_occupancy_kernel.py

Goal
----
Build the transition matrix that an admixture pulse applies to the occupancy distribution of m lineages.

```python
def pulse_occupancy_kernel(m: int, d: int, source: int, dest: int, alpha: float) -> "np.ndarray":
    """Return the backward-time pulse transition matrix on occupancy states.
 
    Parameters
    ----------
    m : int
        Number of lineages, an integer m >= 0 (``bool`` is invalid).
    d : int
        Number of demes, an integer d >= 2.
    source : int
        Zero-based index of the forward-time source deme, 0 <= source < d.
    dest : int
        Zero-based index of the forward-time destination deme, 0 <= dest < d,
        dest != source.
    alpha : float
        Finite pulse proportion with 0 <= alpha <= 1.
 
    Returns
    -------
    kernel : np.ndarray
        Row-stochastic float array of shape (S, S) on the occupancy states of
        m lineages in d demes, ordered in ascending lexicographic order of
        (x_1, ..., x_d). Entry [x, y] is the probability that configuration x
        on the younger side of the pulse becomes y on the older side.
 
    Raises
    ------
    ValueError
        If m is not an integer >= 0, d is not an integer >= 2, source or dest
        is not an integer index in [0, d), source equals dest, or alpha is not
        a finite number in [0, 1].
    """
    return kernel
```

### Step 3

03_first_coalescence_curve.py

Goal
----
Compute the exact first-coalescence hazard ICR_k(t) and log-survival log S_k(t) for a structured sample under a pulse-then-split demography.

```python
def first_coalescence_curve(sample_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    """Return [ICR_k(t), log S_k(t)] at the requested times.
 
    Parameters
    ----------
    sample_counts : np.ndarray
        One-dimensional array of shape (d,) with the number of lineages
        sampled in each present-day deme; entries are nonnegative integers
        (integral floats are accepted) and k = sum(sample_counts) >= 2.
    demography : dict
        Dictionary with exactly these required keys (extra keys are ignored):
        ``"sizes"`` : (2, d) finite, strictly positive diploid sizes, row 0
        for [0, t_pulse) and row 1 for [t_pulse, t_split);
        ``"migration"`` : (2, d, d) finite backward-time migration rates per
        lineage per generation (entry [e, i, j] moves a lineage from deme i
        to deme j in interval e; off-diagonal entries nonnegative, diagonal
        entries ignored);
        ``"t_pulse"``, ``"t_split"`` : finite times with
        0 < t_pulse < t_split;
        ``"pulse_source"``, ``"pulse_dest"`` : distinct integer deme indices
        in [0, d) giving the forward-time source and destination of the pulse;
        ``"alpha"`` : finite pulse proportion in [0, 1];
        ``"ancestral_size"`` : finite, strictly positive diploid size of the
        ancestral deme. Requires d >= 2.
    times : np.ndarray
        Nonempty one-dimensional array of finite times t >= 0, in any order.
 
    Returns
    -------
    curve : np.ndarray
        Float array of shape (2, len(times)); row 0 is ICR_k(t) in
        coalescences per generation and row 1 is log S_k(t) (natural log),
        both right-continuous at the event times, in the order of ``times``.
        Values must agree with the exact solution to relative accuracy 1e-9.
 
    Raises
    ------
    ValueError
        If sample_counts is not a one-dimensional array of d nonnegative
        integers with total k >= 2, if any demography key is missing or
        violates the constraints above, or if times is empty, not
        one-dimensional, not finite, or contains a negative value.
    """
    return curve
```

### Step 4

04_cross_coalescence_rate_matrix.py

Goal
----
Build the survival-weighted rate matrix of the exact cross-coalescence calculation for red and blue lineages in d demes.

```python
def cross_coalescence_rate_matrix(n_red: int, n_blue: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    """Return the cross-coalescence rate matrix Q_x on red/blue count states.
 
    Parameters
    ----------
    n_red : int
        Maximum total number of red lineages, an integer >= 1.
    n_blue : int
        Maximum total number of blue lineages, an integer >= 1.
    sizes : np.ndarray
        Finite, strictly positive diploid deme sizes with shape (d,), d >= 1.
    migration : np.ndarray
        Finite (d, d) backward-time migration rates per lineage per
        generation; entry (i, j), i != j, moves a lineage of either colour
        from deme i to deme j. Off-diagonal entries must be nonnegative;
        diagonal entries are ignored.
 
    Returns
    -------
    rate_matrix : np.ndarray
        Float array of shape (S, S). The states are all interleaved vectors
        (rho_1, beta_1, ..., rho_d, beta_d) of nonnegative integers with
        sum_i rho_i <= n_red and sum_i beta_i <= n_blue, ordered in ascending
        lexicographic order of that interleaved tuple; for d = 1,
        n_red = n_blue = 1 the order is (0, 0), (0, 1), (1, 0), (1, 1).
 
    Raises
    ------
    ValueError
        If n_red or n_blue is not an integer >= 1, if sizes is not a finite
        strictly positive one-dimensional array, or if migration does not have
        shape (d, d), is not finite, or has a negative off-diagonal entry.
    """
    return rate_matrix
```

### Step 5

05_cross_coalescence_curve.py

Goal
----
Compute the exact cross-coalescence rate CCR(t) and log-survival log S_x(t) of red and blue samples under a pulse-then-split demography.

```python
def cross_coalescence_curve(red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    """Return [CCR(t), log S_x(t)] at the requested times.
 
    Parameters
    ----------
    red_counts : np.ndarray
        One-dimensional array of shape (d,) with the number of red lineages
        sampled in each present-day deme; nonnegative integers (integral
        floats are accepted) with total >= 1.
    blue_counts : np.ndarray
        One-dimensional array of shape (d,) with the number of blue lineages
        sampled in each present-day deme; nonnegative integers (integral
        floats are accepted) with total >= 1.
    demography : dict
        Dictionary with the required keys ``"sizes"`` (2, d),
        ``"migration"`` (2, d, d), ``"t_pulse"``, ``"t_split"``,
        ``"pulse_source"``, ``"pulse_dest"``, ``"alpha"`` and
        ``"ancestral_size"``, with the same meanings and constraints as in
        ``first_coalescence_curve``: positive diploid sizes, nonnegative
        off-diagonal backward-time migration rates (diagonal ignored),
        0 < t_pulse < t_split, distinct integer deme indices for the
        forward-time pulse source and destination, alpha in [0, 1] and a
        positive ancestral size; d >= 2. Extra keys are ignored.
    times : np.ndarray
        Nonempty one-dimensional array of finite times t >= 0, in any order.
 
    Returns
    -------
    curve : np.ndarray
        Float array of shape (2, len(times)); row 0 is CCR(t) in
        cross-coalescences per generation and row 1 is log S_x(t) (natural
        log), right-continuous at the event times and in the order of
        ``times``. Values must agree with the exact solution to relative
        accuracy 1e-9.
 
    Raises
    ------
    ValueError
        If red_counts or blue_counts is not a one-dimensional array of d
        nonnegative integers with total >= 1, if any demography key is missing
        or violates its constraints, or if times is empty, not
        one-dimensional, not finite, or contains a negative value.
    """
    return curve
```

### Step 6

06_first_event_density_score.py

Goal
----
Evaluate the density of a first-event time and its score with respect to the pulse proportion.

```python
def first_event_density_score(kind: str, samples: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    """Return [f(t), d log f(t) / d alpha] for a first-event time.
 
    Parameters
    ----------
    kind : str
        Exactly ``"icr"`` for the first coalescence time T_k of all sampled
        lineages, or ``"ccr"`` for the first red-blue cross-coalescence time.
    samples : np.ndarray
        For ``"icr"``: shape (d,), lineage counts per present-day deme, as for
        ``first_coalescence_curve``. For ``"ccr"``: shape (2, d); row 0 holds
        the red counts and row 1 the blue counts, as for
        ``cross_coalescence_curve``.
    demography : dict
        Pulse-then-split demography with the keys and constraints documented
        for ``first_coalescence_curve``, except that alpha must satisfy
        0 < alpha < 1 strictly.
    times : np.ndarray
        Nonempty one-dimensional array of finite times t >= 0, in any order.
 
    Returns
    -------
    density_score : np.ndarray
        Float array of shape (2, len(times)). Row 0 is f(t) = c(t) S(t) per
        generation, with relative accuracy 1e-9; row 1 is the score
        d log f(t) / d alpha, with absolute error at most
        1e-8 * max(1, |score|), and 0 wherever f(t) = 0. Both rows are
        right-continuous at the event times.
 
    Raises
    ------
    ValueError
        If kind is not ``"icr"`` or ``"ccr"``, if samples has the wrong shape
        or invalid counts for that kind, if alpha is not strictly between 0
        and 1, or if the demography or times violate the curve contracts.
    """
    return density_score
```

### Step 7

07_pulse_fisher_information.py

Goal
----
Compute the Fisher information about the pulse proportion carried by one observed first-event time.

```python
def pulse_fisher_information(kind: str, samples: "np.ndarray", demography: dict) -> float:
    """Return the Fisher information I(alpha) of one first-event time.
 
    Parameters
    ----------
    kind : str
        Exactly ``"icr"`` (first coalescence time of all sampled lineages) or
        ``"ccr"`` (first red-blue cross-coalescence time).
    samples : np.ndarray
        For ``"icr"``: shape (d,), lineage counts per present-day deme. For
        ``"ccr"``: shape (2, d), with red counts in row 0 and blue counts in
        row 1. Same contracts as ``first_event_density_score``.
    demography : dict
        Pulse-then-split demography with the keys and constraints documented
        for ``first_event_density_score`` (0 < alpha < 1 strictly).
 
    Returns
    -------
    information : float
        I(alpha) = integral_0^infinity f(t) [d log f(t) / d alpha]^2 dt as a
        native Python float, with relative error at most 1e-6.
 
    Raises
    ------
    ValueError
        If kind, samples or demography violate the contracts of
        ``first_event_density_score``.
    """
    return information
```

### Step 8

08_pulse_information_ratio.py

Goal
----
Compare how much information about an admixture pulse the first cross-coalescence time and the first coalescence time carry.

```python
def pulse_information_ratio(sample_counts: "np.ndarray", red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict) -> float:
    """Return I_x(alpha) / I_k(alpha) for one pulse-then-split demography.
 
    Parameters
    ----------
    sample_counts : np.ndarray
        Shape (d,), lineage counts per present-day deme used for the first
        coalescence time T_k; nonnegative integers with total >= 2.
    red_counts : np.ndarray
        Shape (d,), red lineage counts per deme for T_x; nonnegative integers
        with total >= 1.
    blue_counts : np.ndarray
        Shape (d,), blue lineage counts per deme for T_x; nonnegative integers
        with total >= 1.
    demography : dict
        Pulse-then-split demography with the keys and constraints documented
        for ``first_event_density_score`` (0 < alpha < 1 strictly).
 
    Returns
    -------
    ratio : float
        I_x(alpha) / I_k(alpha) as a native Python float, with relative error
        at most 1e-6.
 
    Raises
    ------
    ValueError
        If any input violates the contracts of ``pulse_fisher_information``,
        or if I_k(alpha) is not strictly positive.
    """
    return ratio
```
