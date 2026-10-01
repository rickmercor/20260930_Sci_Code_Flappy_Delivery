# Biology-Biochemistry-17

## Background

Biomolecular processes such as ligand unbinding, conformational switching and folding are rare events: the system spends long times in metastable states and crosses the separating barrier quickly and seldom. Their rate constants and mechanisms are encoded in the ensemble of reactive trajectories, and the committor, the probability that a trajectory started from a configuration reaches the product before the reactant, is the ideal reaction coordinate that orders configurations along the transition. Discrete-state Markov models of conformational dynamics give the same quantities through linear algebra on their transition matrix.

Path sampling methods collect unbiased dynamical trajectories by Monte Carlo in trajectory space: a new path is generated from a point on an old one with the true dynamics and accepted with a rule that preserves the path-space distribution. Transition interface sampling stratifies the transition with a sequence of interfaces between the stable states. Each interface ensemble is biased only through the condition that its paths cross that interface, so the dynamics inside every path remains the unbiased one, and the product of conditional crossing probabilities gives the overall probability of reaching the product state.

Samples drawn under different known biases can be combined into a single estimate of the unbiased distribution. The weighted histogram analysis method does this with binned data by minimising the statistical error; the multistate Bennett acceptance ratio reaches the binless equivalent by maximising the likelihood of all samples at once. The same idea applies when the samples are entire trajectories and the biases are the interface conditions, and the combined estimate should use every sample in every ensemble whose condition it satisfies.

## Problem

Rare conformational transitions of biomolecules, such as a ligand leaving its binding pocket or a peptide folding, cross free-energy barriers of many $k_BT$ and are almost never seen in straightforward simulations. Transition interface sampling places interfaces, level sets of an order parameter, between the reactant state $A$ and the product state $B$, and samples for each interface the ensemble of unbiased dynamical paths that leave $A$ and cross it; reweighting all sampled paths into one reweighted path ensemble then gives crossing probabilities, free-energy projections and committor estimates. When interface sets built on different order parameters are available, for example after the reaction coordinate has been improved between sampling rounds, all their ensembles can be combined in a single maximum-likelihood reweighting of whole trajectories, in the multistate Bennett acceptance ratio sense, that treats every interface ensemble of every set as a selection-biased sample of the same unbiased ensemble of paths leaving $A$. Carry out this joint reweighting for the coarse-grained two-state model below and report the probability that a path leaving $A$ which crosses $x=-0.8$ goes on to reach $B$.

**Model.** Configurations are the sites $(x_i,y_j)$ of a lattice, $x_i=-1.45+0.1\,i$ for $i=0,\dots,29$ and $y_j=-0.95+0.1\,j$ for $j=0,\dots,19$, with free energy $U(x,y)=(x^2-1)^2+y^2$ at inverse temperature $\beta=10$. The dynamics is a discrete-time Metropolis walk: at every step, draw `d = rng.integers(4)` and then `u = rng.random()`; the proposed site is $(i+1,j)$, $(i-1,j)$, $(i,j+1)$ or $(i,j-1)$ for $d=0,1,2,3$; the walker moves there if that site is on the lattice and $u<\exp\{-\beta[U(\text{proposed})-U(\text{current})]\}$, and stays otherwise; both numbers are drawn at every step, and every step adds one frame to the trajectory whether or not the walker moved. State $A$ is $x<-0.9$ and state $B$ is $x>0.9$.

**Interface sets.** Set $s=1,2,3$ uses the order parameter $\lambda^{(1)}=x$, $\lambda^{(2)}=x\cos5^\circ+y\sin5^\circ$ or $\lambda^{(3)}=x+0.1\sin(2\pi y)$. Its interfaces are $\lambda^{(s)}_k=-0.8,-0.7,\dots,-0.1,0.0,0.2$ ($k=1,\dots,10$) for $s=1$ and $s=3$, and $\lambda^{(2)}_k=-0.7,-0.6,\dots,0.0,0.2$ ($k=1,\dots,9$). A path crosses interface $k$ of set $s$ when the largest value of $\lambda^{(s)}$ over its frames is greater than $\lambda^{(s)}_k$.

**Sampling.** Ensemble $(s,k)$ contains the paths whose first frame is in $A$, whose last frame is in $A$ or $B$, whose other frames are in neither state, and which cross interface $k$ of set $s$. Sample each of the 29 ensembles by two-way shooting with its own generator `rng = np.random.default_rng((2026, s, k))`, starting from the path that runs along the row $j=9$ from $i=5$ up to the first site where $\lambda^{(s)}>\lambda^{(s)}_k$ and back to $i=5$ over the same sites. A trial on a current path of $L$ frames, numbered from 0, draws the shooting frame `1 + rng.integers(L - 2)` and runs the walk from its site until the walker enters $A$ or $B$; if it entered $B$, the trial is rejected; otherwise the walk is run again from the shooting site until $A$ or $B$ is entered. The trial path is the first segment reversed followed by the second segment without its first frame; if the trial path does not cross interface $k$ of set $s$, the trial is rejected; otherwise draw `rng.random()` and accept the trial path if that number is below $\min\{1,(L-2)/(L'-2)\}$, where $L'$ is its number of frames, and a rejected trial keeps the current path. A segment still outside both states after $10^5$ steps is abandoned and its trial rejected (this does not occur here). Run 100 trials that are discarded, then 5000 trials, storing the current path after each of them.

**Estimate.** Pool the 145,000 stored paths, each one sample of the ensemble it was drawn from (repeated paths included), so that every ensemble contributes $N=5000$ samples. Reweight them jointly into one ensemble of paths leaving $A$, converging the reweighting to a relative precision of $10^{-12}$, and define

$$P=\frac{\text{total reweighted weight of the paths that cross } x=-0.8 \text{ and reach } B}{\text{total reweighted weight of the paths that cross } x=-0.8}.$$

**Required reasoning.** Justify the computation, covering:
- (a) the reweighting factor of a pooled path in terms of the conditional partition sums of the ensembles, the conditions that fix those partition sums, and which properties of a path the factor depends on;
- (b) the fraction of the total reweighted weight carried by the paths that cross $x=-0.8$, and why $P$ has to be normalised by it for these three sets;
- (c) the number of stored paths, over all 29 ensembles and counting repeats, that reach $B$;
- (d) $P$ obtained in the same way from the ten ensembles of set 1 alone, and from the 19 ensembles of sets 1 and 2 pooled;
- (e) $P$ when each set is instead reweighted on its own, the weights of each set are rescaled so that the mean weight of its paths reaching $B$ is one, and the three rescaled sets are pooled; how it compares with the answer, and what drives the difference, including the value of $P$ that set 2 gives when it is reweighted on its own;
- (f) the exact value of $P$ for this lattice dynamics, obtained from its transition probabilities without sampling, how it is obtained, and the relative deviation of the answer from it;
- (g) how the source study finds the relative statistical error of the logarithm of the jointly reweighted crossing probability to change with the number of combined interface sets, compared with the rescaled combination of (e) at small numbers of paths per ensemble; and how it finds the estimate of the rescaled combination to behave at small numbers of paths per ensemble as more sets are added, and from about how many paths per ensemble that estimate agrees closely with its benchmark crossing probability;
- (h) the relative statistical errors of the logarithm of the mean crossing probability that the source study reports for its host–guest application, for the joint reweighting and for the two rescaled combinations it compares.

**The answer.** Address every point (a)–(h) above, and then give as the final answer $10^5\,P$, as a single decimal number with at least four significant figures.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

metropolis_segment

Goal
----
Run the lattice Metropolis walk of the problem statement from a given site, at a given inverse temperature, until the walker first enters state A or state B or a maximum number of steps has been made, drawing the random numbers from the supplied generator exactly as the problem statement prescribes, and return the visited sites in order. The submitted function must import inside itself whatever it uses.

```python
def metropolis_segment(start: "np.ndarray", beta: float, rng: "np.random.Generator",
                       max_steps: int) -> "np.ndarray":
    """Lattice Metropolis walk from start until state A or B is entered.

    Args:
        start: integer array (i, j) of the starting site.
        beta: inverse temperature (non-negative).
        rng: numpy Generator, advanced by two draws per step.
        max_steps: largest number of steps to make.

    Returns:
        np.ndarray: integer array of shape (n_frames, 2), the visited sites (i, j) in order, starting
            with start; n_frames - 1 is the number of steps made.
    """
    return frames
```

### Step 2

tis_shooting_trial

Goal
----
Perform one two-way shooting trial of the problem statement in the ensemble of paths that leave A and cross the given interface of the given interface set, starting from the supplied current path, and return the path kept after the trial: the trial path if it is accepted, the current path otherwise. The segments are generated with the lattice walk of step 1 and every random number comes from the supplied generator in the order the problem statement prescribes. The submitted function must import inside itself whatever it uses.

```python
def tis_shooting_trial(path: "np.ndarray", set_index: int, interface: float, beta: float,
                       rng: "np.random.Generator", max_steps: int) -> "np.ndarray":
    """One two-way shooting trial in the interface ensemble (set_index, interface).

    Args:
        path: integer array of shape (L, 2), the current path (sites (i, j)), with L >= 3.
        set_index: interface set 1, 2 or 3, selecting the order parameter.
        interface: the interface value that every path of the ensemble must cross.
        beta: inverse temperature.
        rng: numpy Generator, advanced as the problem statement prescribes.
        max_steps: step cap of each segment.

    Returns:
        np.ndarray: integer array of shape (L_kept, 2), the path kept after the trial.
    """
    return new_path
```

### Step 3

sample_interface_ensemble

Goal
----
Sample the interface ensemble (set_index, k) of the problem statement by two-way shooting, with the generator, initial path, interface tables and trial of the problem statement, and return, for each of the n_samples stored paths, the largest value over its frames of each of the three order parameters. The submitted function must import inside itself whatever it uses.

```python
def sample_interface_ensemble(set_index: int, k: int, n_equil: int, n_samples: int, beta: float,
                              seed: int, max_steps: int) -> "np.ndarray":
    """Order-parameter maxima of the stored paths of interface ensemble (set_index, k).

    Args:
        set_index: interface set 1, 2 or 3.
        k: interface index within the set, counted from 1.
        n_equil: number of discarded trials.
        n_samples: number of stored paths.
        beta: inverse temperature.
        seed: first entry of the generator seed (seed, set_index, k).
        max_steps: step cap of each segment.

    Returns:
        np.ndarray: float array of shape (n_samples, 3); row n holds the maxima of lambda1, lambda2 and
            lambda3 over the n-th stored path.
    """
    return maxima
```

### Step 4

joint_log_partition_sums

Goal
----
Given pooled paths from interface ensembles of one or more interface sets, described by the maxima of the sets' order parameters over each path, together with the interfaces of each set and the number of samples drawn in each ensemble, return the natural logarithms of the maximum-likelihood (multistate Bennett acceptance ratio) estimates of the conditional partition sums of all the ensembles, with every pooled path used in the joint estimate and the reweighted weights of the pooled paths normalised to sum to one. The submitted function must import inside itself whatever it uses.

```python
def joint_log_partition_sums(maxima: "np.ndarray", interfaces: list, counts: list) -> "np.ndarray":
    """Log conditional partition sums of all interface ensembles from the jointly reweighted pooled paths.

    Args:
        maxima: float array of shape (n_paths, n_sets); entry (p, s) is the maximum of the order
            parameter of set s over pooled path p.
        interfaces: list of n_sets increasing float arrays, the interfaces of each set.
        counts: list of n_sets arrays; counts[s][k] is the number of samples drawn in ensemble (s, k).

    Returns:
        np.ndarray: float array of length sum of len(interfaces[s]), the log partition sums ordered set by
            set and interface by interface, for path weights normalised to sum to one.

    Raises:
        ValueError: if a pooled path crosses no interface of any set, or the self-consistent solution
            does not converge because the pooled data do not determine it.
    """
    return log_z
```

### Step 5

crossing_probability

Goal
----
Given the pooled paths, interfaces and sample counts of step 4 and the log partition sums of all the ensembles, reweight every pooled path as the joint maximum-likelihood reweighting prescribes and return, for each value lam in lams, the natural logarithm of the reweighted probability that a path which crosses lambda1 = x = lam_ref also crosses x = lam. The submitted function must import inside itself whatever it uses.

```python
def crossing_probability(maxima: "np.ndarray", interfaces: list, counts: list, log_z: "np.ndarray",
                         lam_ref: float, lams: "np.ndarray") -> "np.ndarray":
    """Log of the reweighted probability of crossing each x = lam, given that x = lam_ref is crossed.

    Args:
        maxima: float array of shape (n_paths, n_sets), as in step 4; column 0 is lambda1 = x.
        interfaces: list of n_sets increasing float arrays, as in step 4.
        counts: list of n_sets arrays of sample counts, as in step 4.
        log_z: float array of the log partition sums, ordered as the output of step 4.
        lam_ref: the conditioning value of x.
        lams: float array of values of x.

    Returns:
        np.ndarray: float array of len(lams), the natural logarithms of the conditional probabilities.
    """
    return log_p
```

### Step 6

rescaled_crossing_probability

Goal
----
Reweight each interface set on its own, with the maximum-likelihood reweighting of step 4 restricted to that set's ensembles and order parameter, rescale the weights of each set so that the mean weight of its paths reaching B is one, pool the rescaled sets, and return the natural logarithm of the probability that a path which crosses x = lam_ref reaches B in the pooled ensemble. The submitted function must import inside itself whatever it uses.

```python
def rescaled_crossing_probability(maxima_by_set: list, interfaces: list, counts: list,
                                  lam_ref: float) -> float:
    """Log crossing probability into B from independently reweighted sets rescaled to unit reactive weight.

    Args:
        maxima_by_set: list of float arrays of shape (n_paths_s, 3), the stored paths of each set.
        interfaces: list of increasing float arrays, the interfaces of each set.
        counts: list of arrays of sample counts per ensemble of each set.
        lam_ref: the conditioning value of x.

    Returns:
        float: natural logarithm of the pooled conditional probability of reaching B.

    Raises:
        ValueError: if a set has no path that reaches B.
    """
    return log_p
```

### Step 7

exact_crossing_probability

Goal
----
Compute, without sampling, the natural logarithm of the exact probability that a path of the lattice Metropolis walk which leaves A and crosses x = lam_ref goes on to reach B, at inverse temperature beta, from the transition probabilities of the walk and its equilibrium distribution. The submitted function must import inside itself whatever it uses.

```python
def exact_crossing_probability(beta: float, lam_ref: float) -> float:
    """Exact log probability that a path leaving A which crosses x = lam_ref reaches B.

    Args:
        beta: inverse temperature (non-negative).
        lam_ref: the conditioning value of x, with -0.9 <= lam_ref < 0.9.

    Returns:
        float: natural logarithm of the conditional probability.
    """
    return log_p
```

### Step 8

compare_crossing_estimates

Goal
----
Run the whole comparison of the problem statement for the given numbers of discarded and stored trials, inverse temperature, seed and step cap: sample all 29 interface ensembles, and return 1e5 times the probability that a path leaving A which crosses x = -0.8 reaches B, estimated by the joint reweighting of the three sets, by the joint reweighting of sets 1 and 2, by set 1 alone, by the rescaled combination of the three sets, and exactly. The submitted function must import inside itself whatever it uses.

```python
def compare_crossing_estimates(n_equil: int, n_samples: int, beta: float, seed: int,
                               max_steps: int) -> "np.ndarray":
    """1e5 x the crossing probability into B, given x = -0.8 is crossed, by five estimators.

    Args:
        n_equil: discarded trials per ensemble.
        n_samples: stored paths per ensemble.
        beta: inverse temperature.
        seed: first entry of every generator seed (seed, s, k).
        max_steps: step cap of each segment.

    Returns:
        np.ndarray: float array of length 5, 1e5 times the probability from the joint reweighting of sets
            1-3, the joint reweighting of sets 1-2, set 1 alone, the rescaled combination of sets 1-3,
            and the exact value.
    """
    return estimates
```
