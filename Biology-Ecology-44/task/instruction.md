# Biology-Ecology-44

## Background

Diseases that reinfect the same host, such as many respiratory and enteric infections of people and animals, settle into an endemic state in which infection and recovery balance. On a contact network that state is quasi-stationary rather than truly stationary, since a finite population eventually loses the infection, but it persists for a time that grows rapidly with the size of the population, and its properties are what surveillance observes. Two of them matter for control: how much of the population is infectious, and how soon an individual who has just recovered is infected again, which sets how long a recovered individual contributes nothing to transmission and how bursty repeated infection is at the level of a single host.

Exact treatments of the contact process on a network scale exponentially with its size, so moment closures are the working tools. Mean-field closures treat nodes as independent and misplace the epidemic threshold. Pair closures treat each connected pair exactly and are much better, but they carry an assumption about time as well as space: each node is either susceptible or infectious, with no record of how long it has been so. In a pair closure the infection pressure on a susceptible node is therefore the same the instant it recovers as it is long afterwards, and the predicted interval between a node's infections is exponential. Simulations show otherwise, because a node that has just recovered is likely to have been infected by, or to have infected, a neighbour who is still infectious. Improving on this without abandoning the tractability of a closure over edges requires carrying that temporal correlation in the state of the pair.

## Problem

A directly transmitted pathogen that confers no lasting immunity circulates in a closed community whose contacts form Zachary's karate club network: 34 members labelled 0 to 33, joined by 78 undirected and unweighted ties, the standard edge list distributed as the karate club graph of NetworkX with its edge weights ignored. Transmission follows the continuous-time susceptible-infectious-susceptible contact process: each infectious member infects each susceptible contact at rate $\beta = 0.3$ and recovers to susceptibility at unit rate, so that time is measured in mean infectious periods. The exact process has $2^{34}$ configurations, and the standard pair approximation, which treats every tie exactly and couples ties only through their mean infection pressure, makes the time a member spends susceptible memoryless, although a member who has just recovered is surrounded by contacts who are still likely to be infectious. Work instead with the pair approximation extended in time, in which the susceptible class of every member is resolved into a chain of $K = 8$ equally susceptible stages that act as a clock of the time since the member last recovered, so that lumping the stages reproduces the contact process exactly, and in which the rate at which the stages age is the one the construction prescribes in terms of $\beta$, the mean degree of the network and $K$. In the endemic state that this approximation predicts, compute the probability that member 33, the member with the most ties (17), remains uninfected for longer than $t = 2.5$ after the moment it recovers.

Report the probability to at least five decimal places. The answer is graded within $0.0006$.

Report also, inside the reasoning: the stage ageing rate; the infection rate that member 33 experiences in the susceptible stage it enters on recovery and in the oldest susceptible stage; the stationary probability that member 33 is infectious; the stationary probability of being infectious averaged over all 34 members; and the same survival probability for member 33 as predicted by the standard pair approximation.

State as well, in a line each, the choices your calculation rests on: the rule you used for the stage ageing rate, written as a formula in the quantities it depends on; which susceptible stage a member occupies immediately after it recovers, and what becomes of a member that reaches the oldest stage; how you obtained the joint distribution of a connected pair, and how you closed the system so that the reported quantities are evaluated at a self-consistent solution rather than after a fixed number of passes; how the infection pressure reaching each end of a tie from the rest of the network is formed; how the rate passed along a tie depends on the stage of the member receiving it; how member 33's infection rate in each stage follows from those rates; and which stage your survival calculation starts in.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show the few scalars that determine the final number, together with the brief statements of construction asked for above, and nothing else. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

step_01_nonbacktracking_cavity_rates

Goal
----
When the pair (i, j) is solved, the infection pressure on node j from outside the pair must exclude the message from i itself, since that interaction is already represented exactly inside the pair. The pressure on j in stage x is therefore the sum of the stage-x messages that j receives from all its neighbours except i. As an operator acting on messages indexed by directed edges this is the non-backtracking operator of the graph, applied separately to each stage:

C[i, j, x] = A[i, j] * sum over k != i of A[j, k] * phi[j, k, x].

The value C[i, j, x] is the external infection rate felt by node j when it sits in stage x and is paired with i, and C[j, i, x] is the external rate felt by node i when paired with j. Entries off the edge set are zero. The total pressure on node i in stage x from all its neighbours is the backtracking sum over all k, and it differs from C[j, i, x] by exactly phi[i, j, x].

The adjacency matrix describes a simple undirected graph: square, symmetric, entries 0 or 1, zero diagonal. Messages are non-negative and vanish on non-edges.

```python
def nonbacktracking_cavity_rates(adjacency: np.ndarray, messages: np.ndarray) -> np.ndarray:
    """Apply the stage-resolved non-backtracking operator to edge messages.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric 0/1 adjacency matrix of shape (N, N) with zero diagonal.
    messages : np.ndarray
        Messages phi[i, j, x] of shape (N, N, K), non-negative and zero off the edge set.

    Returns
    -------
    np.ndarray
        Cavity rates C[i, j, x] = A[i, j] * sum over k != i of A[j, k] * phi[j, k, x], of shape (N, N, K).

    Raises
    ------
    ValueError
        When the adjacency fails to be a square symmetric 0/1 matrix of size at least 2 with zero diagonal, or when the messages fail to be a finite non-negative array of shape (N, N, K) with K at least 1 that vanishes off the edge set.
    """
    return
```

### Step 2

step_02_pair_transition_matrix

Goal
----
A connected pair (i, j) is treated as a Markov chain on the (K + 1)^2 joint states. Inside the pair, an infectious node infects its susceptible partner at rate beta. From outside the pair, the first node, when in stage S(x), is infected at the external rate a[x], and the second node, when in stage S(y), at the external rate b[y]. The transitions are, for every x and y from 1 to K:

- recovery, at rate 1: (I, I) to (I, S(K)) and to (S(K), I); (I, S(y)) to (S(K), S(y)); (S(x), I) to (S(x), S(K));
- ageing, at rate gamma, for x or y at least 2: (S(x), I) to (S(x - 1), I); (I, S(y)) to (I, S(y - 1)); (S(x), S(y)) to (S(x - 1), S(y)) and to (S(x), S(y - 1));
- infection of a node whose partner is infectious: (S(x), I) to (I, I) at rate beta + a[x]; (I, S(y)) to (I, I) at rate beta + b[y];
- infection of a node whose partner is susceptible: (S(x), S(y)) to (I, S(y)) at rate a[x] and to (S(x), I) at rate b[y].

The diagonal holds minus the total exit rate, so every row sums to zero.

State ordering. A single node's state is encoded by the integer 0 for I and x for S(x). The joint state (u, v), with u the state of the first node and v that of the second, has flat index u * (K + 1) + v. Entry [r, c] of the returned matrix is the rate from state r to state c.

```python
def pair_transition_matrix(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Assemble the generator of the memory-augmented SIS chain of one connected pair.

    Parameters
    ----------
    external_first : np.ndarray
        External infection rates a[x] of the first node in stage S(x), length K.
    external_second : np.ndarray
        External infection rates b[y] of the second node in stage S(y), length K.
    beta : float
        Per-contact transmission rate, above zero.
    gamma : float
        Ageing rate between successive susceptible stages, non-negative.

    Returns
    -------
    np.ndarray
        Generator of shape ((K + 1)^2, (K + 1)^2), flat index u * (K + 1) + v with 0 for I and x for S(x).

    Raises
    ------
    ValueError
        When the external rates fail to be finite non-negative one-dimensional arrays of equal length at least 1, when beta fails to be finite and above zero, or when gamma fails to be finite and non-negative.
    """
    return
```

### Step 3

step_03_pair_stationary_distribution

Goal
----
Solve the balance equations of the pair chain for its stationary distribution, replacing one redundant equation by the normalisation, and return the distribution in the flat state ordering of the previous step.

Solving the linear system in floating point can leave entries of order the machine precision with either sign. The distribution is returned with every entry floored at 1e-16 and then renormalised to unit sum, so that the conditional probabilities formed from it in the next step are always defined. At any endemic operating point the floor lies many orders of magnitude below every entry.

```python
def pair_stationary_distribution(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Compute the stationary distribution of the memory-augmented SIS chain of one connected pair.

    Parameters
    ----------
    external_first : np.ndarray
        External infection rates a[x] of the first node in stage S(x), length K.
    external_second : np.ndarray
        External infection rates b[y] of the second node in stage S(y), length K.
    beta : float
        Per-contact transmission rate, above zero.
    gamma : float
        Ageing rate between successive susceptible stages, non-negative.

    Returns
    -------
    np.ndarray
        Stationary distribution of length (K + 1)^2 in the flat ordering u * (K + 1) + v.

    Raises
    ------
    ValueError
        When the inputs are invalid as for the pair generator, or when the chain has no unique stationary distribution.
    """
    return
```

### Step 4

step_04_stage_conditional_infection_rates

Goal
----
Read both messages of a connected pair off the stationary distribution of its chain, one value per susceptible stage in each direction.

Both directions are read from the same pair distribution. With the pair ordered as (first, second), row 0 of the returned array holds beta P(I_second | S(x)_first), the message received by the first node, and row 1 holds beta P(I_first | S(x)_second), the message received by the second node.

```python
def stage_conditional_infection_rates(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Read the stage-conditional infection messages of both ends off the stationary pair distribution.

    Parameters
    ----------
    external_first : np.ndarray
        External infection rates a[x] of the first node in stage S(x), length K.
    external_second : np.ndarray
        External infection rates b[y] of the second node in stage S(y), length K.
    beta : float
        Per-contact transmission rate, above zero.
    gamma : float
        Ageing rate between successive susceptible stages, non-negative.

    Returns
    -------
    np.ndarray
        Array of shape (2, K): row 0 is beta P(I_second | S(x)_first), row 1 is beta P(I_first | S(x)_second).

    Raises
    ------
    ValueError
        When the inputs are invalid as for the pair generator, or when the pair chain has no unique stationary distribution.
    """
    return
```

### Step 5

step_05_memory_pair_messages

Goal
----
The ageing rate of the susceptible stages is not a free parameter of the fit. It is tied to the time scale of infection so that the stage clock resolves the interval over which reinfection happens and that resolution sharpens as K grows:

gamma = beta q sqrt(K - 1), with q = (sum over i, j of A[i, j]) / N - 1,

where q is one less than the mean degree of the network. With K = 1 the ageing rate is zero and the stage structure is empty.

The iteration is sequential, edge by edge. A sweep visits every undirected edge (i, j) with i < j in row-major order of the adjacency matrix. For each edge it computes the cavity rates from the messages as they currently stand, including those already updated earlier in the same sweep, solves the pair with i first and j second, where the external rates on i are C[j, i] and those on j are C[i, j], and overwrites phi[i, j] with row 0 and phi[j, i] with row 1 of step 4. Updating in place reaches the same fixed point as a synchronous update, in which every edge of a sweep sees only the messages of the previous sweep, in about half as many sweeps. The sweep repeats until the largest absolute change of any message over a sweep falls below tol. The iteration starts from messages A[i, j] times K values evenly spaced from 0.1 to 0.2 (the single value 0.1 when K = 1). Above the epidemic threshold of the approximation this converges to the endemic fixed point, which is reached from any strictly positive start. Below it the oldest-stage messages decay geometrically towards zero, while the younger-stage messages become ratios of probabilities that vanish together and carry no information; as soon as every oldest-stage message after a sweep lies below 1e-12 the stage therefore stops and returns the disease-free solution, all messages zero, with residual zero. If neither has happened after max_sweeps sweeps the stage raises RuntimeError.

```python
def memory_pair_messages(
    adjacency: np.ndarray,
    beta: float,
    num_stages: int,
    tol: float,
    max_sweeps: int,
) -> dict:
    """Solve the self-consistent stage-resolved messages of the memory-augmented pair approximation.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric 0/1 adjacency matrix of shape (N, N) with zero diagonal.
    beta : float
        Per-contact transmission rate, above zero, with unit recovery rate.
    num_stages : int
        Number K of susceptible stages, at least 1.
    tol : float
        Convergence tolerance on the largest absolute message change per sweep.
    max_sweeps : int
        Maximum number of sweeps.

    Returns
    -------
    dict
        Under the keys messages, ageing_rate, mean_degree_less_one, sweeps and residual.

    Raises
    ------
    ValueError
        When the adjacency is not a simple undirected graph with at least one edge, when beta or tol fails to be finite and above zero, or when num_stages or max_sweeps fails to be an integer at least 1.
    RuntimeError
        When the iteration has not converged after max_sweeps sweeps.
    """
    return
```

### Step 6

step_06_reinfection_survival

Goal
----
Two quantities follow from it. The first is the stationary probability that the node is infectious, obtained from the stationary distribution of the (K + 1)-state chain. The second is the distribution of the inter-infection time Delta_I, the time from the moment the node recovers to the moment it is next infected. At recovery the node is in S(K). Let R be the generator of the chain with I made absorbing: R[x, x - 1] = gamma for x at least 2, R[x, I] = lambda[x], the diagonal set so that every row of the susceptible block sums to zero, and the row of I zero. Then the entry of exp(t R) from S(K) to I is the probability that the node has been reinfected by time t, and the survival function is

P(Delta_I > t) = 1 - [exp(t R)]_{S(K), I}.

For K = 1 this is the exponential exp(-lambda t) of the ordinary pair approximation, whereas for K at least 2 it is a mixture that decays quickly at first, while the node is young and likely to have infectious neighbours, and more slowly later. The mean susceptible time E[Delta_I] is the expected time to absorption from S(K); because each infectious period has unit mean, renewal gives P(I) = 1 / (1 + E[Delta_I]), which ties the two quantities together.

```python
def reinfection_survival(stage_rates: np.ndarray, ageing_rate: float, times: np.ndarray) -> dict:
    """Evaluate a node's inter-infection survival function and stationary infectious probability.

    Parameters
    ----------
    stage_rates : np.ndarray
        Infection rates lambda[x] of the node in stages S(1) to S(K), length K.
    ageing_rate : float
        Ageing rate between successive susceptible stages.
    times : np.ndarray
        Non-negative times at which to evaluate P(Delta_I > t).

    Returns
    -------
    dict
        Under the keys survival, infected_probability and mean_susceptible_time.

    Raises
    ------
    ValueError
        When the stage rates fail to be a finite non-negative one-dimensional array of length at least 1 with a positive oldest-stage rate, when the ageing rate fails to be finite and non-negative or is zero with K at least 2, or when the times fail to be a finite non-negative one-dimensional array with at least one entry.
    """
    return
```

### Step 7

step_07_memory_closure_reinfection_survival

Goal
----
Before the survival function is evaluated, steps 1 to 4 are applied once more to every tie of the chosen node as a certificate of the fixed point: the cavity rates are recomputed from the converged messages, the pair generator and its stationary distribution are rebuilt, the distribution must be annihilated by the generator, and the stage-conditional messages read off it must reproduce the converged messages to within a hundred times tol, and never to less than 1e-8, failing which the stage raises RuntimeError.

The graded quantity is P(Delta_I > t) for the chosen node under the K-stage closure. The stage also returns the ageing rate and q, the node's infection rates in its oldest and youngest stages, its stationary infectious probability and mean susceptible time, the network average of the stationary infectious probability, and, as the memoryless baseline, the same survival probability from the one-stage closure run through steps 5 and 6 with K = 1.

The survival function describes reinfection in an endemic state, so the stage refuses a configuration whose K-stage messages have collapsed to the disease-free solution: it raises ValueError when the chosen node's oldest-stage infection rate is below 1e-8.

```python
def memory_closure_reinfection_survival(
    adjacency: np.ndarray,
    beta: float,
    num_stages: int,
    node: int,
    time: float,
    tol: float,
    max_sweeps: int,
) -> dict:
    """Predict a node's probability of staying uninfected for longer than t after recovery under the K-stage memory pair closure.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric 0/1 adjacency matrix of shape (N, N) with zero diagonal and no isolated node.
    beta : float
        Per-contact transmission rate, above zero, with unit recovery rate.
    num_stages : int
        Number K of susceptible stages, at least 1.
    node : int
        Index of the chosen node.
    time : float
        Time t since recovery, non-negative.
    tol : float
        Convergence tolerance of the message iteration.
    max_sweeps : int
        Maximum number of sweeps of the message iteration.

    Returns
    -------
    dict
        Under the keys survival, ageing_rate, mean_degree_less_one, oldest_stage_rate, youngest_stage_rate, infected_probability, mean_susceptible_time, infected_fraction, standard_pair_survival and sweeps.

    Raises
    ------
    ValueError
        When the adjacency is not a simple undirected graph without isolated nodes, when beta or tol fails to be finite and above zero, when num_stages or max_sweeps fails to be an integer at least 1, when node fails to be an integer index of the graph, when time fails to be finite and non-negative, or when the closure predicts no endemic state at the chosen node.
    RuntimeError
        When the message iteration has not converged after max_sweeps sweeps, or when the converged messages fail the fixed-point certificate on the ties of the chosen node.
    """
    return
```
