# Physics-Condensed_Matter_Physics-6

## Background

Tensor networks express many-body calculations as contractions of local objects joined by bond indices. Exact contraction can become expensive on graphs with cycles. Belief propagation provides a less costly local approximation by passing descriptions of each site's surrounding network along its bonds, but those descriptions can miss correlations carried around cycles.

Corrections built from correlated regions can account for effects absent from that local approximation. This is relevant to thermodynamic calculations in lattice models and to tensor-network contractions arising in quantum simulation and error correction. The usefulness of a finite correction depends on the size of the contributions that remain unaccounted for.

## Problem

Compute the physical free-energy density of an open $3\times4$ Ising grid using the connected-cluster expansion of tensor-network belief propagation (BP) for $\log Z$, in its generalized-loop formulation, through total cluster edge weight 10. Label sites $v=4r+c$ for $0\le r<3$ and $0\le c<4$, use the ordered spin basis $(-1,+1)$ and $\beta=1$, and take the dimensionless energy to be $E(s)=-\sum_{\{u,v\}\in E}K_{\{u,v\}}s_us_v-\sum_v h_vs_v$. Horizontal couplings $K^H_{r,c}$ join $(r,c)$ to $(r,c+1)$, vertical couplings $K^V_{r,c}$ join $(r,c)$ to $(r+1,c)$, and the site fields are:
$$
K^H=\begin{pmatrix}
0.368&0.464&0.304\\
0.400&0.336&0.496\\
0.432&0.288&0.384
\end{pmatrix},\qquad
K^V=\begin{pmatrix}
0.352&0.448&0.416&0.320\\
0.480&0.384&0.304&0.432
\end{pmatrix},\qquad
h=\begin{pmatrix}
0.07&-0.04&0.02&0.09\\
-0.05&0.06&-0.08&0.03\\
0.04&-0.06&0.05&-0.02
\end{pmatrix}.
$$

Represent this instance with bond indices $x_e\in\{0,1\}$, factors $w_e(s,0)=\sqrt{\cosh K_e}$ and $w_e(s,1)=s\sqrt{\sinh K_e}$, and local tensors $T_v((x_{vu})_{u\sim v})=\sum_{s=\pm1}e^{h_vs}\prod_{u\sim v}w_{\{v,u\}}(s,x_{vu})$, whose axes follow increasing neighbor labels. For BP, update all directed messages simultaneously by contracting each local tensor with incoming messages on every leg except the outgoing leg; start each message at $(1,0)$, normalize every update to unit Euclidean norm with positive zeroth component, use no damping, and converge until $\max_{v\to w}\|\mu_{v\to w}^{(t+1)}-\mu_{v\to w}^{(t)}\|_2<10^{-13}$. Orient each edge $e=(u,v)$ with $u<v$; when representing an edge operation as a matrix, its row index belongs to $u$ and its column index belongs to $v$, with the incoming messages at those endpoints being $\mu_{v\to u}$ and $\mu_{u\to v}$, respectively. Let $\widetilde F_{10}$ be the BP-vacuum contribution plus the connected-cluster correction to $\log Z$ at the stated weight cutoff, and report the single scalar $f_{\mathrm{phys}}^{(10)}=-\widetilde F_{10}/12$ rounded to six decimal places.

## Output format

```
Output Format Requirements:
Begin the response with exactly one finite decimal wrapped in <final_answer>...</final_answer>, followed by the scientific reasoning wrapped in <reasoning>...</reasoning>. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
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

build_ising_tensor_network

Goal
----
Construct the numerical data for an open rectangular Ising tensor network.

```python
def build_ising_tensor_network(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    '''Build topology, site tensors, and edge factors for a rectangular grid.

    Parameters
    ----------
    K_h : np.ndarray
        Finite nonnegative array with shape ``(R,C-1)`` containing horizontal
        couplings for an ``R`` by ``C`` open grid.
    K_v : np.ndarray
        Finite nonnegative array with shape ``(R-1,C)`` containing vertical
        couplings for the same grid.
    h : np.ndarray
        Finite array with shape ``(R,C)`` containing site fields.  ``R`` and
        ``C`` must both be positive.

    Returns
    -------
    edges : np.ndarray
        Integer array of shape ``(E,2)``.  Each row is ``(u,v)`` with ``u<v``;
        rows are sorted lexicographically.
    neighbors : np.ndarray
        Integer array of shape ``(R*C,4)``.  Active neighbor labels appear in
        increasing order and unused entries are ``-1``.
    degrees : np.ndarray
        Integer array of shape ``(R*C,)`` giving the number of active neighbors.
    tensors : np.ndarray
        Float array of shape ``(R*C,16)``.  For site ``v`` of degree ``d``, the
        first ``2**d`` entries are the C-order flattening of its local tensor in
        the neighbor-axis order above; remaining entries are zero.
    bond_factors : np.ndarray
        Float array of shape ``(E,2,2)``.  Axis 1 uses spin order ``(-1,+1)``
        and axis 2 uses bond-index order ``(0,1)``.

    Raises
    ------
    ValueError
        If the array dimensions are inconsistent, a required value is
        nonfinite, or a coupling is negative.
    '''
    return None
```

### Step 2

solve_bp_messages

Goal
----
Compute the directed tensor-BP fixed-point messages for a supplied grid network.

```python
def solve_bp_messages(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", tol: float,
) -> "np.ndarray":
    '''Return converged directed BP messages in the canonical edge orientation.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` returned by
        ``build_ising_tensor_network``.
    neighbors : np.ndarray
        Integer array of shape ``(N,4)`` returned by the preceding step.
    degrees : np.ndarray
        Integer array of shape ``(N,)`` returned by the preceding step.
    tensors : np.ndarray
        Float array of shape ``(N,16)`` returned by the preceding step.
    tol : float
        Finite positive threshold for the maximum Euclidean message change
        between consecutive simultaneous sweeps.

    Returns
    -------
    messages : np.ndarray
        Float array of shape ``(E,2,2)``.  For edge ``(u,v)`` with ``u<v``,
        ``messages[e,0]`` is ``u -> v`` and ``messages[e,1]`` is ``v -> u``.
        Every returned message has unit Euclidean norm and positive zeroth
        component.

    Raises
    ------
    ValueError
        If the supplied arrays are structurally inconsistent or ``tol`` is not
        finite and positive.
    RuntimeError
        If the prescribed fixed point is not reached within the internal
        safety limit.
    '''
    return None
```

### Step 3

compute_bp_vacuum

Goal
----
Compute the BP-vacuum normalization data and oriented edge decomposition arrays.

```python
def compute_bp_vacuum(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", messages: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]":
    '''Return site factors, edge overlaps, edge matrices, and the BP log term.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` in canonical orientation.
    neighbors : np.ndarray
        Integer array of shape ``(N,4)`` in increasing active-neighbor order.
    degrees : np.ndarray
        Integer array of shape ``(N,)``.
    tensors : np.ndarray
        Float array of shape ``(N,16)``.
    messages : np.ndarray
        Float array of shape ``(E,2,2)`` returned by ``solve_bp_messages``.

    Returns
    -------
    site_factors : np.ndarray
        Positive float array of shape ``(N,)``.
    overlaps : np.ndarray
        Float array of shape ``(E,)`` in edge-list order.
    projectors0 : np.ndarray
        Float array of shape ``(E,2,2)`` in the documented endpoint orientation.
    projectors_perp : np.ndarray
        Float array of shape ``(E,2,2)`` in the same orientation.
    F0 : float
        BP-vacuum contribution to ``log Z``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or a required normalization is nonpositive
        or nonfinite.
    '''
    return None
```

### Step 4

enumerate_connected_generalized_loops

Goal
----
Enumerate connected generalized-loop edge masks up to a supplied edge-weight cutoff.

```python
def enumerate_connected_generalized_loops(
    edges: "np.ndarray", max_weight: int,
) -> "np.ndarray":
    '''Return canonical masks for connected generalized loops within the cutoff.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` describing a simple undirected graph.
        Each row must satisfy ``u<v``; rows must be unique.
    max_weight : int
        Nonnegative maximum number of selected edges.

    Returns
    -------
    loops : np.ndarray
        Integer array of shape ``(L,E)`` with entries in ``{0,1}``, sorted by
        selected-edge count and then by the edge-position bit mask.  If no loop
        qualifies, the shape is ``(0,E)``.

    Raises
    ------
    ValueError
        If ``edges`` is malformed or ``max_weight`` is negative.
    '''
    return None
```

### Step 5

compute_normalized_loop_weights

Goal
----
Compute the normalized excitation weight for each supplied connected-loop mask.

```python
def compute_normalized_loop_weights(
    edges: "np.ndarray", h: "np.ndarray", bond_factors: "np.ndarray",
    projectors0: "np.ndarray", projectors_perp: "np.ndarray",
    site_factors: "np.ndarray", loops: "np.ndarray",
) -> "np.ndarray":
    '''Evaluate normalized weights for the supplied connected-loop masks.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` in canonical endpoint orientation.
    h : np.ndarray
        Finite rectangular site-field array; row-major flattening defines site
        labels used by ``edges``.
    bond_factors : np.ndarray
        Float array of shape ``(E,2,2)`` returned by the network builder.
    projectors0 : np.ndarray
        Float array of shape ``(E,2,2)`` returned by ``compute_bp_vacuum``.
    projectors_perp : np.ndarray
        Float array of shape ``(E,2,2)`` returned by ``compute_bp_vacuum``.
    site_factors : np.ndarray
        Positive float array of shape ``(h.size,)`` returned by
        ``compute_bp_vacuum``.
    loops : np.ndarray
        Integer binary array of shape ``(L,E)`` in canonical loop order.

    Returns
    -------
    loop_weights : np.ndarray
        Float array of shape ``(L,)`` preserving the input loop order.

    Raises
    ------
    ValueError
        If array shapes are inconsistent, loop entries are not binary, or the
        vacuum normalization is nonpositive or nonfinite.
    '''
    return None
```

### Step 6

enumerate_connected_loop_clusters

Goal
----
Enumerate connected loop-multiset clusters through a total edge-weight cutoff.

```python
def enumerate_connected_loop_clusters(
    edges: "np.ndarray", loops: "np.ndarray", max_weight: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    '''Return multiplicities, connected-cluster coefficients, and total weights.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` in canonical orientation.
    loops : np.ndarray
        Integer binary array of shape ``(L,E)`` containing connected loops in
        canonical order.
    max_weight : int
        Nonnegative upper bound on the sum of loop edge counts in a cluster.

    Returns
    -------
    multiplicities : np.ndarray
        Integer array of shape ``(C,L)``.  Entry ``[c,l]`` is the number of
        copies of loop ``l`` in cluster ``c``.
    ursell_coefficients : np.ndarray
        Float array of shape ``(C,)`` in cluster-row order.
    cluster_weights : np.ndarray
        Integer array of shape ``(C,)`` giving total loop-edge weight.

    Raises
    ------
    ValueError
        If shapes are inconsistent, loop masks are not binary, or
        ``max_weight`` is negative.
    '''
    return None
```

### Step 7

evaluate_cluster_correction

Goal
----
Evaluate the numerical connected-cluster contributions and their total correction.

```python
def evaluate_cluster_correction(
    loop_weights: "np.ndarray", multiplicities: "np.ndarray",
    ursell_coefficients: "np.ndarray",
) -> "tuple[np.ndarray, float]":
    '''Return per-cluster numerical terms and their total correction.

    Parameters
    ----------
    loop_weights : np.ndarray
        Finite float array of shape ``(L,)``.
    multiplicities : np.ndarray
        Nonnegative integer array of shape ``(C,L)``.
    ursell_coefficients : np.ndarray
        Finite float array of shape ``(C,)`` in the same cluster-row order.

    Returns
    -------
    cluster_contributions : np.ndarray
        Float array of shape ``(C,)`` preserving cluster-row order.
    delta : float
        Sum of all returned connected-cluster contributions.

    Raises
    ------
    ValueError
        If dimensions are inconsistent, a required value is nonfinite, or a
        multiplicity is negative.
    '''
    return None
```

### Step 8

compute_cluster_free_energy

Goal
----
Compute the finite-weight connected-cluster physical free-energy density.

```python
def compute_cluster_free_energy(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray",
    max_weight: int, bp_tol: float,
) -> float:
    '''Return the connected-cluster physical free-energy density for the grid.

    Parameters
    ----------
    K_h : np.ndarray
        Finite nonnegative array of shape ``(R,C-1)``.
    K_v : np.ndarray
        Finite nonnegative array of shape ``(R-1,C)``.
    h : np.ndarray
        Finite field array of shape ``(R,C)`` with ``R,C >= 1``.
    max_weight : int
        Nonnegative total cluster edge-weight cutoff.
    bp_tol : float
        Finite positive Euclidean stopping tolerance for the prescribed BP
        fixed point.

    Returns
    -------
    free_energy_density : float
        Physical free-energy density per spin at beta equal to one.

    Raises
    ------
    ValueError
        If the supplied grid or scalar settings violate the documented domain.
    RuntimeError
        If the prescribed BP fixed point is not reached within the internal
        safety limit.
    '''
    return None
```
