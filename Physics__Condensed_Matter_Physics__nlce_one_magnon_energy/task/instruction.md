# nlce_one_magnon_energy

## Background

Symbols and conventions. A cluster $c$ is a connected piece of the lattice, here an open chain of $m$ sites. $H$ is the full Hamiltonian on that cluster and $H_0$ its unperturbed part. $S$ is the matrix whose columns are the eigenvectors of $H$ and $\Lambda$ the diagonal matrix of its eigenvalues. A target subspace is spanned by a chosen set of unperturbed basis states; for a spin chain in a field the natural choice is the set of single-spin-flip states, one per site. Indices $i,j$ label sites; $a,b$ label eigenstates; $r$ is a separation in sites.

Effective Hamiltonians. The purpose of an effective Hamiltonian is to reproduce part of the spectrum of a large Hilbert space inside a much smaller one. Formally one seeks a unitary $T$ that block-diagonalises $H$, so that $T^\dagger HT$ has no matrix elements connecting the target subspace to the rest. Any such $T$ can be written as $T=SF$ with $F$ block-diagonal and unitary, so $T$ is not unique: block diagonalisation alone does not define an effective Hamiltonian. Some further condition is needed to fix the freedom, and different conditions give genuinely different effective Hamiltonians for the same $H$.

The classical choice, due to Cederbaum, Schirmer and Meyer, is minimal deformation: among all block-diagonalising transformations, take the one closest to the identity in the Frobenius norm, $\lVert T-1\rVert_F$. It is a global condition: it treats every block on the same footing.

Linked-cluster expansions. A quantity defined on an infinite lattice can be built from exact calculations on finite clusters. Write the extensive observable per subunit as a sum over all inequivalent connected clusters, each entering with its embedding multiplicity $l_c$ (the number of ways the cluster sits in the infinite lattice per subunit) and its weight $W_c$. The weight is what is left of the cluster's own value once every smaller piece has been accounted for, defined recursively by subtracting the weights of all connected subclusters. Truncating the sum at clusters of size $n$ gives the $n$th-order estimate. The expansion is exact order by order for a quantity that is cluster additive, and for a gapped system with short-range effective couplings it converges rapidly, because the weight of a large cluster is suppressed by the correlation length.

Exact diagonalisation. For a spin-$1/2$ chain of $m$ sites the Hilbert space has dimension $2^m$ and the Hamiltonian can be built directly as a sum of Kronecker products of Pauli matrices. Numerical routines return eigenvalues in ascending order and eigenvectors as columns; the overall sign of an eigenvector is arbitrary, and within a degenerate eigenspace any orthonormal basis is as good as any other, so a construction that depends on such a choice is not well defined.

The transverse-field Ising chain. $H=-\sum_i\sigma^z_i-J\sum_i\sigma^x_i\sigma^x_{i+1}$ is the standard testbed for these methods: it is exactly solvable, so a computed excitation spectrum can be checked against a closed form, and for $0\le J<1$ it has a unique polarised ground state and a finite gap. Because the perturbation flips two neighbouring spins, it changes the number of flipped spins by $0$ or $\pm2$, so states with an even and an odd number of flips never mix.

## Problem

A magnon travelling along a transverse-field Ising chain does not hop only to its neighbour: once the spins are allowed to fluctuate, an effective hopping appears at every separation, and the one-magnon band is set by the whole sequence of amplitudes. Your task is to compute that band at one wavevector, by constructing the effective Hamiltonian numerically on finite chains and removing the boundaries with a linked-cluster expansion.

The model is the open transverse-field Ising chain of $m$ sites,

$$H=-\sum_{i=1}^{m}\sigma^z_i-J\sum_{i=1}^{m-1}\sigma^x_i\sigma^x_{i+1}$$

with $J=0.8$, where $\sigma^x_i$ and $\sigma^z_i$ are Pauli matrices. Work in the one-magnon sector: the target subspace is spanned by the $m$ single-spin-flip states, the state with site $i$ flipped being the $i$-th basis vector of that subspace.

The difficulty is that block diagonalisation does not by itself define an effective Hamiltonian. Any transformation of the form $T=SF$ with $F$ block-diagonal and unitary block-diagonalises $H$, so a further condition is needed, and different conditions give different answers. Fix the freedom with the minimality criterion introduced in recent work on numerically constructed effective Hamiltonians, and choose the eigenstates of each chain that build the effective Hamiltonian by the selection rule that work prescribes; the choice of eigenstates changes the answer at the coupling used here.

Once the eigenstates are chosen, the effective Hamiltonian is written in the single-flip basis with rows and columns ordered by site. The eigenvalues are diagonal in the basis of the chosen eigenstates, so the construction must carry them back onto the single-flip basis states; a matrix that is still indexed by eigenstates is not the effective Hamiltonian and changes with how the eigenstates are labelled.

Having obtained the effective Hamiltonian on each chain of $m$ sites, form for each separation $r$ the forward pair sum $T_{(r,m)}=\sum_i[H_{\mathrm{eff}}]_{i,i+r}$, taken over the $m-r$ pairs with $i<i+r$; for $r=0$ subtract $m$ times the chain's ground-state energy from the diagonal sum, so that excitation energies stay cluster additive. Remove the boundary effects with a numerical linked-cluster expansion in the chain length, truncated at chains of $8$ sites: the weight $W_{(r,m)}$ of a chain is its own $T_{(r,m)}$ less the weights of all its connected subclusters in the standard linked-cluster sense, and the amplitudes $t_r$ are the resulting estimates per site of the infinite chain. Finally combine the amplitudes into the one-magnon energy

$$E(k)=t_0+2\sum_{r\ge1}t_r\cos(kr)$$

at $k=2\pi/5$. Report $E(k)$ as it comes out of the order-8 expansion. The chain is exactly solvable, so the expansion is converging towards a known closed form, but the order-8 value is not that limit and the closed form is not the answer.

## What to report

Report the one-magnon energy $E(k=2\pi/5)$ from the order-8 expansion as the final answer, to at least six significant figures. In the reasoning, state the expressions you used for the minimality criterion, the transformation that satisfies it on the one-magnon block, the eigenstate-selection rule, the effective Hamiltonian and the subcluster multiplicities, then show these scalars, each to at least six significant figures, and no others: the ground state energy of the 8-site chain; the value of that criterion's deformation measure for the eigenstates selected on the 8-site chain; the nearest-neighbour forward pair sum $T_{(1,8)}$ of the 8-site chain itself, before any subcluster weights are removed; the linked-cluster weight $W_{(1,8)}$; the amplitudes $t_0$, $t_1$ and $t_3$; and the final energy.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

hamiltonian_element

Goal
----
Build the transverse-field Ising chain and report one matrix element.
Returns the element, dimensionless.

```python
def hamiltonian_element(a: int, b: int, m: int, dev: dict) -> float:
    r"""Build the transverse-field Ising chain and report one matrix element.

    Returns the element, dimensionless.

    Parameters
    ----------
    a : int
        Row index in the product basis, $0\le a<2^m$.
    b : int
        Column index in the product basis, $0\le b<2^m$.
    m : int
        Number of sites in the chain, $m\ge1$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $0\le J<1$.

    Returns
    -------
    element : float
        The matrix element $H_{ab}$.

    Raises
    ------
    ValueError
        If $a$, $b$, $m$ or $J$ is not finite, or if $a$ or $b$ lies outside
        $0\le a,b<2^m$.
    """
    return None
```

### Step 2

selection_deviation

Goal
----
Choose the eigenstates that deform the target block least and report that deformation.
Returns the Frobenius deviation, dimensionless.

```python
def selection_deviation(m: int, dev: dict) -> float:
    r"""Choose the eigenstates that deform the target block least and report that deformation.

    Returns the Frobenius deviation, dimensionless.

    Parameters
    ----------
    m : int
        Number of sites in the chain, $m\ge1$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $J\ge0$ (the selection is defined beyond the gapped range
        $0\le J<1$ used by the expansion).

    Returns
    -------
    deviation : float
        The Frobenius deviation $\lVert T_{11}-1\rVert_F$ of the chosen set.

    Raises
    ------
    ValueError
        If $m$ or $J$ is not finite, or if $m<1$.
    """
    return None
```

### Step 3

effective_element

Goal
----
Construct the effective Hamiltonian in the one-magnon sector and report one element.
Returns the element, dimensionless.

```python
def effective_element(i: int, j: int, m: int, dev: dict) -> float:
    r"""Construct the effective Hamiltonian in the one-magnon sector and report one element.

    Returns the element, dimensionless.

    Parameters
    ----------
    i : int
        Row site index, $0\le i<m$.
    j : int
        Column site index, $0\le j<m$.
    m : int
        Number of sites in the chain, $m\ge1$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $0\le J<1$.

    Returns
    -------
    element : float
        The element $[H_{\mathrm{eff}}]_{ij}$.

    Raises
    ------
    ValueError
        If $i$, $j$, $m$ or $J$ is not finite, or if $i$ or $j$ lies outside
        $0\le i,j<m$.
    """
    return None
```

### Step 4

cluster_quantity

Goal
----
Sum the effective Hamiltonian over all site pairs a fixed distance apart.
Returns the pair sum, dimensionless.

```python
def cluster_quantity(m: int, r: int, dev: dict) -> float:
    r"""Sum the effective Hamiltonian over all site pairs a fixed distance apart.

    Returns the pair sum, dimensionless.

    Parameters
    ----------
    m : int
        Number of sites in the chain.
    r : int
        Separation in sites, $r\ge0$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $0\le J<1$.

    Returns
    -------
    pair_sum : float
        The distance-$r$ pair sum $T_{(r,m)}$ on the $m$-site chain; zero
        when $r\ge m$.

    Raises
    ------
    ValueError
        If $m$ or $r$ is not finite, if $r<0$, or if $J$ is not finite while
        $r<m$.
    """
    return None
```

### Step 5

cluster_weight

Goal
----
Strip the subchain contributions from a chain's pair sum.
Returns the linked-cluster weight, dimensionless.

```python
def cluster_weight(m: int, r: int, dev: dict) -> float:
    r"""Strip the subchain contributions from a chain's pair sum.

    Returns the linked-cluster weight, dimensionless.

    Parameters
    ----------
    m : int
        Number of sites in the chain, $m\ge1$.
    r : int
        Separation in sites, $r\ge0$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $0\le J<1$.

    Returns
    -------
    weight : float
        The linked-cluster weight $W_{(r,m)}$ of the $m$-site chain at
        separation $r$.

    Raises
    ------
    ValueError
        If $m$ or $r$ is not finite, if $m<1$, if $r<0$, or if $J$ is not
        finite while $r<m$.
    """
    return None
```

### Step 6

nlce_hopping

Goal
----
Assemble the linked-cluster weights into the hopping amplitude at one separation.
Returns the amplitude, dimensionless.

```python
def nlce_hopping(r: int, dev: dict) -> float:
    r"""Assemble the linked-cluster weights into the hopping amplitude at one separation.

    Returns the amplitude, dimensionless.

    Parameters
    ----------
    r : int
        Separation in sites, $r\ge0$.
    dev : dict
        Parameter dict. Keys read here:

        - ``J``: transverse-field Ising coupling, $0\le J<1$.
        - ``n_max``: largest chain length kept in the linked-cluster
          expansion, $n_{\max}\ge1$.

    Returns
    -------
    amplitude : float
        The order-$n_{\max}$ hopping amplitude $t_r$ at separation $r$.

    Raises
    ------
    ValueError
        If $r$ or $n_{\max}$ is not finite, if $n_{\max}<1$, if $r<0$, or if
        $J$ is not finite while $r<n_{\max}$.
    """
    return None
```

### Step 7

one_magnon_energy

Goal
----
Combine the hopping amplitudes into the one-magnon energy at the reported wavevector.
Returns the energy, dimensionless.

```python
def one_magnon_energy(dev: dict) -> float:
    r"""Combine the hopping amplitudes into the one-magnon energy at the reported wavevector.

    Returns the energy, dimensionless.

    Parameters
    ----------
    dev : dict
        Parameter dict. Keys read here:

        - ``J``: transverse-field Ising coupling, $0\le J<1$.
        - ``n_max``: largest chain length kept in the linked-cluster
          expansion, $n_{\max}\ge1$.
        - ``k``: wavevector at which the one-magnon energy is reported, in rad.

    Returns
    -------
    energy : float
        The one-magnon energy $E(k)$ from the order-$n_{\max}$ expansion.

    Raises
    ------
    ValueError
        If $J$, $k$ or $n_{\max}$ is not finite, or if $n_{\max}<1$.

    Notes
    -----
    Uses ``nlce_hopping(r, dev)`` for $r=0,\ldots,n_{\max}-1$.
    """
    return None
```
