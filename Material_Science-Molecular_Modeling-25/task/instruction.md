# Multi-regime orientational audit of a tensed plane wormlike chain

## Background

A semiflexible polymer in a plane can be represented by a sequence of unit tangents whose angles are measured from a fixed axis. Thermal bending, external tension, chain-end effects, and angular discretization jointly determine the orientation distribution. Single-site distributions give extension and orientational order, while pair distributions determine local coupling and tangent correlations.

For an open nearest-neighbour chain, forward and backward transfer messages sum over configurations without enumerating them. The same messages give normalized one-site marginals and neighbouring pair probabilities. A connected combination of pair probabilities can remove contributions from the rest of the chain, allowing a local interaction to be inferred independently of chain-end effects.

In a planar chain, orientational correlations, extension and nematic alignment probe different aspects of the same distribution. Comparing them across changes in flexibility and loading helps distinguish a change in local bending response from a change caused only by the applied field.

## Problem

A semiflexible polymer lies in a plane as an open chain of equal segments under axial tension. Use the conventions of the published source on which this task is based (not attached) for the two-dimensional discrete wormlike chain's angular grid, persistence length, periodic bending, transfer marginals, connected bond closure, tangent correlation, and planar nematic order. For this instance the reduced energy of a bend through its periodic minimum-image angle $\Delta$ is $\beta\kappa\Delta^2(1+c\Delta^2)/(2a)$, where $c$ has units of inverse squared radians; this anharmonic factor is an instance choice, not a formula from that source. The canonical state is $(l_p/a,c,f,N_s,n)=(2.2,0.2,0.8,32,96)$, with reduced tension $f=\beta Fa$.

Compute the canonical record $(B_\pi,P,K_h,l_1,l_h,X,R,Q)$: antiparallel bond energy, mid-chain peak angular probability times $n$, the connected closure exponent at its $(h,h)$ corner, first and last apparent persistence values, reduced extension, root-mean-square end-to-end distance divided by the contour length, and mid-chain nematic order, where $h=n/4$. For the closure block use signed indices $-h,\ldots,h$ and take $K_h$ at signed indices $(h,h)$; obtain the apparent-persistence profile from index pairs $(k,-k)$ for $k=1,\ldots,h$, so $l_1$ and $l_h$ are its endpoints.

Next evaluate 27 regimes in lexicographic order over $l_p/a\in\{0.5l_p/a,l_p/a,2l_p/a\}$, $c\in\{0,c,2c+0.1\}$ and $f\in\{f-0.8,f,f+0.8\}$; if the supplied $c$ is zero, use $(0,0.1,0.3)$ for that axis. For each regime form the six-column response matrix $\mathcal R$ with row $(X,Q,P,K_h,l_1,l_h)$. Divide each column by its uncentered root-mean-square over all 27 rows, then subtract the scaled column mean; let $s_1\ge\cdots\ge s_6$ be the singular values of this centered matrix and $H=-\sum_i p_i\log p_i/\log 6$, where $p_i=s_i^2/\sum_j s_j^2$. Reshape the scaled but uncentered response to $3\times3\times3\times6$, contract its first three axes with $q=(1,-2,1)/\sqrt6$, and call the norm of the resulting six-vector $M$. Define $W=\sum_{r=1}^{27}\sum_{j=1}^{6}(r+j/10)\mathcal R_{rj}$ using the unscaled response matrix and one-based $r,j$, and define $D$ as the 0.9 minus 0.1 quantile of $l_h-l_1$ across regimes using NumPy's default linear rule.

The final scalar is $J=(H+M+|D|/(1+|D|))/(3+|Q|)$, with $Q$ from the canonical record. Report the canonical $X,R,Q$ and audit values $W,s_1,s_6,H,M,D$ as supporting results, and give $J$ to eight significant figures.

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

dwlc_bond_energy

Goal
----
Builds the reduced bending energy of one anharmonic bond on the periodic grid of directions.

```python
def dwlc_bond_energy(lp_over_a: float, c: float, n: int) -> "np.ndarray":
    r"""lp_over_a: positive float, the persistence length in units of the segment length. c:
    non-negative float, the anharmonicity, in inverse squared radians. n: integer at least 8 and
    divisible by four, the number of directions.

    A plane semiflexible chain is cut into segments of length $a$ whose orientations are
    restricted to n equally spaced directions covering the full circle, indexed $r = 0,\dots,n-1$
    in increasing angle from a fixed axis. The bending energy of one bond is the continuum
    wormlike-chain bending energy evaluated on the discretization, that is $\beta\kappa/(2a)$
    times the square of the bend angle between the two orientations, multiplied by the
    anharmonic factor $1 + c\,\Delta^2$ in the same bend angle $\Delta$. At $c = 0$ the bond is
    the source's harmonic one.

    Use the planar continuum convention $l_p=2\beta\kappa$ to set the bending stiffness.
    The finite-grid discrete correlation need not equal $\exp(-|i-j|a/l_p)$ exactly.

    Returns a numpy float64 array of shape $(n, n)$: the reduced bending energy for every ordered
    pair of directions, row index r and column index s. It is symmetric and vanishes on the
    diagonal.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, or an n that is
            not an integer, is below eight, or is not divisible by four.
    """
    return None
```

### Step 2

dwlc_marginals

Goal
----
Computes the single-segment angular densities of the open chain.

```python
def dwlc_marginals(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    r"""lp_over_a, c, n: as in the bond energy step. f: float, the reduced tension, the product of
    the applied force and the segment length in units of the thermal energy; a positive f favours
    alignment with the axis. Ns: integer at least 2, the number of segments.

    The chain is open with free ends. The reduced tension acts on each segment, while the bond
    energy from the preceding step acts between neighboring segments. Compute the angular
    marginal of each segment under this model. Each row sums to one.

    Returns a numpy float64 array of shape $(N_s, n)$: one normalized angular density per segment,
    segment index first, direction index second.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n, a
            non-integer or smaller than 2 value of Ns, or a partition function that overflows.
    """
    return None
```

### Step 3

dwlc_extension

Goal
----
Evaluates the per segment projections on the force direction and the extension reduced by the contour length.

```python
def dwlc_extension(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step.

    Compute the axial projection of each segment under its angular marginal and the resulting
    chain extension reduced by the contour length.

    Returns a numpy float64 array of shape $(N_s + 1,)$: the per segment projections in segment
    order in the first Ns entries, and the reduced extension in the last entry.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n,
            or a non-integer or smaller than 2 value of Ns.
    """
    return None
```

### Step 4

dwlc_closure_exponent

Goal
----
Measures the connected part of the neighbouring pair distribution on the central block of directions.

```python
def dwlc_closure_exponent(lp_over_a: float, c: float, f: float, Ns: int, n: int, bond: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step. bond: integer in $0 \le \mathrm{bond}
    \le N_s - 2$, the bond joining segments bond and bond+1.

    Let $C^{rs}$ be the joint probability that segment bond points along direction r and segment
    bond+1 along direction s. Compute the source's exact bond-local connected closure exponent
    from this neighboring-pair distribution, using direction 0 as the reference state. The
    result must remain numerically finite for the tested stiff-chain cases.

    Return it on the central block of signed indices $|r_\sigma| \le n/4$ and
    $|s_\sigma| \le n/4$, where the signed index $r_\sigma$ runs from $-n/2$ to $n/2 - 1$ and the
    bend angle of a pair is unambiguous.

    Returns a numpy float64 array of shape $(m, m)$ with $m = 2\lfloor n/4\rfloor + 1$, rows and
    columns ordered by increasing signed index. It is symmetric and vanishes on the row and the
    column of the reference state.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n, a
            non-integer or smaller than 2 value of Ns, or a bond index outside its range.
    """
    return None
```

### Step 5

dwlc_apparent_persistence

Goal
----
Reads the measured connected combination backwards to give the persistence length a harmonic reading would assign at each bend angle.

```python
def dwlc_apparent_persistence(lp_over_a: float, c: float, f: float, Ns: int, n: int, bond: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n, bond: as in the closure exponent step.

    Infer the apparent persistence length that a harmonic interpretation would assign to the
    source's connected closure at the symmetric signed-index pairs $(k, -k)$ for
    $k = 1, \dots, \lfloor n/4 \rfloor$. 

    At $c = 0$ every entry must return lp_over_a exactly, since the bond really is harmonic there.
    For $c > 0$ the entries vary with k, and that variation is what the harmonic reading misses.

    Returns a numpy float64 array of shape $(\lfloor n/4 \rfloor,)$: the apparent persistence
    length in units of the segment length, for k ascending from one.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n, a
            non-integer or smaller than 2 value of Ns, or a bond index outside its range.
    """
    return None
```

### Step 6

dwlc_tangent_correlation

Goal
----
Computes the tangent correlation between a reference segment and every other segment.

```python
def dwlc_tangent_correlation(lp_over_a: float, c: float, f: float, Ns: int, n: int, i0: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step. i0: integer in $0 \le i_0 \le N_s - 1$,
    the reference segment.

    The tangent correlation is the mean cosine of the angle between the reference segment and
    each segment j, taken in their joint distribution. It equals one at $j = i_0$ and is bounded
    by one in modulus. 

    Returns a numpy float64 array of shape $(N_s,)$: the correlation for every segment j in
    segment order.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n, a
            non-integer or smaller than 2 value of Ns, or an i0 outside its range.
    """
    return None
```

### Step 7

dwlc_nematic_profile

Goal
----
Evaluates the plane nematic order parameter of every segment from its angular density.

```python
def dwlc_nematic_profile(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step.

    Compute the source's planar nematic order parameter from each segment's angular density.
    It vanishes for a uniform density and equals one for a segment locked along the axis.

    Returns a numpy float64 array of shape $(N_s,)$: the order parameter of every segment in
    segment order.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n,
            or a non-integer or smaller than 2 value of Ns.
    """
    return None
```

### Step 8

dwlc_audit

Goal
----
Builds a 27-regime physical response tensor and reduces it to a source-dependent robustness audit.

```python
import numpy as np

def dwlc_audit(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step.
    Build the canonical eight-entry physical record from all seven earlier public functions. Let
    im = Ns//2 and h = n//4. The record is (B_pi, P, K_h, l_1, l_h, X, R, Q): antiparallel bond
    energy, mid-chain peak density times n, closure exponent at the (h,h) corner with signed
    indices r=s=h, endpoint apparent persistence values, reduced extension, RMS end-to-end
    distance divided by the contour length Ns*a, and mid-chain nematic order.

    Next evaluate 27 regimes in lexicographic order over lp_over_a*(0.5,1,2), c-axis
    (0,c,2*c+0.1), and f-axis (f-0.8,f,f+0.8). If c is zero, use (0,0.1,0.3). For each regime
    form a response row (X,Q,P,K_h,l_1,l_h). Scale each column by sqrt(mean(column**2)) and center
    the scaled columns. Let s be the descending singular values and
    H = -sum(p*log(p))/log(6), p=s**2/sum(s**2). Reshape the scaled uncentered response to
    (3,3,3,6), contract the first three axes with q=(1,-2,1)/sqrt(6), and call its Euclidean norm M.
    Let W=sum((r+j/10)*response[r-1,j-1]) for one-based r=1..27 and j=1..6. Let D be the NumPy
    default-linear 0.9 quantile minus 0.1 quantile of l_h-l_1. Define
    J=(H+M+abs(D)/(1+abs(D)))/(3+abs(Q)).

    Returns a numpy float64 array of shape (15,) containing the canonical record followed by
    (W,s[0],s[-1],H,M,D,J).
    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, invalid n,
        fewer than four segments, or a non-finite or degenerate audit reduction.
    """
    return None
```
