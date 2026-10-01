# Physics-Condensed_Matter_Physics-2

## Background

Quantum magnetism studies collective behavior arising from interacting localized spins. Anisotropy, frustration and quantum fluctuations can make even the qualitative response difficult to infer from classical spin configurations. Imaginary-time correlation functions connect equilibrium fluctuations to linear response and are important building blocks for understanding magnetic phases.

Fermionic representations of spins make diagrammatic many-body techniques available, but their enlarged state spaces, anticommuting fields and normalization factors must be treated consistently. Perturbation theory around a solvable reference can capture correlations absent from the reference itself. Comparing finite-order results with exact finite-system calculations helps distinguish approximation error from errors in the representation or numerical implementation.

## Problem

Consider a finite XYZ spin system represented by a shifted, color-preserving, three-Majorana expansion about a Gaussian reference. Quantify cancellation among its individual connected third-order spin-response diagrams.

Use four sites numbered 0 through 3, $\hbar=1$, $S^a=\sigma^a/2$, and $H_{\rm spin}=\sum_{a\in\{x,y,z\}}\sum_{j<k}J^a_{jk}S_j^aS_k^a$. The only nonzero unordered bonds are $J^x_{01}=0.9$, $J^y_{12}=1.1$, and $J^z_{23}=0.7$.

Let $\{\rho_j^a,\rho_k^b\}=2\delta_{jk}\delta_{ab}$ and $H_0=(i/2)\sum_{a,j,k}A^a_{jk}\rho_j^a\rho_k^a$, with real skew-symmetric $A$, zero scalar offset, and only the upper-triangular entries $A^x_{01}=0.23$, $A^y_{02}=-0.31$, and $A^z_{03}=0.19$ nonzero. Hold this supplied reference fixed, without fitting a self-consistency condition, set the copy-sector Hamiltonian to zero, and define $V=H_{\rm Maj}-H_0$ and $H(\xi)=H_0+\xi V$. Here $H_{\rm Maj}$ is the three-Majorana representation of the stated physical Hamiltonian.

At inverse temperature $\beta=2.4$ and imaginary time $\tau=0.83$, in reciprocal units of the quoted energies, define
$$\chi(\tau;\xi)=\frac{\operatorname{Tr}_{\rm Maj}[e^{-(\beta-\tau)H(\xi)}S_0^z e^{-\tau H(\xi)}S_0^z]}{\operatorname{Tr}_{\rm Maj}e^{-\beta H(\xi)}}=\sum_{n\geq0}\chi_n\xi^n,$$
using the full thermal trace and Taylor coefficients rather than raw derivatives.

Fix the diagram basis as follows. Expand each nonzero unordered spin bond into one quartic Majorana monomial, with the lower-numbered site's spin first and cyclic color order inside each spin. Expand each nonzero upper-triangular reference entry into one quadratic monomial of $-H_0$, with the lower-numbered site first. Keep these six residual monomials as separate vertex types, without normal-ordering or regrouping them. Each insertion independently chooses one vertex type. The two external spin operators are separate blocks, also in cyclic Majorana order. A diagram is an individual perfect Wick pairing of the fields in these blocks, with ordinary normalized-$H_0$ thermal contractions. It retains all monomial and spin prefactors, the Dyson sign, and the fermionic pairing sign. Do not combine pairings, symmetry-related vertex choices, or counterterm and quartic contributions before taking a magnitude.

For fixed insertion times $\beta\geq t_1\geq\cdots\geq t_n\geq0$, call a pairing connected when the graph whose vertices are the $n$ interaction blocks and the two external spin blocks is connected; a contraction supplies an edge between its endpoint blocks. Internal contractions are permitted. Let $w_{\lambda,P}(\mathbf t)$ denote the complete signed weight of vertex choice $\lambda$ and pairing $P$, including $(-1)^n$. Define
$$C_{{\rm abs},n}=\int_{\beta\geq t_1\geq\cdots\geq t_n\geq0}d^nt\;\sum_{\lambda}\sum_{P\ {\rm connected}}|w_{\lambda,P}(\mathbf t)|.$$
There is no additional factorial on this ordered simplex. This is a sum of magnitudes of individual diagrams before summation and integration, not the magnitude of the net connected coefficient. Equivalent relabelings that preserve the separate weighted diagrams are acceptable.

Determine the single signed average-phase percentage
$$\mathcal S_3=100\,\frac{\operatorname{Re}\chi_3}{C_{{\rm abs},3}}$$
to absolute accuracy $10^{-4}$ percentage points. For this real-weight instance it is the average sign of the connected third-order diagrams. This finite-instance cancellation diagnostic is an application of the diagrammatic framework, not a quoted numerical example from a paper.

In short reasoning, justify the copy-space treatment, counterterms, colored fermionic signs, and removal of vacuum components from individual absolute weights. Report the coefficient of $\rho_0^y\rho_0^z\rho_1^y\rho_1^z$ for the $J^x_{01}$ bond and the residual coefficient of $\rho_0^x\rho_1^x$, with equivalent reordered expressions acceptable. $\rho_0^y\rho_0^z\rho_1^y\rho_1^z$ for the $J^x_{01}$ bond, and $\rho_0^x\rho_1^x$ are graded to an absolute tolerance of $10^{-10}$.
Also report $\chi_3$, $C_{{\rm abs},2}$, $C_{{\rm abs},3}$, and the disconnected absolute weight $X_{{\rm abs},3}-C_{{\rm abs},3}$. Here $X_{{\rm abs},n}$ is the same integral of individual pairing magnitudes with both external spins and with all pairings retained, including vacuum components; $Z_{{\rm abs},n}$ is its vacuum counterpart with no external spins, normalized to the same Gaussian reference, with $Z_{{\rm abs},0}=1$. Report $X_{{\rm abs},n}$ and $Z_{{\rm abs},n}$ for $n=1,2,3$. Numerical integration and equivalent exact algorithms are acceptable if they evaluate this specified diagram-weight measure to the requested accuracy. The above reported values from $\chi_3$ onward are graded to an absolute tolerance of $10^{-6}$.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words), reporting the requested diagnostic scalars and the arguments needed to determine the final number.
Do not paste the input matrices, per-diagram weights, or quadrature-node tables.

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

map_shifted_vertices

Goal
----
Map an XYZ spin interaction and its quadratic counterterm.

```python
from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def map_shifted_vertices(
    couplings: np.ndarray, hopping: np.ndarray
) -> np.ndarray:
    """Encode V=H_Maj-H0 as ordered even Majorana monomials.

    Parameters
    ----------
    couplings : ndarray, shape (3,L,L)
        Finite real symmetric zero-diagonal XYZ couplings, L even and >=2.
        H_spin=sum_{a=0}^2 sum_{j<k} couplings[a,j,k] S_j^a S_k^a.
        Color order is x,y,z; hbar=1 and S=Pauli/2.
    hopping : ndarray, shape (3,L,L)
        Finite real skew-symmetric tensor, the same shape as couplings.
        H0=(i/2) sum_{a,j,k} hopping[a,j,k] rho_{aL+j} rho_{aL+k}.
        There is no scalar offset. The supplied reference is held fixed;
        it is not assumed to solve a self-consistency equation.

    Returns
    -------
    vertices : complex ndarray, shape (V,6)
        Each row is [coefficient, degree, i1, i2, i3, i4] and means
        coefficient*rho_i1*...*rho_i_degree in precisely that order.
        Majoranas satisfy {rho_p,rho_q}=2 delta_pq and are color-major.
        Use S_j^a=-(i/2) rho_{bL+j} rho_{cL+j}, with cyclic (a,b,c).
        First emit the nonzero spin bonds in increasing (a,j,k), j<k,
        retaining the spin-product field order (b,j),(c,j),(b,k),(c,k).
        Then emit the nonzero counterterm bonds in the same index order.
        They have degree two; unused indices are -1. Retain every nonzero
        coupling separately; an empty table has shape (0,6).

    Raises
    ------
    ValueError
        For nonfinite, nonreal, incorrectly shaped or incorrectly symmetric
        inputs, nonzero diagonals, odd L, L<2, or unequal tensor shapes.
    """
    return 0.0
```

### Step 2

thermal_majorana_kernel

Goal
----
Compute equilibrium Majorana contractions at arbitrary imaginary times.

```python
from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def thermal_majorana_kernel(
    hopping: np.ndarray, beta: float, times: np.ndarray
) -> np.ndarray:
    """Compute the color-preserving free thermal two-point kernel.

    Parameters
    ----------
    hopping : ndarray, shape (3,L,L)
        Finite real skew-symmetric zero-diagonal matrices, L even and >=2.
        H0=(i/2) sum_{a,j,k} hopping[a,j,k] rho_{aL+j} rho_{aL+k},
        with {rho_p,rho_q}=2 delta_pq and color-major indexing.
    beta : float
        Finite positive inverse temperature.
    times : ndarray, shape (T,)
        Nonempty finite real vector in [0,beta]. Unsorted or repeated
        times are allowed. Imaginary-time evolution is exp(t H0).

    Returns
    -------
    kernel : complex ndarray, shape (T,T,3L,3L)
        For positive times[u]-times[v], the entry [u,v,p,q] is
        <rho_p(times[u]) rho_q(times[v])>_0. For a negative difference
        it is -<rho_q(times[v]) rho_p(times[u])>_0. At equal times it
        is the ordered contraction <rho_p rho_q>_0, including diagonal
        value one. This kernel has no extra factor -i. Cross-color
        entries vanish. Include the full Gaussian thermal ensemble.

    Raises
    ------
    ValueError
        For invalid tensor shape/symmetry, L, nonfinite or nonreal inputs,
        beta<=0, or empty, nonvector or out-of-interval times.
    """
    return 0.0
```

### Step 3

wick_contraction_matrices

Goal
----
Assemble the ordered contractions of Majorana strings.

```python
from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def wick_contraction_matrices(
    kernel: np.ndarray, indices: np.ndarray, slots: np.ndarray
) -> np.ndarray:
    """Build the skew matrices whose Pfaffians are Gaussian string averages.

    Parameters
    ----------
    kernel : ndarray, shape (T,T,D,D)
        Finite complex time-ordered thermal contraction tensor, T,D>0.
        Equal-time entries describe the ordered product and include the
        diagonal contraction one, as in thermal_majorana_kernel.
    indices : ndarray, shape (B,2m)
        Finite integer-valued Majorana indices in [0,D), one operator
        string per row. B or m may be zero. Repeated fields are allowed.
    slots : ndarray, shape (2m,)
        Finite integer-valued indices in [0,T), specifying the kernel
        time slot for each string position; shared across batch rows.

    Returns
    -------
    matrices : complex ndarray, shape (B,2m,2m)
        Zero diagonal. For u<v, entry [r,u,v] equals
        kernel[slots[u],slots[v],indices[r,u],indices[r,v]]. The lower
        triangle is its negative transpose. Preserve the input string
        order; it determines the signs even when times coincide.

    Raises
    ------
    ValueError
        For nonfinite data, incompatible shapes, odd string length,
        noninteger indices/slots, or indices/slots outside their ranges.
    """
    return 0.0
```

### Step 4

pfaffian_batch

Goal
----
Evaluate signed complex Gaussian contraction weights.

```python
from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def pfaffian_batch(matrices: np.ndarray) -> np.ndarray:
    """Return Pfaffians with their signs and complex phases.

    Parameters
    ----------
    matrices : ndarray, shape (B,2m,2m)
        Finite complex skew-symmetric matrices. The batch may be empty,
        m may be zero, and individual matrices may be singular or have
        a zero leading pivot. Inputs are not mutated.

    Returns
    -------
    weights : complex ndarray, shape (B,)
        Pfaffians in input order, with Pf([[0,a],[-a,0]])=a and
        Pf of the 0-by-0 matrix equal to one. A singular matrix has
        Pfaffian zero. The transpose in skew symmetry has no conjugation.

    Raises
    ------
    ValueError
        For nonfinite data, incorrect shape, odd matrix dimension,
        or matrices that are not skew-symmetric.
    """
    return 0.0
```

### Step 5

absolute_wick_sums

Goal
----
Compute the total magnitude of individual perfect-pairing products for a batch of complex skew matrices, including empty and singular cases.

```python
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def absolute_wick_sums(matrices: np.ndarray) -> np.ndarray:
    """Sum magnitudes of individual perfect-pairing products.

    Parameters
    ----------
    matrices : complex ndarray, shape (B, 2m, 2m)
        Finite skew-symmetric matrices (transpose, not adjoint), with
        zero diagonal, within absolute tolerance 1e-12. B or m may be
        zero. Repeated fields and singular matrices are permitted.

    Returns
    -------
    sums : real ndarray, shape (B,)
        For each matrix K, sum over all perfect matchings P of
        product(abs(K[u, v]) for (u, v) in P), with u < v.
        The empty matching has weight one. Take absolute values before
        summing matchings. Inputs are unchanged.

    Raises
    ------
    ValueError
        For nonfinite data, incorrect shape, odd dimension, nonzero
        diagonal or failure of skew symmetry at the stated tolerance.
    """
    return 0.0
```

### Step 6

shifted_wick_weights

Goal
----
Evaluate both numerator and vacuum insertion weights, with signed Wick sums and sums of individual magnitudes kept separate. Support single time tuples and batches while retaining all residual vertices.

```python
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def shifted_wick_weights(
    hopping: np.ndarray,
    beta: float,
    vertices: np.ndarray,
    vertex_times: np.ndarray,
    sites: np.ndarray,
    component: int,
    tau: float,
) -> np.ndarray:
    """Evaluate signed and pairing-absolute insertion weights.

    Parameters
    ----------
    hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal skew reference with even L >= 2, as in
        thermal_majorana_kernel; the reference is held fixed.
    beta : float
        Finite positive inverse temperature.
    vertices : complex ndarray, shape (V, 6)
        Rows [coefficient, degree, i1, i2, i3, i4] as in
        map_shifted_vertices. Degrees are 2 or 4, active indices are
        integers in [0, 3L), and unused indices are -1. Empty tables
        and repeated active indices are allowed. Retain distinct rows.
    vertex_times : real ndarray, shape (n,) or (B, n)
        Finite times in [0, beta], 0 <= n <= 3. The second form batches
        independent time tuples, and B may be zero. Unsorted times and
        ties are permitted. Preserve field order inside every even
        block and initial block order for ties.
    sites : ndarray, shape (2,)
        Integer-valued sites j, l in [0, L).
    component : int
        Spin color a in {0, 1, 2}, with cyclic partners b, c.
    tau : float
        Finite external time in [0, beta].

    Returns
    -------
    weights : complex ndarray, shape (4,) or (B, 4)
        Columns are signed numerator, signed vacuum, unsigned numerator,
        unsigned vacuum. Sum every ordered choice of n vertex rows.
        The numerator inserts S_j^a(tau) and S_l^a(0) after the vertex
        blocks; S_j^a = -(i/2) rho_j^b rho_j^c. Signed weights use the
        normalized H0 Wick average. Unsigned weights sum magnitudes of
        each separate vertex-choice/perfect-pairing contribution, with
        spin prefactor magnitude 1/4. Absolute values precede all sums.
        All contractions, including vacuum components, are retained.
        No Dyson sign, integration, factorial or series division is
        included. At n=0 both vacuum weights are one. Inputs unchanged.

    Raises
    ------
    ValueError
        For invalid tensor, temperature, vertex table, times, sites,
        component or tau, including nonfinite/out-of-range values.
    """
    return 0.0
```

### Step 7

ordered_time_rule

Goal
----
Build the prescribed ordered-simplex Gauss rule by folding times into half a thermal period, resolving both external-time boundaries and pairwise half-period crossings. Return ordered nodes and weights.

```python
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def ordered_time_rule(
    beta: float, tau: float, order: int, quadrature: int
) -> np.ndarray:
    """Construct a folded-half-period ordered-simplex Gauss rule.

    Parameters
    ----------
    beta : float
        Finite positive interval length.
    tau : float
        Finite external time in [0, beta].
    order : int
        Number n of insertion times, 0 <= n <= 3.
    quadrature : int
        Gauss-Legendre nodes per variable, 2 <= q <= 24.

    Returns
    -------
    rule : real ndarray, shape (R, n+1)
        Rows [t1, ..., tn, weight], with descending physical times in
        [0, beta]. For n=0 return [[1.0]]. Otherwise set h=beta/2 and
        f=tau modulo h. Enumerate bit tuples in {0,1}**n lexicographically.
        For each tuple enumerate k=0,...,n folded variables above f,
        skipping zero-volume regions. Enumerate the ascending unit-
        interval Gauss nodes in lexicographic tensor-product order.
        Generate descending folded variables u on [f,h] for the first
        k positions and [0,f] for the rest. In each block start at its
        upper bound, map x to lower+(upper-lower)*x, multiply the weight
        by (upper-lower) times its unit Gauss weight, and replace upper
        with the new point. Set physical t to the descending sort of
        u+h*bits. No extra factorial. Total weight is beta**n/n!.
        These panels resolve pairwise half-period crossings and the
        external time. They do not guarantee exact integration for
        arbitrary hopping; convergence remains a numerical question.

    Raises
    ------
    ValueError
        For nonfinite beta/tau, nonpositive beta, tau outside [0,beta],
        or noninteger/out-of-range order or quadrature.
    """
    return 0.0
```

### Step 8

connected_coefficient_series

Goal
----
Integrate the shifted insertion weights at every order through p, then remove vacuum components from both the signed and positive diagram-weight series by formal power-series division.

```python
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def connected_coefficient_series(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> np.ndarray:
    """Integrate signed and pairing-absolute connected coefficients.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2, as in map_shifted_vertices.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0, beta].
    sites : ndarray, shape (2,)
        Integer-valued sites j,l in [0,L).
    component : int
        Color a in {0,1,2} of both external spins.
    order : int
        Highest insertion order p in {0,1,2,3}.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    series : complex ndarray, shape (6, p+1)
        Rows [X, Z, C, X_abs, Z_abs, C_abs]. X_n and Z_n are
        normalized-H0 numerator/vacuum Dyson coefficients of xi**n
        for H(xi)=H0+xi*(H_Maj-H0). C=X/Z as a formal power series.
        X_abs,n and Z_abs,n integrate individual pairing magnitudes
        before any vertex/pairing sums; use the exact row decomposition
        from map_shifted_vertices and no Dyson minus sign. C_abs is
        the formal quotient X_abs/Z_abs. This removes vacuum components
        while preserving magnitudes of separate connected diagrams.
        A component must include both external spin blocks because
        each interaction vertex has even population of every color.
        Z_0=Z_abs,0=1. Apply the supplied quadrature to every order,
        including lower orders entering the quotient. No projection or
        reference refitting is applied. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid tensor, temperature, observable, order or quadrature
        under the contracts above and of the producing functions.
    """
    return 0.0
```

### Step 9

connected_average_sign

Goal
----
Compose the earlier steps to return the real average phase of connected diagrams at the selected order, expressed as a percentage.

```python
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def connected_average_sign(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> float:
    """Return the real average phase of connected order-p diagrams.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2. Use map_shifted_vertices without
        regrouping its monomial rows or changing the reference.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0,beta].
    sites : ndarray, shape (2,)
        Integer-valued sites in [0,L) for the two external spins.
    component : int
        Spin color in {0,1,2}.
    order : int
        Selected insertion order p in {0,1,2,3}; not a truncated sum.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    percentage : float
        100*Re(C_p)/C_abs,p from connected_coefficient_series with the
        supplied quadrature. C_abs,p sums magnitudes of individual
        connected pairings before summation and integration. For real
        diagram weights this is the average sign in percent; otherwise
        it is the real average phase in percent. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid inputs under connected_coefficient_series, a
        nonfinite coefficient/result, or C_abs,p <= 1e-12.
    """
    return 0.0
```
