# Physics-Condensed_Matter_Physics-25

## Background

Energies are measured in t and coordinates in a. Label vertices j=5q+p for p,q=0,...,4, and set
\[
 r_j=(p-2+0.08\sin(2q+p),\ q-2+0.06\cos(p-3q)),\quad
 u_j=-0.6+0.09\cos j,\quad v_j=0.07\sin(1.3j),\quad g_j=1+0.14\cos(0.7j).
\]
Edges connect horizontal and vertical nearest neighbours of the underlying 5 by 5 grid, with open boundaries. Each listed bond is an unordered physical edge. Each vertex has two orbitals, ordered vertex first, then orbital. For the directed matrix block j,k let theta_jk=atan2(y_k-y_j,x_k-x_j), and let
\[
 A_{jj}=u_j\sigma_z+v_j I_2,\qquad A_{jk}=0.5\sigma_z+0.25i(\sigma_x\cos\theta_{jk}+\sigma_y\sin\theta_{jk}),\quad A_{kj}=A_{jk}^{\dagger}.
\]
The layer-first Hamiltonian is H(c)=[[A,cG],[cG^dagger,A*]], with G=diag(g_j) tensor sigma_y; star is entrywise conjugation. Its fermionic time reversal is J K, with J=[[0,I_50],[-I_50,0]]. Position operators repeat each vertex coordinate over both orbitals and both layers. The probe is (x,y,E)=(0.13,-0.17,0.05), kappa=0.08 t/a, and the Hermitian localizer uses material-first tensor order:
\[
 L(c)=(H(c)-EI)\otimes\sigma_z+\kappa(X-xI)\otimes\sigma_x+\kappa(Y-yI)\otimes\sigma_y.
\]
The atomic reference replaces A by (|E|+1)I_50 and G by zero at the same probe and kappa. The Pauli matrices have their usual right-handed convention. All trigonometric arguments are radians and all given decimals are exact benchmark inputs. The attached method's symmetry reduction, Pfaffian orientation and local-gap definition determine xi and mu. A diagnostic density-of-states plot uses Lorentzian broadening 0.03t. The finite lattice texture and coupling integral are constructed benchmark instances of the paper's method. Retain full precision until reporting.

## Problem

A finite, distorted two-layer insulator is subjected to a spatially nonuniform interlayer coupling c, in units of t, over 0 <= c <= 0.6. Determine its dimensionless integrated local topological protection R = integral_0^0.6 ((1-xi(c))/2) mu(c) dc at the position-energy probe specified below, using the parity-sensitive real-space method based on a skew-symmetric factorization. Here xi is normalized to +1 in the commuting atomic limit and -1 in the nontrivial phase, and mu is the local protection energy in units of t; isolated singular points contribute zero measure. Use the supplied open lattice, spatial texture and full layer mixing. Report R as the tagged scalar, with absolute error at most 2e-6. In the short reasoning, report every distinct interior transition coupling in increasing order, the total coupling width W of the nontrivial intervals, mu(0), and the contribution R_last of the last nontrivial interval, each within 2e-6 in these dimensionless units. State the normalization relative to the atomic reference and report xi on each open interval separated by these transitions. Explain the symmetry and orientation choice that fixes the parity, the physical perturbations controlled by mu, and the role of conventional spectral gaps. Explain qualitatively how the source’s auxiliary-probe prescription calibrates a residual overall Pfaffian-sign convention and how the specified atomic reference serves that purpose. The final coding subproblem must invoke and combine all preceding public subproblem functions.

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

01_bhz_layer.py

Goal
----
Assemble the single-layer angular tight-binding operator.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def bhz_layer(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, t1: float, delta: float) -> np.ndarray:
    'Assemble the single-layer angular tight-binding operator.\n\nParameters\n----------\ncoords : real (N,2), distinct vertex coordinates in a.\nedges : integer (B,2), unique undirected non-self edges, in any orientation.\nmass, potential : real scalars or (N,), on-site mass and scalar potential in t.\nt1, delta : real scalars, bond coefficients in t.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. N is positive; coordinates are distinct. Edge indices are in [0,N), with no self-edges or repeated undirected edges. Empty edges have shape (0,2). On-site arrays are either scalars or exactly (N,).\n\nReturns\n-------\ncomplex ndarray (2N,2N), Hermitian A in vertex-major, orbital-minor order; units t.'
    return result
```

### Step 2

02_aii_pencil.py

Goal
----
Reduce the class-AII localizer to an affine real skew-symmetric pencil.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def aii_pencil(layer: ArrayLike, coords: ArrayLike, profile: ArrayLike, probe: ArrayLike, kappa: float) -> np.ndarray:
    'Reduce the class-AII localizer to an affine real skew-symmetric pencil.\n\nParameters\n----------\nlayer : complex (2N,2N), arbitrary Hermitian single-layer matrix in t.\ncoords : real (N,2), vertex coordinates in a.\nprofile : real (N,), dimensionless interlayer texture.\nprobe : real (3,), ordered x/a,y/a,E/t.\nkappa : nonnegative float in t/a.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. N is positive. The layer is Hermitian within relative max-entry tolerance 1e-12. Coordinates have exactly shape (N,2).\n\nReturns\n-------\nreal ndarray (2,8N,8N), ordered S0,S1 with S(c)=S0+c*S1; units t. Roundoff-only symmetric and imaginary components are removed.'
    return result
```

### Step 3

03_paired_ordering.py

Goal
----
Construct a matching-preserving, fill-aware initial permutation.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def paired_ordering(skew: ArrayLike) -> np.ndarray:
    'Construct a matching-preserving, fill-aware initial permutation.\n\nParameters\n----------\nskew : real (n,n), finite skew-symmetric matrix of positive even size; graph edges mean exact nonzero entries.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Skew symmetry uses relative max-entry tolerance 1e-12; the matrix has positive even size.\n\nReturns\n-------\ninteger ndarray (n,), a permutation. Consecutive entries are ascending vertex pairs. Pair order is minimum-current-degree elimination on the quotient graph, with clique fill at each elimination and lexicographic pair ties. Any maximum-cardinality matching is valid.'
    return result
```

### Step 4

04_skew_factor.py

Goal
----
Factor a reordered skew matrix by stable two-by-two congruence pivots.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def skew_factor(skew: ArrayLike, ordering: ArrayLike) -> np.ndarray:
    'Factor a reordered skew matrix by stable two-by-two congruence pivots.\n\nParameters\n----------\nskew : real finite (n,n), skew-symmetric, positive even n.\nordering : integer permutation (n,), initial matched row/column order.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Skew symmetry uses relative max-entry tolerance 1e-12. The ordering contains each index 0,...,n-1 exactly once.\n\nReturns\n-------\nreal ndarray (n+2,n). Rows 0:n hold L; row n is final original-vertex permutation p; row n+1 entries 0:n/2 hold upper-right T-block pivots, entry n/2 holds nonnegative scale, and all remaining entries are zero. The identity is S[p,p]=scale*L*T*L.T. Singular inputs have one or more zero pivots.'
    return result
```

### Step 5

05_pfaffian_certificate.py

Goal
----
Recover oriented parity, logarithmic Pfaffian magnitude and local protection.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def pfaffian_certificate(skew: ArrayLike, factor: ArrayLike, orientation: int = 1) -> np.ndarray:
    'Recover oriented parity, logarithmic Pfaffian magnitude and local protection.\n\nParameters\n----------\nskew : real finite (n,n), skew-symmetric, positive even n.\nfactor : real (n+2,n), any valid scale/L/T/permutation factor from skew_factor.\norientation : integer +1 or -1, raw Pfaffian sign of the chosen trivial reference; default +1.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Skew symmetry uses relative max-entry tolerance 1e-12. The packed permutation contains each integer index once, L is unit lower triangular within absolute tolerance 1e-12, scale is nonnegative, and reserved entries are zero. Zero scale requires a zero matrix and zero pivots. The factor identity must hold within relative max-entry tolerance 1e-9 (absolute tolerance 1e-9 for the zero matrix).\n\nReturns\n-------\nreal ndarray (3,), ordered oriented sign, log(abs(Pf(skew))), local gap. Gap has the units of skew; logarithm uses the supplied numeric energy unit. Singular inputs return [0,0,0], with the middle zero a defined finite sentinel.'
    return result
```

### Step 6

06_pencil_crossings.py

Goal
----
Find all distinct interior gap closures of an affine skew pencil.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def pencil_crossings(pencil: ArrayLike, lower: float, upper: float) -> np.ndarray:
    'Find all distinct interior gap closures of an affine skew pencil.\n\nParameters\n----------\npencil : real (2,n,n), ordered S0,S1, skew-symmetric of positive even size.\nlower, upper : finite floats with lower<upper. Supported instances have real crossings separated by >1e-5, endpoints separated from other roots by >1e-5, and complex roots separated from the real axis by >1e-5. Exact repeated roots and exact endpoint roots are allowed. Numerical equality to the real axis, to another root, or to an endpoint uses absolute tolerance 1e-10; this is distinct from the 1e-6 output root-matching tolerance.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Both pencil matrices satisfy skew symmetry within relative max-entry tolerance 1e-12. The pencil is regular and obeys the root-separation domain stated above.\n\nReturns\n-------\nreal ndarray (n/2+1,), first entry number K of distinct strictly interior roots, next K entries sorted roots, remaining entries zero. Root matching tolerance is 1e-6; exact endpoint roots are excluded.'
    return result
```

### Step 7

07_phase_integrals.py

Goal
----
Integrate local protection over every nontrivial phase interval.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def phase_integrals(pencil: ArrayLike, crossings: ArrayLike, lower: float, upper: float, orientation: int = 1, order: int = 64) -> np.ndarray:
    'Integrate local protection over every nontrivial phase interval.\n\nParameters\n----------\npencil : real (2,n,n), ordered S0,S1.\ncrossings : real (n/2+1,), count K, K sorted interior closures, zero padding as in pencil_crossings.\nlower, upper : finite increasing bounds.\norientation : +1 or -1, sign of the trivial reference.\norder : positive integer quadrature order, default 64.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Both pencil matrices have positive even size and satisfy skew symmetry within relative max-entry tolerance 1e-12. The pencil obeys the regularity and root-separation domain of pencil_crossings. The crossing record has exactly shape (n/2+1,), an integer-valued count in [0,n/2], that many strictly increasing interior roots, and exact zero padding. It contains every distinct interior closure within root tolerance 1e-6.\n\nReturns\n-------\nreal ndarray (K+1,4), rows in increasing c order; columns lower endpoint, upper endpoint, oriented interval sign, integrated gap contribution. The final column is zero in trivial intervals. Units are c,c,dimensionless,energy*c.'
    return result
```

### Step 8

08_topological_exposure.py

Goal
----
Compute the integrated protection of the complete class-AII lattice.

```python
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def topological_exposure(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, profile: ArrayLike, t1: float, delta: float, probe: ArrayLike, kappa: float, cmax: float, order: int = 64) -> float:
    'Compute the integrated protection of the complete class-AII lattice.\n\nParameters\n----------\ncoords : real (N,2), coordinates in a.\nedges : integer (B,2), unique non-self undirected edges.\nmass, potential : real scalars or (N,), onsite coefficients in t.\nprofile : real (N,), dimensionless interlayer texture.\nt1, delta : real bond coefficients in t.\nprobe : real (3,), ordered x/a,y/a,E/t.\nkappa : nonnegative float in t/a.\ncmax : positive upper coupling in t.\norder : positive Gauss–Legendre order per interval, default 64. Inputs satisfy the regular-pencil and root-separation domain of pencil_crossings.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. All input contracts of bhz_layer, aii_pencil and pencil_crossings apply, including distinct coordinates, exact edge shape, finite arrays, nonnegative kappa and a regular pencil. cmax and the integer quadrature order are positive.\n\nReturns\n-------\nfloat, sum of gap integrals over all nontrivial intervals between 0 and cmax, in the supplied dimensionless t units (physical dimension t squared).'
    return result
```
