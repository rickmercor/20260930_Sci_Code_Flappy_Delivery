# Physics-Optics-34

## Background

Electric-magnetic constitutive coupling allows the electric displacement to depend on the magnetic field and the magnetic induction to depend on the electric field. In relative units the six-component material matrix acts on fields ordered as (E1,E2,E3,H1,H2,H3). The synthetic materials here have positive Hermitian parts and positive dissipative parts, so the example describes passive anisotropic optical media at a single frequency rather than a fitted material spectrum.

A Fourier-modal calculation in a layer uniform along x3 reduces the source-free Maxwell equations to a longitudinal propagation eigenproblem on tangential fields. Discontinuous constitutive parameters require a representation that respects electromagnetic interface conditions. The specified finite harmonic set and directional ordering are part of the observable, not convergence tolerances.

## Problem

Consider propagation through a periodic optical layer with electric-magnetic constitutive coupling, uniform along x3 and containing one rectangular inclusion in each x1-x2 unit cell. Its local relation is $(D,B)=P(E,H)$ with $P_j=[[\epsilon_j,\xi_j],[\xi_j^\dagger,\mu_j]]+i\eta I_6$ for inclusion j=1 and host j=0; all fields and material parameters use relative units with vacuum light speed equal to one, and dagger denotes conjugate transpose. The inputs are the two material tensors, rectangular geometry and retained reciprocal orders, and the output is the greatest real part of a dimensionless longitudinal propagation eigenvalue.

Use the finite reciprocal-space Maxwell eigenproblem for time dependence $\exp(-i\omega t)$ with material products represented by the sequential interface-continuous Fourier construction for general electric-magnetic tensors in its June 2026 journal formulation, not a continuum or effective-medium limit. Its directional ordering is x1 before x2 without averaging the two orderings; retain the specified harmonics at each directional construction, use exact Fourier integrals for the piecewise-constant rectangle and do not introduce spatial sampling or a posteriori mode filtering.

Fractional unit-cell coordinates $u=x_1/a_1$ and $v=x_2/a_2$ are periodic modulo one, and the inclusion is $|u-u_0|_{\rm per}<f_1/2$ and $|v-v_0|_{\rm per}<f_2/2$; use the basis $\exp(i[(k_1+2\pi m/a_1)x_1+(k_2+2\pi n/a_2)x_2+k_3x_3])$. In the short reasoning explain the material continuity choice and report the Frobenius norm of the final 54 by 54 matrix mapping $(E,H)$ Fourier coefficients to $(D,B)$ Fourier coefficients and the selected complex value of $k_3/k_0$; your final answer must be a single number: $\max\operatorname{Re}(k_3/k_0)$ rounded to six decimal places, where $k_0=\omega$.
a1 = 1
a2 = 1.3
k0*a1 = 3.3
k1/k0 = 0.21
k2/k0 = -0.17
m = -1, 0, 1
n = -1, 0, 1
f1 = 0.43
f2 = 0.37
u0 = 0.11
v0 = -0.08
eta = 0.025
epsilon_1 = [[9,1.3,0.7],[1.3,7,-0.8],[0.7,-0.8,6]]
mu_1 = [[1.7,0.18,-0.12],[0.18,1.3,0.15],[-0.12,0.15,1.5]]
xi_1 = [[1.2i,0.55+0.25i,0.2-0.3i],[-0.25+0.4i,0.9i,0.45+0.15i],[0.35+0.1i,-0.3+0.2i,1.1i]]
epsilon_0 = [[2.4,0.15,-0.05],[0.15,2.1,0.12],[-0.05,0.12,2.7]]
mu_0 = [[1.1,0.04,0],[0.04,1.2,-0.03],[0,-0.03,1.05]]
xi_0 = [[0.15i,0.08,0.03i],[-0.04,0.2i,0.06],[0.02i,-0.05,0.1i]]

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

01_material_operator

Goal
----
Assemble the relative constitutive operator P=[[epsilon,xi],[xi^dagger,mu]]+i eta I_6 in component order (E1,E2,E3,H1,H2,H3) for inputs and (D1,D2,D3,B1,B2,B3) for outputs. All three tensors are finite complex 3 by 3 arrays with entry magnitudes at most 1e6. eta is a finite real scalar in [0,1]. Return the complex 6 by 6 matrix; invalid inputs raise ValueError. The dagger denotes conjugate transpose, not elementwise conjugation.

```python
def material_operator(epsilon: "np.ndarray | list | tuple", mu: "np.ndarray | list | tuple", xi: "np.ndarray | list | tuple", eta: float) -> "np.ndarray":
    """Assemble one electric-magnetic material matrix.

    Parameters
    ----------
    epsilon, mu, xi : array_like
        Complex 3 by 3 tensors with finite entries of magnitude at most 1e6.
    eta : float
        Real finite loss coefficient in [0,1].

    Returns
    -------
    ndarray
        Complex 6 by 6 constitutive matrix in the stated component order.

    Raises
    ------
    ValueError
        For invalid shape, magnitude, finiteness or loss coefficient.
    """
    return None
```

### Step 2

02_indicator_matrix

Goal
----
Construct the exact Fourier multiplication matrix of a periodic interval with width fraction f and center c. Retain orders -M through M in increasing order and define T[p,q]=f sinc((p-q)f) exp(-2 pi i (p-q)c), where sinc(x)=sin(pi x)/(pi x) with sinc(0)=1. M is an integer in [0,4], f is a finite real scalar in [0,1] and c is a finite real scalar in [-1,1]. Boolean M is invalid. Return the complex Hermitian matrix; invalid inputs raise ValueError.

```python
def indicator_matrix(order: int, fraction: float, center: float) -> "np.ndarray":
    """Return the interval's exact finite Fourier multiplication matrix.

    Parameters
    ----------
    order : int
        Maximum retained harmonic from 0 through 4, excluding booleans.
    fraction : float
        Real interval width in [0,1].
    center : float
        Real interval center in [-1,1].

    Returns
    -------
    ndarray
        Complex matrix of shape (2*order+1,2*order+1).

    Raises
    ------
    ValueError
        For invalid type, range or nonfinite scalar.
    """
    return None
```

### Step 3

03_directional_factorization

Goal
----
Compute one directional interface-continuous Fourier representation of a six-component electric-magnetic operator. Each local operator has shape (6s,6s), ordered component-major with s old harmonics per component; columns are (E1,E2,E3,H1,H2,H3) and rows are (D1,D2,D3,B1,B2,B3). Group components axis and axis+3 as a and the other components in increasing order as b. For each local P write A=Paa, B=Pab, C=Pba and D=Pbb, then Q=A^{-1}, R=QB, L=CQ and W=D-CQB. For the supplied k by k indicator T, lift each block pair by H(U)=kron(T,U_in-U_out)+kron(I_k,U_out). Reconstruct V=H(Q)^{-1}, Paa=V, Pab=V H(R), Pba=H(L)V and Pbb=H(W)+H(L)V H(R). Each block initially has order (new harmonic, grouped component, old harmonic); undo grouping and return (original component, new harmonic, old harmonic). Products are not conjugated. Accept finite entries of magnitude at most 1e6, 1<=s<=25, 1<=k<=9 and 6sk<=150. axis is an integer in [0,2], not boolean. T must be Hermitian to maximum-entry tolerance 1e-10 and its Hermitian part's eigenvalues must lie in [-1e-10,1+1e-10]. Required inverses must exist and all computed arrays must be finite. Raise ValueError otherwise; do not impose a condition-number cutoff on invertible matrices. The mathematical construction, not a particular floating-point evaluation order, defines the output. Return each entry to relative tolerance 1e-7 plus absolute tolerance 1e-9 for the supplied binary64 inputs, including normal pivots of order 1e-20 with nearly common off-diagonal blocks whose Schur terms cancel. Use a cancellation-resistant reformulation or adequate working precision; these valid cases must not be refused merely for conditioning.

```python
def directional_factorization(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", indicator: "np.ndarray | list | tuple", axis: int) -> "np.ndarray":
    """Factorize a two-material directional Fourier operator.

    Parameters
    ----------
    p_in, p_out : array_like
        Same-shaped complex (6s,6s) arrays, 1<=s<=25, entries at most 1e6.
    indicator : array_like
        Complex Hermitian contractive (k,k) array, 1<=k<=9, entries at most 1e6.
        Absolute Hermiticity and spectral slack are 1e-10; 6*s*k<=150.
    axis : int
        Normal direction 0, 1 or 2, excluding booleans.

    Returns
    -------
    ndarray
        Complex (6*s*k,6*s*k) matrix in original component order, then new
        harmonic and old harmonic. Original order is (E1,E2,E3,H1,H2,H3).
        Entrywise accuracy is rtol=1e-7, atol=1e-9, including finite cancellation
        cases with pivots of order 1e-20. Inputs mean their binary64 values.

    Raises
    ------
    ValueError
        For invalid shapes, bounds, axis, indicator or singular inverses,
        or nonfinite intermediates/results. No condition-number cutoff applies.
    """
    return None
```

### Step 4

04_rectangle_operator

Goal
----
Compose the directional constitutive Fourier construction for a rectangular inclusion. First apply directional_factorization to inclusion P1 and host P0 with the x1 indicator and axis 0. Outside the x2 interval the first-stage host is kron(P0,I_k1), with k1 the x1 indicator size. Apply directional_factorization to these first-stage operators with the x2 indicator and axis 1. Return the final matrix in (component,n,m) order, where m is the old x1 harmonic and n the new x2 harmonic. Inputs P1 and P0 are finite complex 6 by 6 matrices with magnitudes at most 1e6; both indicator matrices satisfy the directional step's Hermitian/contractive tolerance 1e-10, have odd sizes from 1 to 9 and their size product is at most 25. Required inverses must exist and results must be finite. Invalid inputs raise ValueError.

```python
def rectangle_operator(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", tx: "np.ndarray | list | tuple", ty: "np.ndarray | list | tuple") -> "np.ndarray":
    """Compose the x1-then-x2 rectangular Fourier operator.

    Parameters
    ----------
    p_in, p_out : array_like
        Finite complex 6 by 6 material matrices, entry magnitudes at most 1e6.
    tx, ty : array_like
        Odd-sized Hermitian contractive indicators, size 1 through 9 each,
        with size product at most 25 and tolerance 1e-10.

    Returns
    -------
    ndarray
        Complex matrix ordered by component, x2 harmonic then x1 harmonic.

    Raises
    ------
    ValueError
        For invalid shapes or any violated directional factorization contract.
    """
    return None
```

### Step 5

05_maxwell_operator

Goal
----
Reduce the constitutive Fourier operator to the dimensionless longitudinal Maxwell matrix. Input P has shape (6N,6N) in (component,n,m) order, with N=(2mx+1)(2my+1), m fast, mx and my integers in [0,4] and N<=25. Its entry magnitudes are at most 1e6. The positive real period ratio a2/a1 is in [0.25,4], positive k0*a1 is in [0.25,20] and real k1/k0 and k2/k0 lie in [-2,2]. Define X=diag(k1/k0+2 pi m/(k0*a1)), Y=diag(k2/k0+2 pi n/((a2/a1)(k0*a1))) and tangential order t=(E1,E2,H1,H2). Normal order is z=(E3,H3). Let C_z=[[0,0,Y,-X],[-Y,X,0,0]] and Z=Pzz^{-1}(C_z-Pzt); its row halves are Z_E and Z_H. Let U=Ptt+Ptz Z and split its rows as U_D1,U_D2,U_B1,U_B2. Return rows [U_B2+X Z_E;-U_B1+Y Z_E;-U_D2+X Z_H;U_D1+Y Z_H]. No condition-number cutoff applies. All inputs and intermediate/results must be finite; a singular Pzz or invalid input raises ValueError.

```python
def maxwell_operator(p: "np.ndarray | list | tuple", mx: int, my: int, period_ratio: float, frequency: float, qx: float, qy: float) -> "np.ndarray":
    """Return the dimensionless tangential propagation matrix.

    Parameters
    ----------
    p : array_like
        Finite complex (6N,6N) constitutive operator, magnitude at most 1e6.
    mx, my : int
        Retained orders 0 through 4, excluding booleans; N<=25.
    period_ratio : float
        a2/a1 in [0.25,4].
    frequency : float
        k0*a1 in [0.25,20].
    qx, qy : float
        k1/k0 and k2/k0 in [-2,2].

    Returns
    -------
    ndarray
        Complex (4N,4N) matrix with eigenvalues k3/k0.

    Raises
    ------
    ValueError
        For invalid inputs, singular longitudinal block or nonfinite arithmetic.
    """
    return None
```

### Step 6

06_spectral_abscissa

Goal
----
Return the maximum real part of all eigenvalues of a finite complex square matrix. The input dimension is 1 through 100 and every entry magnitude is at most 1e6. No filtering by imaginary part, eigenvector norm or propagation direction is performed. Return a native float without decimal rounding. Invalid shape, nonfinite input, eigensolver failure or nonfinite eigenvalues raises ValueError.

```python
def spectral_abscissa(matrix: "np.ndarray | list | tuple") -> float:
    """Return the greatest real part of the full eigenvalue spectrum.

    Parameters
    ----------
    matrix : array_like
        Complex square matrix of dimension 1 through 100, entries at most 1e6.

    Returns
    -------
    float
        Maximum real eigenvalue component, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite data, exceeded bounds or eigensolver failure.
    """
    return None
```

### Step 7

07_run_pipeline

Goal
----
Compute the finite-harmonic propagation target for the fixed passive rectangular optical layer. Assemble inclusion and host matrices with material_operator, form exact interval indicators with indicator_matrix, compose x1-then-x2 rectangle_operator, eliminate longitudinal fields with maxwell_operator and return spectral_abscissa. Use epsilon1=[[9,1.3,0.7],[1.3,7,-0.8],[0.7,-0.8,6]], mu1=[[1.7,0.18,-0.12],[0.18,1.3,0.15],[-0.12,0.15,1.5]], xi1=[[1.2i,0.55+0.25i,0.2-0.3i],[-0.25+0.4i,0.9i,0.45+0.15i],[0.35+0.1i,-0.3+0.2i,1.1i]], epsilon0=[[2.4,0.15,-0.05],[0.15,2.1,0.12],[-0.05,0.12,2.7]], mu0=[[1.1,0.04,0],[0.04,1.2,-0.03],[0,-0.03,1.05]] and xi0=[[0.15i,0.08,0.03i],[-0.04,0.2i,0.06],[0.02i,-0.05,0.1i]]. Both materials add loss 0.025i times identity and use xi conjugate transpose in the lower-left block. Set a2/a1=1.3, k0*a1=3.3, k1/k0=0.21, k2/k0=-0.17, interval centers 0.11 and -0.08. Inputs mx,my are nonboolean integers in [0,4] with (2mx+1)(2my+1)<=25; fractions fx,fy are real in [0,1]. Return the unrounded scalar. Invalid input or upstream failure raises ValueError.

```python
def run_pipeline(mx: int, my: int, fx: float, fy: float) -> float:
    """Return the fixed-material rectangular propagation target.

    Parameters
    ----------
    mx, my : int
        Retained harmonic orders 0 through 4; total harmonics at most 25.
    fx, fy : float
        Real inclusion interval fractions in [0,1].

    Returns
    -------
    float
        Greatest real component of k3/k0, without decimal rounding.

    Raises
    ------
    ValueError
        For invalid inputs or a violated upstream numerical contract.
    """
    return None
```
