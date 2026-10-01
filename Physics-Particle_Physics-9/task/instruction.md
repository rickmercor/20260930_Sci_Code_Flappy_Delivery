# Physics-Particle_Physics-9

## Background

The scalar nonabelian two-gluon dipole combines its strongly ordered limit, a finite-energy correction bracket, and the quark-pair and squared-polarization terms it shares with the scalar quark-antiquark dipole. In the identified-particle kinematics mapping, the identified parton's forward momentum fraction z depends on the relative angle of the two emissions, and this dependence couples the invariant kernel to the unresolved phase space. Sector decomposition exposes the overlapping soft and collinear singularities; endpoint distributions then give the Laurent coefficients of the integrated radiator. In the back-to-back configuration a fixed quadrature rule over these sectors gives an estimate of the finite coefficient.

## Problem

Compute the prescribed order-16 quadrature estimate of the finite coefficient of the integrated scalar nonabelian two-gluon dipole in the parton-shower inspired local subtraction scheme for double-real NNLO corrections to color singlet decay. Locate the scalar-radiator method and its dimensionally regulated phase-space integral. Use the back-to-back Born configuration with invariant mass squared one, kappa=-1, and the paper's integrated-radiator normalization I^(nab)_(i;k). Strip color and coupling factors and use the unpartitioned dipole with distinct hard legs. Implement the phase-space mapping, sector subtraction and Laurent expansion for this coefficient, using the numerical rule specified below.

Use x=(t,a,b,r,chi) in the unit five-cube, xi1=t and xi2=t*y. The sector maps below include the alpha=0 hemisphere selectors. They cover one energy order and the hemisphere eta1<1/2. Apart from the rescalings that impose these selectors, the I3 angular map differs from the printed Table V and the azimuth Jacobian differs from Eq. (A26); both corrections are derived below.

| Sector | eta1 | eta2 | y |
|---|---|---|---|
| I1 | `a/2` | `a*b/4` | `r` |
| I2 | `a*b/4` | `b/2` | `a*r` |
| I3 | `a*b*r/4` | `b/2` | `r` |
| I4 | `a/2` | `(a/2)*(1-b/2)` | `r` |
| I5 | `(b/2)*(1-a/2)` | `b/2` | `r` |
| II1 | `a/2` | `1-b/2` | `a*r/2` |
| II2 | `a*min(r,1/2)` | `1-b/2` | `r` |

The maps partition `0<eta1<1/2`, `0<eta2<1`, `0<y<1`. For `eta1<eta2/2` and `eta2<1/2`, write `q=2*eta1/eta2`. I2 covers `y<q`; I3 covers `q<y` through `q=a*r` and `y=r`. Thus I3 uses the energy ratio r in its angular map. Reading the unhatted xi2 in the I3 entry of Table V as the physical xi2=t*y would leave `t*y<q<y` uncovered. In II2, the hemisphere selector applied to the uncut Table V map `eta1=hat_eta1*y` gives the cap `eta1=a*min(y,1/2)`, where hat_eta1 is the Table V sector variable in (0,1).

For the relative azimuth, let `R=sqrt(eta1*(1-eta1)*eta2*(1-eta2))`, `delta=abs(eta1-eta2)`, and `eta3=eta1+eta2-2*eta1*eta2-2*R+4*chi*R`. Use `eta12=delta**2/eta3`, `sinphi=2*delta*sqrt(chi*(1-chi))/eta3`, and `abs(dphi/dchi)=delta/(eta3*sqrt(chi*(1-chi)))`. To check the azimuth correction, put `A=eta1+eta2-2*eta1*eta2` and `B=2*R`. Then `A**2-B**2=delta**2`, `eta3=A-B+2*B*chi`, and solving Eq. (A27) for cosphi, with `cos(theta_i)=1-2*eta_i`, gives `cosphi=(2*A*chi-A+B)/eta3`. Differentiation gives the stated Jacobian. At epsilon zero its integral is pi; the reciprocal factor printed in Eq. (A26) would give `pi*A/delta`. Evaluate `A-B` as `delta**2/(A+B)`, and in I4 and I5 use the exact difference `delta=a*b/4`: the order-16 nodes reach 1.5e-7 on each singular axis and 5e-12 in chi, where direct subtraction loses about half the digits of delta and nearly all digits of A-B.

Factor the t,a,b,r powers as `x_j**(-1-c_j*eps)`, where the space-time dimension is `4-2*eps`. The rate vectors are I1:(4,2,1,2), I2:(4,3,2,2), I3:(4,1,2,3), I4 and I5:(4,2,2,2), II1:(4,3,1,2), II2:(4,1,1,3). Keep `dchi/sqrt(chi*(1-chi))` as the angular measure. Construct the regular Taylor coefficients through eps degree four, including the exact continuous limits on the lower faces and their intersections. Apply `x**(-1-c*eps)=-delta(x)/(c*eps)+sum_{k>=0}(-c*eps)**k/k!*[log(x)**k/x]_+`, with plus action `integral_0^1 log(x)**k*(f(x)-f(0))/x dx`.

Use order n=16. Let `(s_i,w_i)` be the ascending n-point Gauss-Legendre rule on [0,1]. For every plus coordinate use `H(s)=s**3/(s**3+(1-s)**3)` and `Hprime(s)=3*s**2*(1-s)**2/(s**3+(1-s)**3)**2`: the physical nodes are `x_i=H(s_i)` and ordinary weights are `v_i=w_i*Hprime(s_i)`. For II2 only, concatenate `H(s_i)/2` and `(1+H(s_i))/2` in r, with weights `v_i/2` on both halves. At an interior node, the plus coefficient of epsilon degree k is `v_i/x_i*(-c*log(x_i))**k/k!`. At zero it is minus the interior sum; the pole weight is -1/c. Take the tensor product over all four axes, including every zero face and intersection.

For the angular measure put `U(s)=10*s**3-15*s**4+6*s**5`, `Uprime(s)=30*s**2*(1-s)**2`, and `chi_i=sin(pi*U(s_i)/2)**2`, with weight `pi*w_i*Uprime(s_i)`. Use this fixed order-16 rule as stated, with no extrapolation in n. No extra powers of t, a, b, r or chi multiply the integrand. Sector quantities are before any symmetry multiplier; sum their finite coefficients and multiply by four for the other energy order and hard-leg hemisphere.

Report the values this order-16 rule gives for the seven sector finite estimates before the factor four, in the order I1,I2,I3,I4,I5,II1,II2, for the final total, and for the total's four pole coefficients of eps^-4, eps^-3, eps^-2 and eps^-1 after the factor four, all to six decimal places. Put the total in <final_answer> and the seven sector estimates and the four pole coefficients in <reasoning>. Each reported value is accepted within absolute tolerance 5e-5; equivalent decimal representations are accepted. In the short reasoning, state the coefficient that multiplies the squared polarization term in the source's full nonabelian dipole and the normalized physical angular prefactor C(eps) after the unused orientations are integrated. Define `L=1-xi1*xi2*(1-eta12)` and `z=(1-xi1)*(1-xi2)/L`. Use the factorization in which C(eps) multiplies the radial measure `(z*xi1*xi2)**(1-2*eps)/L**(3-2*eps) dxi1 dxi2`, the polar measure `(eta_i*(1-eta_i))**(-eps) deta_i` for each emission, the relative-azimuth measure `sinphi**(-2*eps) dphi` on [0,pi], and the invariant kernel. Apply the coordinate Jacobians before extracting the endpoint powers. Keep the explanation in seven numbered parts: quark-pair series and angular prefactor; nonabelian kernel; quadrature rule; Laurent contraction; I1 through I5; II1 and II2; the total and its pole coefficients.

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

01_assemble_qqbar_sector_jets

Goal
----
Construct the regular quark-pair and polarization jets in any of the seven sectors.

```python
import numpy as np

def assemble_qqbar_sector_jets(sector: str, points: np.ndarray) -> np.ndarray:
    """Construct the regular quark-pair and polarization jets in any of the seven sectors.

    Parameters
    ----------
    sector : str
        I1, I2, I3, I4, I5, II1 or II2.
    points : array_like, shape (N,5)
        Real (t,a,b,r,chi) rows, 0<=t,a,b,r<1 and 0<chi<1.

    Returns
    -------
    jets : ndarray, shape (N,2,5)
        F_Sqq and F_A2 Taylor coefficients through eps degree four.

    Raises
    ------
    ValueError
        Invalid sector, complex dtype, invalid shape, nonfinite input, or input outside the stated domain.
    """
    return None
```

### Step 2

02_assemble_nonabelian_sector_jets

Goal
----
Assemble the full nonabelian regular jets using the quark-pair and polarization series.

```python
import numpy as np

def assemble_nonabelian_sector_jets(sector: str, points: np.ndarray, qqbar_jets: np.ndarray, moments: tuple = (0, 0, 0, 0, 0)) -> np.ndarray:
    """Assemble the full nonabelian regular jets using the quark-pair and polarization series.

    Parameters
    ----------
    sector : str
        One of the seven named sectors.
    points : array_like, shape (N,5)
        Real points in the preceding domain.
    qqbar_jets : array_like, shape (N,2,5)
        Finite real, aligned F_Sqq and F_A2 series.
    moments : sequence of five integers
        Powers from zero through two; bool entries are not accepted.

    Returns
    -------
    jets : ndarray, shape (N,5)
        Moment-weighted nonabelian coefficients in eps powers zero through four.

    Raises
    ------
    ValueError
        Invalid sector, invalid moments, complex dtype, invalid shape, nonfinite input, a row-count mismatch, or input outside the stated domain.
    """
    return None
```

### Step 3

03_assemble_laurent_sector_rule

Goal
----
Build an endpoint-smoothed pole and plus-distribution quadrature.

```python
import numpy as np

def assemble_laurent_sector_rule(sector: str, order: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Build an endpoint-smoothed pole and plus-distribution quadrature.

    Parameters
    ----------
    sector : str
        One of the seven named sectors.
    order : int
        Common order n, one through sixteen; bool is not accepted.
    start : int, optional
        First flat row, default zero.
    stop : int or None, optional
        End of the contiguous flat rows; None selects the complete row count.

    Returns
    -------
    rule : ndarray, shape (N,30)
        Requested rows of product nodes, four Laurent vectors, and angular weights.

    Raises
    ------
    ValueError
        Invalid sector, order outside one through sixteen or not an integer, or invalid slice bounds.
    """
    return None
```

### Step 4

04_combine_laurent_sector_coefficients

Goal
----
Contract the regular jets with every bulk, face and pole-intersection term.

```python
import numpy as np

def combine_laurent_sector_coefficients(rule: np.ndarray, jets: np.ndarray) -> np.ndarray:
    """Contract the regular jets with every bulk, face and pole-intersection term.

    Parameters
    ----------
    rule : array_like, shape (N,30)
        Arbitrary finite real rows in the preceding column layout; coordinate columns 0:5 are unused.
    jets : array_like, shape (N,5)
        Finite real Taylor coefficients aligned with the rule rows.

    Returns
    -------
    coefficients : ndarray, shape (5,)
        Coefficients of eps powers -4,-3,-2,-1,0.

    Raises
    ------
    ValueError
        Complex dtype, invalid shape, nonfinite input or a row-count mismatch.
    """
    return None
```

### Step 5

05_assemble_opposite_hemisphere_coefficients

Goal
----
Integrate the opposite-hemisphere contributions II1 and II2.

```python
import numpy as np

def assemble_opposite_hemisphere_coefficients(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> np.ndarray:
    """Integrate the opposite-hemisphere contributions II1 and II2.

    Parameters
    ----------
    order : int, optional
        Common quadrature order, default sixteen, one through sixteen.
    moments : sequence of five integers, optional
        Coordinate powers, default all zero.

    Returns
    -------
    coefficients : ndarray, shape (2,5)
        Sector rows II1,II2; eps powers -4 through zero.

    Raises
    ------
    ValueError
        Invalid order or moments, propagated from the predecessors.
    """
    return None
```

### Step 6

06_combine_sector_finite_parts

Goal
----
Total the seven sector finite coefficients and apply the factor four for the other energy order and hard-leg hemisphere.

```python
import numpy as np

def combine_sector_finite_parts(same_hemisphere: np.ndarray, opposite: np.ndarray) -> float:
    """Total the seven sector finite coefficients and apply the factor four for the other energy order and hard-leg hemisphere.

    Parameters
    ----------
    same_hemisphere : sequence of five real numbers
        The I1,I2,I3,I4,I5 finite coefficients before symmetry multiplication.
    opposite : ndarray, shape (2, 5)
        The II1 and II2 coefficient rows, epsilon powers -4 through zero.

    Returns
    -------
    finite_part : float
        Four times the sum of the seven finite-coefficient estimates.

    Raises
    ------
    ValueError
        Complex dtype, invalid shapes or nonfinite values.
    """
    return None
```

### Step 7

07_compute_nonabelian_integrated_finite_part

Goal
----
Sum the seven sector finite parts and apply the factor four for the other energy order and hard-leg hemisphere.

```python
import numpy as np

def compute_nonabelian_integrated_finite_part(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> float:
    """Sum the seven sector finite parts and apply the factor four for the other energy order and hard-leg hemisphere.

    Parameters
    ----------
    order : int, optional
        Common quadrature order, default sixteen, one through sixteen.
    moments : sequence of five integers, optional
        Coordinate moment powers, default all zero.

    Returns
    -------
    finite_part : float
        Four times the sum of the seven finite-coefficient estimates.

    Raises
    ------
    ValueError
        Invalid order or moments, propagated from the predecessors.
    """
    return None
```
