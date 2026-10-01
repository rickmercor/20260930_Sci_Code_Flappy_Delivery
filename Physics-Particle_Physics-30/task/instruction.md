# Physics-Particle_Physics-30

## Background

The model describes finite-length flux tubes on a three-dimensional periodic dual lattice. Use zero-based coordinates and spatial directions 0,1,2, with independent plaquettes stored in the order (01,02,12).

With periodic wrapping, define
d_mu^+ f(x) = f(x+mu)-f(x),
d_mu^- f(x) = f(x)-f(x-mu).

The complex monopole field is chi=c1+i*c2, and its covariant difference is
D_mu chi(x) = chi(x)-exp(i*B_mu(x))*chi(x+mu).

The noncompact field strength is
F_uv(x) = d_u^+ B_v(x)-d_v^+ B_u(x)-2*pi*Sigma_uv(x).

At beta_g=1 the action is
S = 0.5*sum_{x,u<v} F_uv(x)^2
  + 0.5*m_B^2*sum_{x,mu} |D_mu chi(x)|^2
  + (m_B^2*m_chi^2/8)*sum_x (|chi(x)|^2-1)^2.

The normalized residuals are X_mu=dS/dB_mu and Y_alpha=(dS/dc_alpha)/m_B^2. Derive their local Newton curvatures from this action. Update the two scalar components jointly using the full symmetric 2-by-2 curvature block, including its off-diagonal entry.

For reproducibility, initialize B_mu=0, c1=1 and c2=0 everywhere. A sweep visits even site parity (x+y+z)%2=0 before odd parity. Within each parity, update B0, B1 and B2 in that order, followed by the joint scalar update. Recompute residuals and curvatures before each stage, and update all sites of the selected parity simultaneously within a stage. Subtract 0.8 times the local Newton correction. Check convergence before each sweep and after the final permitted sweep; require every normalized residual magnitude to be below 1e-9. Allow up to 20000 sweeps and raise ValueError on nonconvergence. No particular sweep count is graded.

The periodic nearest-neighbor Laplacian is Delta_L=sum_mu d_mu^- d_mu^+. Its massless Green function must use a consistent treatment of the constant mode. The finite-volume Hodge decomposition must likewise retain the required correction to the solenoidal field. Identify these two prescriptions rather than assuming a continuum kernel or discarding a finite-volume contribution. Their explicit formulas belong to the code sub-problems and the source-based reasoning, not this background.

For component orientations, use the charge density
j0 = -(d_0^+ Sigma_12-d_1^+ Sigma_02+d_2^+ Sigma_01).
Let psi=G*j0 denote periodic convolution and use
C_uv = -epsilon_uvk*d_k^- psi,
Fcoul_uv = 2*pi*C_uv,
Breg_mu = G*(sum_nu d_nu^- F_munu),
where epsilon_012=+1 and F is extended antisymmetrically from the stored plaquettes. Obtain the corrected solenoidal field and the action decomposition consistently with these definitions. V_sole(r) is the solenoidal action evaluated on the stationary fields at separation r.

The reference tension used to normalize the difference of the two adjacent secant tensions is pi*m_B^2. It is a continuum Bogomolnyi reference, not an assertion that the finite-lattice tension equals that value.

## Problem

Compute the finite-box curvature of the solenoidal quark-antiquark potential in the three-dimensional dual-lattice dual Ginzburg-Landau model on a periodic L=16 cubic lattice with a=beta_g=N_q=1 and m_B=m_chi=0.5; use zero-based coordinates and components, with plaquettes ordered (01,02,12).

For each r in {3,5,7}, set Sigma_01(0,0,z)=-1 for z=1,...,r and all other independent Dirac plaquettes to zero, use F=curl(B)-2*pi*Sigma and the full complex monopole field, and solve from B=0, chi=1+0i until the maximum absolute normalized gauge/scalar residual is below 1e-9 under the background's checkerboard Newton protocol.

Obtain V_sole(r) from the periodic Hodge decomposition using the literature-prescribed zero-mode-consistent Green function and finite-volume-corrected solenoidal field; identify and justify both withheld conventions rather than substituting continuum or radial formulas.

Compute Q=100*[V_sole(7)-2*V_sole(5)+V_sole(3)]/[2*pi*m_B^2].

In short numbered reasoning, report only the three solenoidal potentials, their second difference and Q; also explain the Hodge allocation and vanishing cross term, derive the Coulombic pair-energy identity, state the residual criterion, and interpret Q through adjacent secant tensions and periodic-box effects.

Use only NumPy and the Python standard library, without precomputed fields, network access, file I/O or plotting; the network restriction does not apply to literature retrieval in browsing-enabled reasoning evaluation.

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

build_finite_source

Goal
----
Construct the finite Dirac-string source on a periodic cubic lattice using zero-based coordinates 0,...,L-1 and independent plaquettes ordered as (01,02,12). Set Sigma_01(0,0,z)=-charge for z=1,...,R inclusive and set every other entry to zero. Require an even integer L>=4, an integer separation satisfying 1<=R<L/2, and a nonzero integer charge. Raise ValueError for invalid inputs.

```python
import numpy as np

def build_finite_source(
    L: int,
    R: int,
    charge: int = 1,
) -> np.ndarray:
    """Construct the prescribed finite Dirac source.

    Parameters
    ----------
    L : int
        Even cubic side length, at least 4; booleans are excluded.
    R : int
        Separation with 1 <= R < L/2; booleans are excluded.
    charge : int, default 1
        Nonzero signed integer charge; booleans are excluded.

    Returns
    -------
    numpy.ndarray
        Real array of shape (3,L,L,L), plaquettes (01,02,12).
        Only sigma[0,0,0,1:R+1] is nonzero and equals -charge.

    Raises
    ------
    ValueError
        If L, R or charge violates the specified integer/range contract.
    """
    return np.empty(0, dtype=float)
```

### Step 2

lattice_curl

Goal
----
Compute the forward lattice curl of a real gauge-link array B with shape (3,Lx,Ly,Lz). Return the three independent plaquette components in order (01,02,12), using curl(B)_uv(x)=B_u(x)+B_v(x+u)-B_u(x+v)-B_v(x). All spatial shifts wrap periodically. Require finite inputs and each spatial side length at least 2; raise ValueError for invalid inputs. Do not reduce the resulting plaquette values modulo 2*pi.

```python
import numpy as np

def lattice_curl(B: np.ndarray) -> np.ndarray:
    """Compute the periodic noncompact forward lattice curl.

    Parameters
    ----------
    B : numpy.ndarray
        Finite real gauge links of shape (3,Lx,Ly,Lz), with each
        spatial side at least 2. The input is not modified.

    Returns
    -------
    numpy.ndarray
        Real array with the same shape, plaquettes (01,02,12).
        Values use periodic forward differences and are not reduced
        modulo 2*pi.

    Raises
    ------
    ValueError
        If the numeric input has incorrect shape, a side below 2,
        or a nonfinite value.
    """
    return np.empty(0, dtype=float)
```

### Step 3

lattice_equations

Goal
----
Evaluate the normalized field-equation residuals and local Newton curvatures for the three-dimensional dual-lattice DGL model using Eqs. (43)–(48) and (51)–(52).



The state array has shape (5,Lx,Ly,Lz), with channels (B0,B1,B2,Re chi,Im chi). The source array sigma has shape (3,Lx,Ly,Lz), with plaquettes ordered as (01,02,12). Use periodic boundaries and epsilon_12=+1.



Return eleven channels in this exact order:

(X0,X1,X2,Y1,Y2,dX0,dX1,dX2,H11,H12,H22).



Here X_mu=dS/dB_mu and Y_alpha=(dS/dc_alpha)/mB^2 for beta_g=1. The dX channels are diagonal gauge curvatures, and H is the full symmetric 2-by-2 scalar Newton block.



Require compatible finite arrays, spatial side lengths at least 2, and positive finite masses. Raise ValueError for invalid inputs.

```python
import numpy as np

def lattice_equations(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Evaluate normalized residuals and local Newton curvatures.

    Parameters
    ----------
    state : numpy.ndarray
        Finite real array (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2.
        Each spatial side is at least 2. The input is not modified.
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz), plaquettes (01,02,12).
        Spatial dimensions must match state.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass; need not equal mB in this step.

    Returns
    -------
    numpy.ndarray
        Real array (11,Lx,Ly,Lz) with channels
        X0,X1,X2,Y1,Y2,dX0,dX1,dX2,H11,H12,H22.
        X=dS/dB and Y=(dS/dc)/mB**2; H is the normalized scalar block.

    Raises
    ------
    ValueError
        For invalid array shapes, incompatible dimensions, nonfinite
        numeric values or nonpositive masses.
    """
    return np.empty(0, dtype=float)
```

### Step 4

newton_sweep

Goal
----
Perform one damped checkerboard block-Newton sweep without modifying the input arrays.



The state channels are (B0,B1,B2,Re chi,Im chi). Visit even site parity (x+y+z)%2=0 first, then odd parity. Within each parity, update B0, B1, and B2 in that order, followed by a joint update of the two scalar components.



Recompute residuals and curvatures before each of these four stages. Within a stage, update all selected sites simultaneously. For a gauge channel, subtract damping*X/dX. For the scalar block, subtract damping*inverse(H)*Y.



Require compatible finite arrays, positive finite masses, even spatial side lengths of at least 2, and 0<damping<=1. Raise ValueError for invalid inputs or a nonpositive Newton curvature/block. Return a new state array.

```python
import numpy as np

def newton_sweep(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    damping: float = 0.8,
) -> np.ndarray:
    """Perform one damped checkerboard block-Newton sweep.

    Parameters
    ----------
    state : numpy.ndarray
        Finite real state (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2.
        Spatial sides must be even and at least 2. Not modified.
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz) matching state; not modified.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass.
    damping : float, default 0.8
        Finite correction multiplier with 0 < damping <= 1.

    Returns
    -------
    numpy.ndarray
        A new real array of the same shape as state after one sweep.
        Visit even then odd parity; at each parity update B0,B1,B2
        then the coupled c1,c2 block, recomputing between stages.

    Raises
    ------
    ValueError
        For incompatible/nonfinite numeric inputs, invalid side
        lengths, nonpositive masses, invalid damping, nonpositive
        selected gauge curvatures or a non-positive-definite scalar block.
    """
    return np.empty(0, dtype=float)
```

### Step 5

solve_finite_tube

Goal
----
Solve the three-dimensional finite-source DGL field equations using the specified checkerboard Newton sweep.



Initialize B0=B1=B2=0, Re chi=1, and Im chi=0 everywhere. Before each sweep, check the maximum absolute value over the five normalized residual channels (X0,X1,X2,Y1,Y2). Return the state when this maximum is strictly below tolerance.



Use damping=0.8, tolerance=1e-9, and max_sweeps=20000 by default. Check convergence again after the last permitted sweep. Raise ValueError if the solver does not converge.



Require a finite source array of shape (3,Lx,Ly,Lz), even spatial side lengths of at least 4, positive finite masses and tolerance, a positive integer max_sweeps, and 0<damping<=1. Do not modify the source array or use precomputed fields.

```python
import numpy as np

def solve_finite_tube(
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
    max_sweeps: int = 20000,
    damping: float = 0.8,
) -> np.ndarray:
    """Solve the finite-source stationary DGL equations.

    Parameters
    ----------
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz), plaquettes (01,02,12).
        Spatial sides must be even and at least 4. Not modified.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass.
    tolerance : float, default 1e-9
        Positive finite threshold for all five normalized residuals.
    max_sweeps : int, default 20000
        Positive integer sweep budget; booleans are excluded.
    damping : float, default 0.8
        Finite correction multiplier with 0 < damping <= 1.

    Returns
    -------
    numpy.ndarray
        Real state (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2, obtained
        from B=0,c1=1,c2=0. The largest normalized residual magnitude
        is strictly below tolerance. Convergence is checked before
        each sweep and after the last allowed sweep.

    Raises
    ------
    ValueError
        For invalid numeric input shapes, side lengths, masses,
        tolerance, sweep budget or damping; for an invalid Newton
        curvature/block; or if convergence is not reached.
    """
    return np.empty(0, dtype=float)
```

### Step 6

periodic_green

Goal
----
Construct the real, zero-mean massless Green function on a periodic cubic lattice of size L^3.



Use the nearest-neighbor lattice Laplacian

Delta_L f(x)=sum_mu[f(x+mu)+f(x-mu)-2*f(x)].



The Green function must satisfy

Delta_L G(x)=-delta_x0+1/L^3.



Follow the finite-volume prescription in Appendix A, Eqs. (A6)–(A8): use the inverse lattice-Laplacian eigenvalue for every nonzero momentum mode and set the joint zero-mode coefficient to zero. Do not substitute the continuum 1/r Green function.



Require an integer L>=2 and raise ValueError for invalid input.

```python
import numpy as np

def periodic_green(L: int) -> np.ndarray:
    """Construct the zero-mean periodic massless Green function.

    Parameters
    ----------
    L : int
        Cubic side length at least 2; booleans are excluded.
        Odd lengths are allowed in this Green-function step.

    Returns
    -------
    numpy.ndarray
        Real array (L,L,L), with source at (0,0,0) and zero mean.
        Nonzero Fourier modes invert minus the lattice Laplacian;
        the joint zero mode is zero, giving Delta_L G=-delta+1/L**3.

    Raises
    ------
    ValueError
        If L is not an integer (excluding booleans) at least 2.
    """
    return np.empty(0, dtype=float)
```

### Step 7

hodge_action

Goal
----
Evaluate the full and Hodge-decomposed lattice actions at beta_g=1 using Eqs. (42)–(44) and (55)–(69).







Inputs are the state array with channels (B0,B1,B2,Re chi,Im chi), the Dirac source with plaquettes ordered as (01,02,12), and the matching zero-mean periodic massless Green function G.







Compute the charge density, Coulombic field, and regular Hodge potential using periodic differences and convolution. Include the finite-volume correction Fsole=curl(Breg)-2*pi*xi/V, where xi is the spatial sum of each source component and V is the number of lattice sites.







Evaluate Ssole using the corrected solenoidal gauge field and the original scalar kinetic and potential terms.







Return [S,Scoul,Ssole,max_field_reconstruction_error,energy_reconstruction_error], where the last entry is the signed difference S-Scoul-Ssole. Raise ValueError for incompatible shapes, nonfinite inputs, or nonpositive masses.



Returns

-------

A real NumPy array of length 5 containing [S,Scoul,Ssole,max_field_reconstruction_error,energy_reconstruction_error]. The energy error is the signed quantity S-Scoul-Ssole.

```python
import numpy as np

def hodge_action(
    state: np.ndarray,
    sigma: np.ndarray,
    G: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Evaluate the full action, Hodge split and reconstruction errors.

    Parameters
    ----------
    state : numpy.ndarray
        Finite real array (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2.
        Each spatial side is at least 2.
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz), plaquettes (01,02,12),
        with spatial dimensions matching state.
    G : numpy.ndarray
        Finite real array (Lx,Ly,Lz). The caller supplies the
        matching zero-mean periodic Green function; this function
        validates shape and finiteness, not the Green equation.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass.

    Returns
    -------
    numpy.ndarray
        Real array of length 5: S,Scoul,Ssole,max_field_error,energy_error.
        max_field_error=max(abs(F-Fsole-Fcoul)); energy_error=S-Scoul-Ssole
        is signed. Ssole retains both original matter terms and the
        finite-volume field correction. Input arrays are not modified.

    Raises
    ------
    ValueError
        For invalid/incompatible numeric shapes, nonfinite arrays
        or masses, spatial sides below 2 or nonpositive masses.
    """
    return np.empty(0, dtype=float)
```

### Step 8

run_length_curvature

Goal
----
Compute the finite-length solenoidal-potential curvature on a periodic cubic lattice.



Construct unit-charge Dirac sources at separations R-halfspan, R, and R+halfspan. For each source, solve the full three-dimensional DGL equations using the specified vacuum initialization, checkerboard ordering, damping 0.8, and residual tolerance. Use the periodic Green function and corrected Hodge decomposition to obtain each solenoidal potential.



Return the scalar

Q=100*[Vsole(R+halfspan)-2*Vsole(R)+Vsole(R-halfspan)]/[halfspan*pi*mB^2].



Defaults are L=16, R=5, halfspan=2, mB=mchi=0.5, and tolerance=1e-9.



Require an even integer L>=4, integer R, positive integer halfspan, positive finite equal masses, positive finite tolerance, and all three separations satisfying 1<=separation<L/2. Raise ValueError for invalid inputs or solver nonconvergence.

```python
import numpy as np

def run_length_curvature(
    L: int = 16,
    R: int = 5,
    halfspan: int = 2,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
) -> float:
    """Compute finite-box solenoidal curvature through all earlier steps.

    Parameters
    ----------
    L : int, default 16
        Even cubic side length at least 4; booleans are excluded.
    R : int, default 5
        Central separation; booleans are excluded.
    halfspan : int, default 2
        Positive integer separation increment; booleans are excluded.
        Each of R-halfspan, R, R+halfspan must lie in [1,L/2).
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass, equal to mB in this benchmark.
    tolerance : float, default 1e-9
        Positive finite threshold for all normalized residual channels.

    Returns
    -------
    float
        Q=100*(Vsole(R+halfspan)-2*Vsole(R)+Vsole(R-halfspan))
        /(halfspan*pi*mB**2). This is a finite-box difference of
        secant tensions expressed as a percentage of pi*mB**2.
        Each solve uses damping 0.8 and at most 20000 sweeps.

    Raises
    ------
    ValueError
        For violations of the stated numeric/integer contracts,
        invalid separation ranges, unequal masses, solver failure
        (including invalid Newton curvature) or a nonfinite result.
    """
    return 0.0
```
