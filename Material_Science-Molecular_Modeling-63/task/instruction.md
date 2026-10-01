# Material_Science-Molecular_Modeling-63

## Background

Environment-dependent screened charges couple local atomic structure to periodic electrostatics. Their coordinate dependence contributes to the electrostatic forces and force constants.

Long-wavelength longitudinal optical motion in an ionic solid couples to a macroscopic electric field. The frequency shift combines analytic lattice stiffness with the polarization response. The present model uses finite Ewald sums, scaled Born tensors and isotropic electronic screening. Translational acoustic constraints and the atomic mass metric define the optical spectrum.

## Problem

Calculate the longitudinal shift of the highest optical frequency of the neutral periodic model below. Use the attached study's type-weighted charge redistribution and phase-compensated periodic dipole. The model charges are screened charges. Electronic screening is isotropic; no independent dielectric constant is supplied.

Derive the force constants and polarization response of this environment-dependent charge model. Apply the acoustic constraints specified below and evaluate the analytic and longitudinal optical spectra for direction (2, -1, 3). The requested observable is

$$
\Delta\omega=\sqrt{\lambda_{\max}(D_{\rm opt}^{L})}-\sqrt{\lambda_{\max}(D_{\rm opt}^{A})}.
$$

Here A denotes the analytic operator and L includes the long-wavelength longitudinal correction. Use reduced units with Coulomb prefactor one and no conversion by 2 pi.

Provide the conserved charge vector, electrostatic energy, electrostatic force on all four atoms, raw scaled Born tensor of atom 0, and largest eigenvalue of each optical operator. The force is the negative Cartesian derivative of the electrostatic energy alone, evaluated before adding the equilibrium counterterm. Report the signed Born tensor before acoustic correction; its row index is polarization and its column index is displacement. Derive the charge-response and energy-derivative terms needed for these quantities.

Model data

All constants are benchmark inputs. Atom and type indices start at zero. The fixed orthorhombic cell lengths are (4.7, 5.1, 6.2).

| Atom | Type | x | y | z | Mass |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0.35 | 0.55 | 0.80 | 1.0 |
| 1 | 1 | 2.05 | 1.65 | 2.70 | 1.8 |
| 2 | 0 | 3.55 | 3.25 | 1.60 | 1.3 |
| 3 | 1 | 1.25 | 4.20 | 4.75 | 2.1 |

For central atom i, include every periodic image satisfying

$$
d=r_j+L\odot n-r_i,\qquad n\in\mathbb Z^3,\qquad 0<|d|<R_c=5.4.
$$

Include nonzero self images. Define

$$
M_{\nu,i}=\sum_{j,n}' c_{z_i z_j}\left(1-\frac{|d|}{R_c}\right)^4d^{\otimes\nu},\qquad \nu=0,1,2,
$$

$$
c=\begin{pmatrix}0.9&1.2\\0.7&1.05\end{pmatrix},\qquad
f_i=(M_0,M_0^2,M_1\cdot M_1,M_2:M_2,M_1^TM_2M_1)_i.
$$

The local raw charge and total charge are

$$
y_i=b_{z_i}+a_{z_i}\cdot f_i,\qquad Q_{\rm total}=0.
$$

Use the following fixed type parameters in the redistribution model.

| Type | b | a0 | a1 | a2 | a3 | a4 | Softness s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.65 | 0.120 | -0.025 | 0.015 | -0.004 | 0.003 | 1.0 |
| 1 | -0.58 | -0.085 | 0.020 | -0.010 | 0.005 | -0.002 | 1.7 |

The electrostatic energy is the finite Ewald sum with alpha = 0.72. Real-image indices run from -2 to 2 on each axis. Reciprocal indices run from -4 to 4 on x and y, and from -5 to 5 on z, excluding the zero vector. These finite index sets define the numerical model. With

$$
V=L_xL_yL_z,\qquad k=2\pi m/L,\qquad S(k)=\sum_iq_i e^{ik\cdot r_i},
$$

the energy is

$$
E_{\rm el}=\frac12\sum_{ij,n}'q_iq_j\frac{\operatorname{erfc}(\alpha|r_i-r_j+L\odot n|)}{|r_i-r_j+L\odot n|}
+\frac{2\pi}{V}\sum_{k\ne0}\frac{e^{-k^2/(4\alpha^2)}}{k^2}|S(k)|^2
-\frac{\alpha}{\sqrt\pi}\sum_iq_i^2
-\frac{\pi}{2\alpha^2V}\left(\sum_iq_i\right)^2.
$$

The real-space prime excludes only i = j, n = 0. Both signs of every reciprocal mode are included. Use explicit image sums and hold cell lengths and image labels fixed during Cartesian differentiation. The final term defines the uniform-background convention; it vanishes along the neutral charge manifold.

The short-range quadratic energy is

$$
E_{\rm short}^{(2)}=\frac12\sum_{i<j}k_{ij}|u_i-u_j|^2.
$$

| Pair | 0-1 | 0-2 | 0-3 | 1-2 | 1-3 | 2-3 |
|---|---:|---:|---:|---:|---:|---:|
| Spring constant | 1.1 | 0.7 | 0.9 | 1.3 | 0.6 | 1.2 |

A linear short-range counterterm cancels the electrostatic force at the reference geometry. The short-range Hessian contains no macroscopic nonanalytic term.

The acoustic constraints are zero analytic-Hessian response to each uniform Cartesian translation and zero atom-sum of every Born-tensor component. For the Hessian, use the nearest symmetric matrix satisfying these constraints in the unweighted Frobenius norm. Correct the Born tensors independently in the unweighted Euclidean norm. Apply these corrections before mass weighting. Both optical operators act on the complement of the three mass-weighted translations.

Numerical reporting

Absolute tolerances apply to each reported scalar or tensor component.

| Quantity | Absolute tolerance |
|---|---:|
| Conserved charges | 1e-6 |
| Electrostatic energy | 5e-7 |
| Electrostatic force | 2e-6 |
| Raw scaled Born tensor of atom 0 | 2e-6 |
| Each largest optical eigenvalue | 2e-6 |
| Final frequency shift | 1e-7 |

Seven decimal places are sufficient for reporting; equivalent scientific notation is accepted. Evaluate the final shift from the unrounded eigenvalues.

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include the charge-response and energy-derivative derivation, the four conserved charges, electrostatic energy, all twelve electrostatic-force components in atom order, the signed 3-by-3 raw scaled Born tensor of atom 0 before acoustic correction, and the largest eigenvalue of each optical operator. Use the numerical tolerances stated in the problem.

A single final numeric value of the frequency shift wrapped in <final_answer>...</final_answer> tags.

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

01_moment_jets

Goal
----
Compute periodic moment tensors of ranks zero, one and two and their first and second Cartesian derivatives.

```python
def moment_jets(positions, box, species, pair_weights, cutoff):
    """Compute periodic moment tensors and their Cartesian derivatives through order two.
    
    Parameters
    ----------
    positions : (N,3) real array, N >= 1
        Cartesian coordinates; wrap each coordinate into [0, box[a]).
    box : (3,) positive real array
        Orthorhombic cell lengths; cell is fixed during differentiation.
    species : (N,) integer array
        Type indices in [0,S).
    pair_weights : (S,S) real array
        Directed type weights c[central_type, neighbor_type].
    cutoff : positive float
        Include every image d=r_j+box*n-r_i with 0<|d|<cutoff,
        n in Z^3. Include nonzero self images; exclude (i=j,n=0).
    
    Define w(d)=c*(1-|d|/cutoff)^4 inside the cutoff and zero outside.
    M0_i=sum w; M1_i=sum w*d; M2_i=sum w*outer(d,d).
    The coordinate order is x=(r_0x,r_0y,r_0z,r_1x,...).
    Image labels stay fixed under differentiation. Nonzero self-image
    vectors do not change when their atom moves. At |d|=cutoff all returned
    contributions vanish. No minimum-image replacement of the image sum.
    
    Returns
    -------
    m : (N,13) float array
        Concatenate M0, M1, and row-major M2.
    g : (N,13,3*N) float array
        First Cartesian derivatives of m.
    h : (N,13,3*N,3*N) float array
        Second Cartesian derivatives of m, with no factorial normalization.
    
    Raises
    ------
    ValueError
        If an array has the wrong shape, entries are nonfinite, box or cutoff
        is nonpositive, species is not integer-valued or out of range, or
        distinct atoms have periodic separation below 1e-10.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (m, g, h)
```

### Step 2

02_invariant_jets

Goal
----
Construct the five scalar moment invariants and propagate their Cartesian gradients and Hessians.

```python
def invariant_jets(moments, first, second):
    """Contract moment jets into five scalar invariants and their first two derivatives.
    
    Parameters
    ----------
    moments : (N,13) real array
        M0, M1[3], M2[3,3] flattened row-major; N>=1.
    first : (N,13,D) real array
        Derivatives with respect to D>=1 variables.
    second : (N,13,D,D) real array
        Unnormalized second derivatives.
    
    Return invariants in this order: M0, M0^2, M1 dot M1,
    sum_ab M2_ab^2, sum_ab M1_a*M2_ab*M1_b.
    Differentiate every factor, including both occurrences of M1 and M2.
    The inputs are arbitrary finite jets; do not impose symmetry on M2.
    
    Returns
    -------
    f : (N,5) float array
        Invariant values.
    df : (N,5,D) float array
        First derivatives.
    ddf : (N,5,D,D) float array
        Second derivatives.
    
    Raises
    ------
    ValueError
        If shapes do not match the stated contract or any entry is nonfinite.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (f, df, ddf)
```

### Step 3

03_redistributed_charge_jets

Goal
----
Evaluate the local raw charges, impose the prescribed total charge and propagate first and second coordinate derivatives through the redistribution map.

```python
def redistributed_charge_jets(features, first, second, species, bias, coefficients, softness, total_charge):
    """Evaluate environment charges and redistribute their residual with type softness.
    
    Parameters
    ----------
    features : (N,5) real array
    first : (N,5,D) real array
    second : (N,5,D,D) real array
        Feature values and unnormalized derivatives; N,D>=1.
    species : (N,) integer array
        Indices in [0,S).
    bias : (S,) real array
    coefficients : (S,5) real array
    softness : (S,) positive real array
        Geometry-independent redistribution weights.
    total_charge : finite float
        Prescribed charge, held constant during differentiation.
    
    For y_i=bias[z_i]+sum_k coefficients[z_i,k]*features[i,k], define
    s_i=softness[z_i], w_i=s_i/sum_j s_j and
    q_i=y_i+w_i*(total_charge-sum_j y_j).
    Return q and its full first and second derivatives. The sums run over
    all atoms, not over species. Do not minimize a QEq energy.
    
    Returns
    -------
    q : (N,) float array
    jq : (N,D) float array
    hq : (N,D,D) float array
    
    Raises
    ------
    ValueError
        If shapes do not match, entries are nonfinite, softness is nonpositive,
        or species is not integer-valued or is out of range.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (q, jq, hq)
```

### Step 4

04_ewald_kernel_jets

Goal
----
Construct the finite periodic Ewald kernel and its first and second Cartesian derivatives.

```python
def ewald_kernel_jets(positions, box, alpha, real_extent, reciprocal_extent):
    """Build a point-charge Ewald matrix and its first two Cartesian derivatives.
    
    Parameters
    ----------
    positions : (N,3) real array, N>=1
        Wrap each coordinate into [0,box[a]); hold cell and image labels fixed.
    box : (3,) positive real array
    alpha : positive float
        Ewald inverse length. Coulomb prefactor is exactly 1.
    real_extent : nonnegative integer
        Include every integer image n in [-real_extent,real_extent]^3.
    reciprocal_extent : (3,) nonnegative integer array
        Include every m with -M[a]<=m[a]<=M[a], except m=(0,0,0).
    
    Let V=prod(box), k=2*pi*m/box and d_ij,n=r_i-r_j+box*n.
    K_ij=sum_n' erfc(alpha*|d_ij,n|)/|d_ij,n|
     + (4*pi/V)*sum_k exp(-k^2/(4*alpha^2))*cos(k dot (r_i-r_j))/k^2
     - (2*alpha/sqrt(pi))*delta_ij - pi/(alpha^2*V).
    The prime removes only i=j,n=0. Both +/- reciprocal modes occur.
    The last term is the uniform neutralizing-background convention and
    occurs in every entry. The self term is diagonal. Neither has coordinate
    derivatives. No real-space distance cutoff is applied. These finite sums
    are the definition of the task; no adaptive convergence changes.
    Coordinate index a=3*i+component. Charge values are not arguments.
    
    Returns
    -------
    k : (N,N) float array
    kg : (N,N,3*N) float array
    kh : (N,N,3*N,3*N) float array
        Value, first derivative and unnormalized second derivative of K.
    
    Raises
    ------
    ValueError
        If shapes do not match, inputs are nonfinite, box or alpha is
        nonpositive, extents are not nonnegative integers, or distinct atoms
        have periodic separation below 1e-10.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (k, kg, kh)
```

### Step 5

05_electrostatic_jet

Goal
----
Compute electrostatic energy, forces and the Cartesian Hessian from the charge and Coulomb-kernel derivatives.

```python
def electrostatic_jet(charges, charge_first, charge_second, kernel, kernel_first, kernel_second):
    """Differentiate E(x)=0.5*q(x)^T*K(x)*q(x) through second order.
    
    Parameters
    ----------
    charges : (N,) real array
    charge_first : (N,D) real array
    charge_second : (N,D,D) real array
    kernel : (N,N) real array
    kernel_first : (N,N,D) real array
    kernel_second : (N,N,D,D) real array
        N,D>=1. Kernel arrays are symmetric in their first two indices.
        All second derivatives are unnormalized. Include all derivatives of
        q as well as K. Charge conservation does not make charge derivatives
        vanish. No Hellmann-Feynman stationarity assumption is permitted:
        these charges are a prescribed geometry map, not an energy minimizer.
    
    Returns
    -------
    energy : float
    force : (D,) float array
        Minus the energy gradient.
    hessian : (D,D) float array
        Full energy Hessian, including the term containing charge_second.
    
    Raises
    ------
    ValueError
        If shapes do not match or any entry is nonfinite. Symmetry is a
        precondition and is not separately validated.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (energy, force, hessian)
```

### Step 6

06_periodic_born_tensors

Goal
----
Compute scaled Born tensors from the phase-compensated derivative of the complex periodic dipole.

```python
def periodic_born_tensors(positions, box, charges, charge_first):
    """Compute scaled Born tensors using a phase-compensated periodic dipole.
    
    Parameters
    ----------
    positions : (N,3) real array, N>=1
    box : (3,) positive real array
    charges : (N,) real array
    charge_first : (N,3*N) real array
        J[j,3*l+b]=dq_j/dr_l,b, including the redistribution derivative.
    
    Define P0_a=L_a/(2*pi*i)*sum_j q_j*exp(2*pi*i*r_j,a/L_a).
    Define Z0[l,a,b]=Re(exp(-2*pi*i*r_l,a/L_a)*dP0_a/dr_l,b).
    The axes of Z0 are atom, polarization component, displacement component.
    The phase factor depends on the displaced atom l. There is no volume
    factor in P0, no epsilon_infinity factor in Z0, and no acoustic correction
    at this step. Coordinates can lie outside the primary cell.
    
    Returns
    -------
    z0 : (N,3,3) float array
        Scaled Born tensors. Include the derivative of the phase as well as
        the derivative of every charge. Do not approximate exp by 1+i*phase.
    
    Raises
    ------
    ValueError
        If shapes do not match, inputs are nonfinite, or box is nonpositive.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return z0
```

### Step 7

07_longitudinal_shift

Goal
----
Apply acoustic corrections, construct the longitudinal dynamical matrix and calculate the shift of the highest optical frequency.

```python
def longitudinal_shift(analytic_hessian, born_scaled, masses, box, direction):
    """Return the longitudinal shift of the highest optical frequency in reduced units.
    
    Parameters
    ----------
    analytic_hessian : (3*N,3*N) real array, N>=2
        Analytic Cartesian force constants before acoustic correction. The
        short-range term is already included; no macroscopic nonanalytic
        correction is contained here.
    born_scaled : (N,3,3) real array
        Atom, polarization, displacement indices; Z*=sqrt(epsilon_inf)*Z0.
    masses : (N,) positive real array
    box : (3,) positive real array
    direction : (3,) nonzero real array
        Wavevector direction; normalization is Euclidean.
    
    Electronic screening is isotropic. First set Z=Z0-mean_atoms(Z0), and
    Hc=P*(H+H.T)/2*P with P=I-T*T.T/N and T=tile(I3,(N,1)). These are the
    specified Euclidean minimum-norm acoustic corrections in Cartesian
    coordinates. Then D_ab=Hc_ab/sqrt(m_atom(a)*m_atom(b)).
    For unit direction n, set v[l,b]=sum_a n[a]*Z[l,a,b]/sqrt(m_l).
    NAC=(4*pi/prod(box))*outer(v.ravel(),v.ravel()). The epsilon_inf factor
    cancels; do not divide NAC by epsilon_inf again. Project D and D+NAC
    onto the complement of the three MASS-WEIGHTED translations, whose
    columns are sqrt(m_l)*e_a. For each projected matrix use its largest
    eigenvalue lambda0 or lambdaL. No tracking of a labeled eigenvector.
    
    Returns
    -------
    shift : float
        sqrt(lambdaL)-sqrt(lambda0), with no 2*pi or SI conversion.
    
    Raises
    ------
    ValueError
        If shapes do not match, any input is nonfinite, masses or box are
        nonpositive, direction has zero norm, or either largest optical
        eigenvalue is <=0.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return shift
```

### Step 8

08_solve_phonon_shift

Goal
----
Run the complete neutral periodic charge model and return the longitudinal shift of its highest optical frequency.

```python
def solve_phonon_shift(positions, box, species, pair_weights, cutoff, bias, coefficients, softness, alpha, real_extent, reciprocal_extent, short_hessian, masses, direction):
    """Compute the neutral periodic model's longitudinal optical shift.

    Parameters
    ----------
    positions : (N, 3) real array, N >= 2
        Cartesian coordinates in a fixed orthorhombic cell.
    box : (3,) positive real array
        Cell lengths.
    species : (N,) integer array
        Type indices in [0, S).
    pair_weights : (S, S) real array
        Directed weights indexed by central and neighbor types.
    cutoff : positive float
        Radius of the complete periodic moment sum.
    bias : (S,) real array
        Type-dependent raw-charge intercepts.
    coefficients : (S, 5) real array
        Coefficients of the five invariant features.
    softness : (S,) positive real array
        Fixed type weights for charge redistribution.
    alpha : positive float
        Ewald splitting parameter.
    real_extent : nonnegative integer
        Real-image indices range from -real_extent to +real_extent.
    reciprocal_extent : (3,) nonnegative integer array
        Reciprocal-index bounds along the three cell axes.
    short_hessian : (3*N, 3*N) real array
        Cartesian short-range Hessian. A linear counterterm cancels the
        electrostatic force at the reference geometry and has zero Hessian.
    masses : (N,) positive real array
        Atomic masses.
    direction : (3,) nonzero real array
        Longitudinal propagation direction.

    Call the public functions from steps 1-7 in this order:

    1. moment_jets(positions, box, species, pair_weights, cutoff)
       returns (m, g, h). Include all images with 0 < |d| < cutoff
       and weight c*(1-|d|/cutoff)^4, including nonzero self images.
       Pack moments as M0, M1, and row-major M2.

    2. invariant_jets(m, g, h)
       returns (f, df, ddf). Feature order:
       M0, M0^2, M1.M1, M2:M2, M1.M2.M1.

    3. redistributed_charge_jets(
           f, df, ddf, species, bias, coefficients, softness, 0.0
       )
       returns (q, jq, hq). The total charge is exactly zero.
       The redistribution weight of atom i is
       softness[species[i]] / sum_j softness[species[j]].

    4. ewald_kernel_jets(
           positions, box, alpha, real_extent, reciprocal_extent
       )
       returns (k, kg, kh). Include both signs of reciprocal modes,
       the Ewald self term and the background term.

    5. electrostatic_jet(q, jq, hq, k, kg, kh)
       returns (energy, force, he), retaining all charge and kernel
       derivatives of E = 0.5*q.T@k@q.

    6. periodic_born_tensors(positions, box, q, jq)
       returns z0 with atom, polarization and displacement indices.
       Use the phase-compensated periodic dipole convention of step 6.

    7. longitudinal_shift(
           he + short_hessian, z0, masses, box, direction
       )
       returns the final scalar. Apply the Cartesian acoustic
       corrections and use the mass-weighted optical subspace.
       Isotropic electronic screening cancels from the correction.

    Earlier public step functions are available in the runtime namespace.
    All model parameters are supplied as arguments.

    Returns
    -------
    shift : float
        sqrt(lambda_longitudinal) - sqrt(lambda_analytic), using the
        largest eigenvalue of each optical matrix. Reduced units;
        no 2*pi frequency conversion.

    Raises
    ------
    ValueError
        If required array shapes are invalid, N < 2, an input is
        nonfinite, box lengths, cutoff, alpha, softness or masses are
        nonpositive, species indices are nonintegral or out of range,
        image extents are negative or nonintegral, distinct atoms have
        periodic separation below 1e-10, direction has zero norm,
        short_hessian has the wrong shape or nonfinite entries, an
        intermediate jet is invalid under the called step's contract,
        or either largest optical eigenvalue is nonpositive.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return shift
```
