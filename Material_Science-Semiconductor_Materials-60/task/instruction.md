# Material_Science-Semiconductor_Materials-60

## Background

Microscopic screening couples reciprocal-space density components. The screened interaction sets the exciton eigenvectors as well as their energies. Optical weight also depends on the local-orbital velocity and its intracell position term.

## Problem

Calculate the fraction of the exciton oscillator measure in the energy window [1.10,2.20] for the supplied two-orbital semiconductor. Use microscopic static screening from the main paper, the local-orbital velocity construction of the Bloch-matrix-element paper, and the dipole-weighted spectral-measure formulation of the optical-absorption paper.

Keep every supplied reciprocal vector and both interband transition directions in the static polarizability. Invert the full dielectric matrix. Assemble the direct Tamm-Dancoff exciton Hamiltonian with unwrapped momentum transfers. At zero transfer, use the specified circular head average, zero wings and screened nonzero-G body. Compute the optical dipole in the cell gauge, including the intracell position contribution. Integrate normalized Lorentzians of half-width 0.065 over the requested window and divide by the total oscillator weight.

In the reasoning, derive the density vertex, polarizability, symmetrized dielectric inverse, direct exciton kernel and optical dipole. State how Bloch-phase changes act on the exciton matrix and dipole. Give the averaged zero-transfer head, the macroscopic dielectric function at q=(0.37,-0.21), the lowest three exciton energies, the lowest binding energy relative to the sampled direct gap, the total oscillator weight, and the oscillator weight of the lowest-energy exciton. Explain the zero-transfer approximation, the distinction between epsilon_00 and the macroscopic dielectric function, and why unweighted eigenvalue counting gives a different observable. Identify the retained part of each supporting method, including the choice of the positive Tamm-Dancoff branch.

Inputs and scope

Use make_inputs() below. It is also supplied in fixture.py. These are synthetic tight-binding inputs. Lattice constant, hbar and cell area are 1. The finite k/G grids, unwrapped transfers and circular quadrature define the target. The Hamiltonian and numerical conventions are stated below.

def make_inputs():
    """Synthetic two-orbital square-lattice semiconductor, a=hbar=cell area=1."""
    import numpy as np
    n=7;axis=2*np.pi*(np.arange(n)-(n-1)/2)/n
    momentum=np.array([(x,y) for x in axis for y in axis])
    return dict(momentum=momentum,
                reciprocal=2*np.pi*np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),
                positions=np.array([[0.,0.],[.23,.31]]),
                parameters=np.array([.92,.31,.24,.12,.43,.54,.29]),
                coupling=.48,spin=2.,radial_order=12,angular_order=20,
                polarization=np.array([1.,.7j]),window=np.array([1.10,2.20]),broadening=.065)

Numerical conventions

parameters=(m,bx,by,a,tx,ty,tz) defines H=dx*sigma_x+dy*sigma_y+dz*sigma_z, with dx=a+tx*cos(kx)+0.19*cos(ky), dy=ty*sin(kx)+tz*sin(ky), and dz=m+bx*(1-cos(kx))+by*(1-cos(ky)). Eigenvectors are columns, ordered valence then conduction. Use the orthonormal point-orbital cell gauge, positions in lattice units, occupations fv=1 and fc=0, and spin/Nk for the static band sum. The bare 2D Coulomb component is 2*pi*coupling/|q+G|. The first reciprocal vector is zero. No exchange term is included.

For q=0, use a circle of radius sqrt(4*pi/Nk), which has area (2*pi)^2/Nk. Integrate its head with radial_order Gauss-Legendre nodes on [0,R] and angular_order equally spaced angles 2*pi*j/angular_order. Recompute the full static response and screened interaction at each node. Set zero-transfer wings to zero; compute the screened nonzero-G body from the q=0 body polarizability. This direct head average replaces a fitted small-q dielectric model.

Use q=k_i-k_j without folding. The direct kernel has weight 1/Nk. Equal transfers may be cached by rounding components to 13 decimals and evaluated at the first unrounded transfer. Normalize polarization by its Euclidean norm. In the spectral measure, use the positive Tamm-Dancoff branch alone, one unit-area Lorentzian on the entire real energy line for each eigenstate, and broadening as its half-width. The denominator is the sum of all oscillator weights.

Report these numerical diagnostics to at least eight digits after the decimal point. The absolute tolerance for each diagnostic is 1e-7.

## Output format

Return one finite decimal inside <final_answer>, followed immediately by <reasoning> with a concise numbered derivation. Absolute tolerance: 1e-7. No article links are required in the answer.

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

01_band_structure

Goal
----
Diagonalize the periodic two-orbital Bloch Hamiltonian and its derivatives.

```python
def band_structure(momentum, parameters):
    """Diagonalize the periodic two-orbital Bloch Hamiltonian and its derivatives.

    momentum is finite real (nk,2), nk>=1, in inverse lattice units.
    parameters is finite real (7,)=(m,bx,by,a,tx,ty,tz).
    H=dx*sigma_x+dy*sigma_y+dz*sigma_z, where
    dx=a+tx*cos(kx)+0.19*cos(ky), dy=ty*sin(kx)+tz*sin(ky),
    dz=m+bx*(1-cos(kx))+by*(1-cos(ky)). Energy unit is fixed by parameters;
    lattice constant and hbar are 1. Use ascending energies (valence,conduction).
    dH/dkx uses (-tx*sin(kx),ty*cos(kx),bx*sin(kx));
    dH/dky uses (-0.19*sin(ky),tz*cos(ky),by*sin(ky)).

    Returns
    -------
    result
        Tuple (energies real (nk,2), eigenvectors complex (nk,2,2) in columns,
        H complex (nk,2,2), dH complex (nk,2,2,2), derivative axis second).
        Eigenvector phases are arbitrary; only physical invariants are specified.

    Raises
    ------
    ValueError
        For non-real/nonfinite inputs, wrong shapes, or a sampled gap <=1e-10.
    """
    return result
```

### Step 2

02_density_vertices

Goal
----
Evaluate point-orbital density vertices in the cell-periodic orbital gauge.

```python
def density_vertices(left, right, transfer, reciprocal, positions):
    """Evaluate point-orbital density vertices in the cell-periodic orbital gauge.

    left and right are finite complex (n,orb) coefficient arrays with n,orb>=1.
    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1,
    positions finite real (orb,2). All coefficients use the same orbital basis.
    I_G=sum_l conj(left_l)*right_l*exp[-i*(transfer+G) dot positions_l].
    Coefficients need not be normalized. Lattice constant=1.

    Returns
    -------
    result
        Complex (n,ng) array of density vertices.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, or non-real geometric arrays.
    """
    return result
```

### Step 3

03_static_polarizability

Goal
----
Compute the static independent-particle polarizability, including both band directions.

```python
def static_polarizability(momentum, transfer, reciprocal, positions, parameters, spin=2.0):
    """Compute the static independent-particle polarizability, including both band directions.

    momentum/parameters follow band_structure; transfer, reciprocal and
    positions follow density_vertices, with positions shape (2,2).
    spin is a finite positive real scalar. At zero temperature f_v=1,f_c=0.
    Evaluate bands at k and k+q without folding q or shifting reciprocal indices.
    chi_GG'=spin/nk * sum_{k,n!=m} [(f_n-f_m)/(E_nk-E_m,k+q)]
    * I_G(nk,mk+q)*conj(I_G'(nk,mk+q)). Include (n,m)=(0,1),(1,0).
    Cell area=1. Do not replace this matrix by its head or use a factor-two
    shortcut for the two transition directions.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) polarizability in inverse energy units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, nonpositive/non-real/nonfinite spin,
        or an interband energy denominator with absolute value <=1e-10.
    """
    return result
```

### Step 4

04_screened_interaction

Goal
----
Invert the symmetrized microscopic dielectric matrix at nonsingular momentum.

```python
def screened_interaction(transfer, reciprocal, polarizability, coupling):
    """Invert the symmetrized microscopic dielectric matrix at nonsingular momentum.

    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1.
    polarizability is finite Hermitian complex (ng,ng), atol=1e-11,rtol=0.
    coupling>0 is finite real. Cell area=1; v_G=2*pi*coupling/|q+G|.
    E=I-sqrt(v)*chi*sqrt(v), W=sqrt(v)*E^-1*sqrt(v).
    Require every |q+G|>1e-12 and E positive definite. The first reciprocal
    vector is the head for callers using a macroscopic dielectric function.
    Return E^-1 as well as W; epsilon_M=1/(E^-1)[0,0], not E[0,0].

    Returns
    -------
    result
        Tuple (W complex (ng,ng), inverse_dielectric complex (ng,ng)).

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-real geometry/coupling,
        nonpositive coupling, non-Hermitian chi, singular q+G, or nonpositive E.
    """
    return result
```

### Step 5

05_gamma_interaction

Goal
----
Regularize the zero-transfer cell by a circular microscopic head average.

```python
def gamma_interaction(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Regularize the zero-transfer cell by a circular microscopic head average.

    Inputs follow static_polarizability and screened_interaction. reciprocal[0]
    must be zero to atol=1e-14 and all other vectors nonzero (>1e-12).
    radial_order>=2 and angular_order>=4 are integers. For nk k-points,
    replace their reciprocal cell of area (2*pi)^2/nk by a circle of radius
    R=sqrt(4*pi/nk). Set W00_bar=(1/pi/R^2)*int_0^R r dr int_0^{2pi} W00(q)dtheta.
    Use radial_order Gauss-Legendre nodes on [0,R] and equally spaced angles
    theta_j=2*pi*j/angular_order. At each node recompute the full chi and W.
    At q=0 use chi's nonzero-G body to obtain the screened body. Set wings
    W0G=WG0=0 for G!=0. This benchmark uses a direct circular average, not a
    fitted linear small-q dielectric approximation.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) regularized zero-transfer interaction.

    Raises
    ------
    ValueError
        For invalid delegated inputs, invalid integer quadrature orders,
        a nonzero first reciprocal vector, or an additional zero reciprocal vector.
    """
    return result
```

### Step 6

06_exciton_hamiltonian

Goal
----
Assemble the direct Tamm-Dancoff exciton Hamiltonian on the supplied k-grid.

```python
def exciton_hamiltonian(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Assemble the direct Tamm-Dancoff exciton Hamiltonian on the supplied k-grid.

    Inputs follow gamma_interaction. Require distinct k-points, no coincident
    nonzero-transfer q+G singularities, and an inversion-symmetric reciprocal
    set containing zero first. k differences are unwrapped; do not fold them.
    Band order is v=0,c=1. For q=k_i-k_j define
    a_G=sum_l conj(C_c,i,l)*C_c,j,l*exp[+i*(q+G) dot tau_l],
    b_G=sum_l conj(C_v,i,l)*C_v,j,l*exp[+i*(q+G) dot tau_l].
    D_ij=(a @ W(q) @ conj(b))/nk; H_ij=(Ec_i-Ev_i)*delta_ij-D_ij.
    At i=j use gamma_interaction; elsewhere recompute full static RPA and W.
    Cache equal transfers using component rounding to 13 decimal places;
    evaluate a cache entry at its first unrounded transfer. Omit exchange.
    Check Hermiticity (atol=2e-10,rtol=0); then symmetrize roundoff only.

    Returns
    -------
    result
        Complex Hermitian (nk,nk) direct exciton Hamiltonian, in energy units.
        Band phases may change entries; compare gauge-invariant quantities.

    Raises
    ------
    ValueError
        For invalid delegated inputs, duplicate k-points at 13-decimal precision,
        an inversion-asymmetric/duplicate reciprocal set, or failed Hermiticity.
    """
    return result
```

### Step 7

07_optical_dipole

Goal
----
Compute interband dipoles with the intracell position contribution to velocity.

```python
def optical_dipole(momentum, positions, parameters, polarization):
    """Compute interband dipoles with the intracell position contribution to velocity.

    momentum and parameters follow band_structure. positions is finite real
    (2,2); polarization is a finite nonzero complex (2,) vector, normalized
    internally by its Euclidean norm. In the point-orbital, cell-periodic gauge,
    v_alpha=dH/dk_alpha-i*[diag(tau_alpha),H], with hbar=1.
    d_k=C_c,k^dagger*(sum_alpha polarization_alpha*v_alpha)*C_v,k/(Ec_k-Ev_k).
    The omitted common phase i does not affect oscillator strengths.

    Returns
    -------
    result
        Complex (nk,) dipole vector in lattice-length units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, non-real/nonfinite/wrong-shaped positions,
        or nonfinite, zero or wrong-shaped polarization.
    """
    return result
```

### Step 8

08_spectral_fraction

Goal
----
Integrate the normalized exciton oscillator measure over an energy window.

```python
def spectral_fraction(hamiltonian, dipole, window, broadening):
    """Integrate the normalized exciton oscillator measure over an energy window.

    hamiltonian is finite Hermitian complex (n,n), n>=1; dipole is finite
    nonzero complex (n,). window=(lo,hi) is finite real with lo<hi;
    broadening>0 is finite real. Let H A_s=E_s A_s, ||A_s||=1,
    f_s=|A_s^dagger dipole|^2. Use the normalized Lorentzian of half-width
    broadening on the entire real energy line. The window fraction is
    sum_s f_s*[atan((hi-E_s)/b)-atan((lo-E_s)/b)]/(pi*sum_s f_s).
    Keep complex conjugation and sum incoherently over different eigenstates.

    Returns
    -------
    result
        Float oscillator fraction in [0,1] up to roundoff.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-Hermitian H (atol=2e-10,
        rtol=0), zero dipole, complex/invalid window or nonpositive broadening.
    """
    return result
```

### Step 9

09_solve

Goal
----
Return the exciton oscillator fraction for the supplied semiconductor model.

```python
def solve(data):
    """Return the exciton oscillator fraction for the supplied semiconductor model.

    data is the make_inputs() dictionary with keys momentum, reciprocal,
    positions, parameters, coupling, spin, radial_order, angular_order,
    polarization, window, broadening. Every value follows the preceding
    contracts. Assemble the direct microscopic-screened Tamm-Dancoff matrix,
    compute the position-corrected dipole, and integrate the oscillator measure.
    The finite k/G grids and circle quadrature are fixed benchmark definitions.

    Returns
    -------
    result
        Float normalized oscillator fraction in the prescribed energy window.

    Raises
    ------
    ValueError
        For a non-dictionary input, absent required key, or any invalid delegated input.
    """
    return result
```
