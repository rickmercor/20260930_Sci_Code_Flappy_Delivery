# Physics-Condensed_Matter_Physics-54

## Background

Geometry and convention. Let a=sqrt(3)/2, a1=(a,1/2), a2=(0,1), rA=(0,0), rB=(1/sqrt(3),0), and Acell=a. A physical site has position x*a1+n*a2+r_alpha, with x=0,...,nx-1 and n any integer. Interleave (x,A),(x,B). The cell-periodic partial Fourier transform is psi_(x,alpha)(k)=sum_n exp(-i*k*n)*psi_(x,n,alpha), with k in [0,2*pi). Thus a directed hop from (x,alpha,n) to (x+dx,beta,n+m) contributes exp(-i*k*m) to H[target,source]. The x direction is open: omit bonds leaving its range. No minimum-image x coordinate is used.

Model. Use onsite energies delta_x+0.2 on A and delta_x-0.2 on B. Directed nearest-neighbor bonds A(x,n)->B(x+dx,n+m) have (dx,m)=(0,0),(-1,0),(-1,1), amplitude 1. Directed next-nearest-neighbor bonds on each sublattice have (dx,m)=(1,0),(0,-1),(-1,1), amplitude +i/3 on A and -i/3 on B. Add each directed bond and its Hermitian conjugate exactly once, including the diagonal-in-block contributions with dx=0,m=-1. These explicit directions fix the Haldane chirality.

Generate scalar stripe disorder, identical on both orbitals in each cell: s_0=54; s_(x+1)=(1664525*s_x+1013904223) mod 2**32; delta_x=0.4*(s_(x+1)/2**32-1/2). Do not recenter the sample. This generator is a task reproducibility convention.

Magnetic coupling. Use the dimensionless Peierls convention exp[-2*pi*i*integral A dot dr], with A=(0,f*(X-x0)/Acell), x0=0. Integrate along the straight physical bond, including the y offsets from a1. In the Bloch block, X_(x,A)=x*a, X_(x,B)=x*a+1/sqrt(3), Y_(x,alpha)=x/2. For a bond with integer y translation m, its extra multiplier is exp[-2*pi*i*f/Acell*((Xs+Xt)/2-x0)*(m+Yt-Ys)]. Flux f is per primitive cell, not per plaquette chosen independently of this convention.

Occupation and density. At each momentum and flux, P_f(k) is the spectral projector of H_f(k) onto E<mu, mu=0.08. Equality is excluded; rank can change with momentum and magnetic field. The cell density n_f(x)=(1/(2*pi))*integral_0^(2*pi) sum_alpha P_f(k)_(x alpha,x alpha) dk uses continuous momentum.

Fermi crossings. The occupation is discontinuous at the interior Fermi crossings, the momenta k in (0, 2*pi) where a band of H_f(k) equals mu; evaluate n_(+f) and n_(-f) with the same mu and disorder.

Local Chern marker. The physical-coordinate definition is c_x=(2*pi*i/Acell)*sum_alpha <x,alpha|P[-i[X,P],-i[y,P]]|x,alpha>, where the expectation includes the transverse average and physical y=n+Y. Express this definition in the cell-periodic partial Fourier transform specified above. At zero field sample k_j=2*pi*j/ny, j=0,...,ny-1, and differentiate the sampled projector in momentum with the source method's prescription for this hybrid position-momentum marker. Take the real part after the commutator contraction, sum both orbital diagonal entries, and divide the momentum sum by ny. Retain the physical Y embedding in transforming -i[y,P]; it is not exhausted by the integer cell coordinate n. Use ordinary open-x position differences.

Comparison. Define cS_x=(n_(+f)(x)-n_(-f)(x))/(2*f), f=0.003. Let d_x=cS_x-c_x. B={0,1,12,13} and I={2,...,11}. Then R_B=sqrt(sum_(x in B) d_x**2/4), R_I=sqrt(sum_(x in I) d_x**2/10), and the requested answer is Delta=R_B-R_I. Neither marker is constrained by hand to an integer, zero mean, or a chosen electron count. All geometry, disorder, flux and finite-size parameters here are benchmark choices; do not identify these numerical values with a published figure.

## Problem

Compute the difference between boundary and interior RMS discrepancies of the hybrid local Chern marker and the finite-flux Streda marker for the disordered Haldane ribbon specified below.
Use nx=14 open-direction cells, ny=129 uniform momentum samples for the zero-field Chern marker, the seed-54 stripe disorder, chemical potential mu=0.08, and magnetic fluxes f=+0.003 and -0.003 in flux-quantum units per primitive cell.
For the density response, evaluate the continuous transverse-momentum integral to an absolute accuracy better than 1e-9 per cell; do not replace this integral by the marker's finite momentum grid.
Derive the marker in the stated cell-periodic Bloch convention, retaining physical orbital positions and the primitive-cell area, and hold the chemical potential fixed across magnetic fields.
Use the first two and last two cells as the boundary set and the other ten cells as the interior set.
Report the interior Fermi-crossing momenta at each signed flux within absolute error 1e-6 radians, the spatial mean of the Streda marker and each RMS discrepancy within absolute error 1e-5, and their boundary-minus-interior difference.
Explain with literature support why the density response uses fixed chemical potential, then interpret the finite-field boundary discrepancy.
Return the boundary-minus-interior RMS difference as one finite dimensionless number within absolute error 1e-5. This is the specified finite-ribbon, mixed-quadrature observable, not a Chern number or a zero-field extrapolation.

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

magnetic_ribbon

Goal
----
Construct the complex Hamiltonian blocks from the physical geometry, directed hopping list, Bloch convention and straight-bond Peierls phase in the interface. Include the Hermitian conjugate of every directed bond once. Treat the scalar stripe disorder equally on both orbitals and truncate only at the open x edges.

```python
def magnetic_ribbon(k: np.ndarray, delta: np.ndarray, flux: float=0.0, center: float=0.0) -> np.ndarray:
    """Construct a magnetic Haldane ribbon with two orbitals per cell.

    Parameters
    ----------
    k : numpy.ndarray
        Nonempty finite real vector of dimensionless y momenta.
    delta : numpy.ndarray
        Finite real vector, length nx>=4, scalar on-site stripe disorder.
    flux : float
        Finite real magnetic flux per primitive cell, in flux-quantum units.
    center : float
        Finite real gauge origin x0 in physical length units.

    Returns
    -------
    numpy.ndarray
        Complex stack (len(k),2*nx,2*nx), basis (x,A),(x,B).
        Geometry: a1=(sqrt(3)/2,1/2), a2=(0,1), rA=(0,0),
        rB=(1/sqrt(3),0), open x and infinite periodic y. On-sites are
        delta[x]+0.2 on A and delta[x]-0.2 on B. NN directed bonds
        A(x,n)->B(x+dx,n+m): (dx,m)=(0,0),(-1,0),(-1,1), t=1.
        NNN directed same-sublattice bonds have (dx,m)=(1,0),(0,-1),
        (-1,1), amplitude i/3 on A and -i/3 on B. Each directed bond
        contributes to H[target,source] and its h.c.; truncate only at x edges.
        Its Bloch multiplier is exp(-i*k*m). Additionally multiply by
        exp[-2*pi*i*flux/Acell*((Xs+Xt)/2-center)*(m+Yt-Ys)],
        Acell=sqrt(3)/2, X=x*Acell+(alpha==B)/sqrt(3), Y=x/2.
        Use physical bond displacement, not just the integer m, in Peierls.

    Raises
    ------
    ValueError
        If k, delta, flux or center violates the stated domain.
    """
    return None
```

### Step 2

fixed_mu_projectors

Goal
----
Build each occupied spectral projector with the strict condition E<mu. Support zero, full and momentum-dependent occupied ranks. The result must be independent of eigenvector phases and rotations within occupied degenerate subspaces.

```python
def fixed_mu_projectors(h: np.ndarray, mu: float) -> np.ndarray:
    """Construct zero-temperature occupied projectors at fixed chemical potential.

    Parameters
    ----------
    h : numpy.ndarray
        Finite Hermitian stack (m,d,d), m>=1, even d>=2, Hermiticity
        absolute error <=1e-10. Occupied rank may differ between blocks.
    mu : float
        Finite real chemical potential, held fixed across fluxes.

    Returns
    -------
    numpy.ndarray
        Complex stack matching h, spectral projector onto eigenvalues E<mu.
        E=mu is excluded. Empty and full occupied subspaces are allowed.
        Return projectors, not eigenvectors; never force rank d/2.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite/non-Hermitian h or invalid mu.
    """
    return None
```

### Step 3

fermi_partition

Goal
----
Find a deterministic partition at the momentum-dependent Fermi crossings. Follow ordered bands through the prescribed scan, bisect sign-changing brackets, retain exact scan-node hits under the specified tolerance, and merge roots before adding the two endpoints. Use the magnetic ribbon builder from step 1.

```python
def fermi_partition(delta: np.ndarray, flux: float, mu: float, scan: int=128) -> np.ndarray:
    """Locate momentum intervals of fixed occupancy by bandwise root bracketing.

    Parameters
    ----------
    delta, flux : numpy.ndarray, float
        Disorder and flux accepted by magnetic_ribbon; gauge center is zero.
    mu : float
        Finite real fixed chemical potential.
    scan : int
        Integer >=8, no booleans; scan+1 equally spaced nodes on [0,2*pi].

    Returns
    -------
    numpy.ndarray
        Sorted real partition including 0 and 2*pi. At scan nodes sort the
        eigenvalues ascending. For each band and adjacent node pair whose
        E-mu values have strictly opposite signs, bisect that ordered band
        until bracket width <=1e-11, then record its midpoint. Also record
        interior scan nodes with abs(E-mu)<=1e-12. Merge interior roots
        separated by <=1e-9, keeping the smaller root; discard roots within
        1e-9 of either endpoint. The task prescribes this detector; it does
        not promise to find tangent roots or multiple roots inside one bin.

    Raises
    ------
    ValueError
        Invalid scan or mu, or invalid magnetic_ribbon parameters.
    """
    return None
```

### Step 4

partition_density

Goal
----
Integrate the local occupied density over every supplied momentum interval using a separate Gauss-Legendre rule. Reconstruct the magnetic Hamiltonian and its fixed-mu projector at the quadrature nodes. Normalize by 2*pi and sum, rather than average, the two orbital densities.

```python
def partition_density(delta: np.ndarray, flux: float, mu: float, cuts: np.ndarray, order: int) -> np.ndarray:
    """Integrate the occupied cell density separately on each momentum interval.

    Parameters
    ----------
    delta, flux, mu : numpy.ndarray, float, float
        Model inputs accepted by earlier functions; gauge center zero.
    cuts : numpy.ndarray
        Finite real strictly increasing vector of length>=2, endpoints
        0 and 2*pi to absolute tolerance 1e-10, used as supplied.
    order : int
        Gauss-Legendre order >=4 per interval; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector (nx,), integral of sum_alpha P_(x alpha,x alpha)(k)
        over 0..2*pi, divided by 2*pi. Map an order-point Gauss-Legendre
        rule independently to every interval; use E<mu at every node.
        Do not enforce a fixed electron count or sample across a cut.

    Raises
    ------
    ValueError
        Invalid cuts/order or invalid earlier-function inputs.
    """
    return None
```

### Step 5

refine_density

Goal
----
Starting from the supplied density estimate, double the per-interval Gauss-Legendre order until the stated maximum-component convergence condition holds. Use step 4 for each refinement. Preserve the partition and compare consecutive summed densities; report nonconvergence instead of silently returning an unconverged estimate.

```python
def refine_density(delta: np.ndarray, flux: float, mu: float, cuts: np.ndarray, initial: np.ndarray, order: int=8, tol: float=1e-10, max_order: int=256) -> np.ndarray:
    """Refine partition quadrature with the supplied first density estimate.

    Parameters
    ----------
    delta, flux, mu, cuts : numpy.ndarray, float, float, numpy.ndarray
        Same domains as partition_density. Cuts remain fixed.
    initial : numpy.ndarray
        Finite real vector matching delta, the density at the given order.
        Treat this as an input estimate; do not discard or recompute it.
    order, max_order : int
        Integers, order>=4 and max_order>=2*order; booleans excluded.
    tol : float
        Finite positive real maximum-component convergence threshold.

    Returns
    -------
    numpy.ndarray
        Refined real density (nx,). Double order, recompute the integral,
        and return the new density at the first max(abs(new-old))<=tol.
        Test the candidate at max_order if reached by doubling. Never use
        an order larger than max_order; if none converges raise ValueError.

    Raises
    ------
    ValueError
        Invalid parameters/estimate, nonconvergence, or earlier input errors.
    """
    return None
```

### Step 6

embedded_chern

Goal
----
Evaluate the open-ribbon local Chern marker from the sampled projector in the cell-periodic Bloch basis. Differentiate the sampled projector in momentum with the source method's prescription for the hybrid marker, combine it with the intra-cell y embedding, use physical open-x coordinates, and retain the primitive-cell-area normalization.

```python
def embedded_chern(p: np.ndarray) -> np.ndarray:
    """Evaluate the open-x marker in the cell-periodic Bloch gauge.

    Parameters
    ----------
    p : numpy.ndarray
        Finite complex stack (ny,2*nx,2*nx), ny>=5, nx>=4.
        Samples at k_j=2*pi*j/ny; general matrices allowed.

    Returns
    -------
    numpy.ndarray
        Real marker per cell (nx,), including orbital embedding and area.
        X_(x,A)=x*sqrt(3)/2, X_(x,B)=X_(x,A)+1/sqrt(3);
        Y_(x,alpha)=x/2; Acell=sqrt(3)/2. Differentiate the sampled P in k
        with the source method's prescription for this hybrid marker.
        D=partial_k P-i[Y,P], A=-i[X,P], and return
        Re[(2*pi*i/(ny*Acell))*sum_(j,alpha) diag(P_j[A_j,D_j])].
        X differences are ordinary open-ribbon differences, never wrapped.

    Raises
    ------
    ValueError
        Invalid shape or nonfinite input.
    """
    return None
```

### Step 7

boundary_discrepancy

Goal
----
Form the centered finite-flux density response and compare it with the supplied zero-field Chern profile. Return its spatial mean and the boundary/interior RMS errors, with both ribbon edges in the boundary set. Keep the boundary-minus-interior sign fixed.

```python
def boundary_discrepancy(c: np.ndarray, nplus: np.ndarray, nminus: np.ndarray, flux: float, edge: int=2) -> np.ndarray:
    """Compare local Chern and symmetric finite-flux Streda markers.

    Parameters
    ----------
    c, nplus, nminus : numpy.ndarray
        Equal finite real vectors (nx,). Densities correspond to +flux/-flux
        at identical chemical potential and geometric gauge convention.
    flux : float
        Finite positive real flux magnitude in flux-quantum units per cell.
    edge : int
        Positive integer, 2*edge<nx; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector [mean(cS), R_edge, R_bulk, R_edge-R_bulk], where
        cS=(nplus-nminus)/(2*flux). R is RMS of cS-c over the first/last
        edge cells or the remaining cells respectively. Count both edges.
        Do not sum or average the two densities before differencing.

    Raises
    ------
    ValueError
        Invalid vectors, flux or edge width.
    """
    return None
```

### Step 8

run_ribbon_comparison

Goal
----
Generate the deterministic scalar stripe disorder and execute all seven preceding functions, calling each directly and using its returned result. Compute the Chern marker on the uniform zero-field momentum grid and each magnetic density by continuous momentum quadrature. Return the boundary-minus-interior RMS discrepancy.

```python
def run_ribbon_comparison(nx: int=14, ny: int=129, seed: int=54, flux: float=0.003, mu: float=0.08, edge: int=2) -> float:
    """Run all seven earlier steps for the ribbon marker comparison.

    Parameters
    ----------
    nx, ny : int
        Cell count>=6 and marker momentum count>=5, booleans excluded.
    seed : int
        Unsigned 32-bit seed, booleans excluded. For each x, update
        s=(1664525*s+1013904223) mod 2**32 then delta[x]=.4*(s/2**32-.5).
    flux, mu, edge : float, float, int
        Positive flux, finite mu, and edge width accepted by earlier steps.

    Returns
    -------
    float
        R_edge-R_bulk. Compute the zero-flux marker on k_j=2*pi*j/ny.
        For each signed flux, obtain cuts with scan=128, start partition
        quadrature at order 8, then refine with tol=1e-10,max_order=256.
        Density integrates continuous ky; marker retains the stated ny grid.
        All seven earlier step functions must be called directly and their outputs
        used. Keep mu fixed at all fluxes; do not impose half filling.

    Raises
    ------
    ValueError
        Invalid parameters, nonconvergent integration or earlier input errors.
    """
    return 0.0
```
