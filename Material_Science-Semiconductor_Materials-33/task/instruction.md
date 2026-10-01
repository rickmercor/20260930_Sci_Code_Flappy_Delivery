# Material_Science-Semiconductor_Materials-33

## Background

Excitons are correlated electron-hole excitations that influence the absorption
and emission of semiconductors. In atomically thin materials, confinement and
the spatial distribution of electronic charge make screening depend on length
scale and direction. The dielectric environment and the material's internal
structure can therefore alter optical excitations substantially.

Microscopic descriptions connect Bloch electronic states to induced charge
and screened interactions. They make it possible to examine how variations
within a unit cell affect optical response, complementing descriptions based
on a single macroscopic dielectric constant. Reliable calculations distinguish
the approximations in the material model from errors introduced by finite
numerical resolution.

## Problem

Determine the thickness curvature of the lowest vertical exciton energy for a constructed two-orbital semiconductor under uniform off-plane dilation, using the symmetric, thickness-averaged point-orbital static-RPA prescription for quasi-two-dimensional excitons and its effective screened-potential approximation. Include all retained local-field components in the direct-channel Tamm–Dancoff problem, with one spinless occupied valence band, one empty conduction band, zero temperature and no exchange.

In the orthonormal lattice Bloch gauge with cell phase \(e^{i\mathbf k\cdot\mathbf R}\), use the following rectangular-lattice Hamiltonian and exact input values (lengths in Å and energies in eV):
\[
\begin{gathered}
x=a_x k_x,\quad y=a_y k_y,\qquad
H(\mathbf k)=\begin{pmatrix}h_z&f\\f^*&-h_z\end{pmatrix},\\
h_z=m+b_x\cos x+b_y\cos y,\qquad
f=t_0+t_xe^{-ix}+t_ye^{-iy}+i\lambda\cos(x-y),\\
(a_x,a_y)=(3.2,4.1),\qquad
(m,b_x,b_y,t_0,t_x,t_y,\lambda)=(1.8,0.2,-0.15,0.45,0.8,0.65,0.27),\\
\boldsymbol\tau_1=(0,0,-1.3),\qquad
\boldsymbol\tau_2=(1.05,0.82,0.9),\qquad d_0=5.5,\qquad C=3.59991137,\\
v(p)=\frac{2\pi C}{a_xa_y p}\quad(p>0),\qquad [C]=\mathrm{eV\,Å},\\
\mathbf k_{ij}=\left(\frac{2\pi(i-4)}{9a_x},\frac{2\pi(j-4)}{9a_y}\right),
\quad i,j=0,\ldots,8,\quad N=81,\\
\mathcal G=\left\{\left(\frac{2\pi r}{a_x},\frac{2\pi s}{a_y}\right):r,s=-1,0,1\right\}.
\end{gathered}
\]
The displayed orbital positions are their values at the evaluation thickness \(d_0\); at variable thickness \(h\), let \(z_a(h)=z_a(d_0)h/d_0\) and the slab span \([-h/2,h/2]\), keeping every in-plane input, the Hamiltonian, the mesh and the reciprocal set fixed, while the strict-2D reference has zero heights and thickness and is independent of \(h\).
The Hamiltonian breaks time-reversal symmetry, and the same equal-weight mesh is used for the response and the electron-hole basis.
For each ordered electron-hole matrix entry use the unreduced transfer \(\mathbf q=\mathbf k-\mathbf k'\), response states at \(\mathbf k+\mathbf q\), the real-space Fourier factors \(e^{+i(\mathbf q+\mathbf G)\cdot\mathbf r}\) and \(e^{-i(\mathbf q+\mathbf G')\cdot\mathbf r'}\) for \(W_{GG'}(\mathbf q)\), and the equal-mesh factor \(1/N\) in the direct kernel.
At exactly \(\mathbf q=0\), set the dielectric head to one and wings to zero, set all screened-potential wings to zero and retain the screened body, and replace the screened head by the source's small-circle regularization in its stated closed form, taking the circle radius as half the nearest nonzero mesh momentum and each screening parameter as the finite-difference momentum slope of the inverse dielectric head from Gamma to the corresponding mesh-adjacent axis transfer, recalculated at each \(h\).

Compute
\[
\mathcal K=10^3\left.\frac{d^2}{dh^2}
\big[E_{\rm Q2D}(h)-E_{\rm 2D}\big]\right|_{h=d_0}
\]
in meV/Å\(^2\) to absolute accuracy 0.002, using analytic thickness derivatives through the screening and lowest-eigenvalue problems, including the change of the exciton eigenstate; numerical thickness differences and continuum extrapolation are excluded.
Support the answer with the minimum direct gap, both lowest excitation energies at \(d_0\), both response heads and macroscopic dielectric values at \(\mathbf q_*=(2\pi/(9a_x),0)\), the Q2D screened head there, the two momentum-screening parameters and regularized heads of both calculations, the first two thickness derivatives of that Q2D screened head and of the Q2D regularized head, and the separate fixed-eigenstate and eigenstate-response contributions to \(\mathcal K\), explaining why the bare regularized head contributes no thickness curvature.
State the response and interaction thickness averages, the effective screened-potential approximation, the direct matrix element and the head closed form with their source locations, and explain how you obtain the inverse-dielectric and eigenstate derivatives.

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

solve_bands

Goal
----
Diagonalize the specified two-orbital, spinless semiconductor model.

```python
def solve_bands(
    k_points: "np.ndarray", lattice: "np.ndarray", model: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Return band energies and orthonormal eigenvectors in the lattice gauge.

    Parameters
    ----------
    k_points : "np.ndarray"
        Finite real array (K, 2), K >= 1, of Cartesian momenta in inverse
        angstroms; momenta may lie outside the first Brillouin zone.
    lattice : "np.ndarray"
        Positive finite real array (2,), the rectangular lattice lengths
        (a_x, a_y) in angstroms.
    model : "np.ndarray"
        Finite real array (7,) containing (m, b_x, b_y, t_0, t_x, t_y,
        lambda), all in eV, with m > abs(b_x) + abs(b_y).
        Set x = a_x*k_x, y = a_y*k_y, h_z = m + b_x*cos(x) + b_y*cos(y),
        and f = t_0 + t_x*exp(-i*x) + t_y*exp(-i*y)
        + i*lambda*cos(x-y). The Hamiltonian is [[h_z, f], [f*, -h_z]].

    Returns
    -------
    energies : "np.ndarray"
        Real array (K, 2) in eV, ordered valence then conduction.
    vectors : "np.ndarray"
        Complex array (K, 2, 2), indexed by momentum, orbital, band.
        Columns are normalized eigenvectors; any column phases are valid.

    Raises
    ------
    ValueError
        If the shapes, finiteness, real-valuedness, positive lattice lengths,
        or strict mass bound above are violated.
    """
    return (0.0, 0.0)
```

### Step 2

thickness_averages

Goal
----
Evaluate distinct thickness averages for symmetric Q2D screening.

```python
def thickness_averages(
    magnitudes: "np.ndarray", heights: "np.ndarray", thickness: float
) -> "np.ndarray":
    """Return dimensionless averages of the off-plane Coulomb exponential.

    Parameters
    ----------
    magnitudes : "np.ndarray"
        Finite nonnegative real array (G,), G >= 1, of |q+G| in inverse
        angstroms.
    heights : "np.ndarray"
        Finite real array (A,), A >= 1, of orbital heights in angstroms.
        Each lies in [-thickness/2, thickness/2].
    thickness : float
        Finite nonnegative slab thickness in angstroms. At zero thickness,
        all heights must be zero and all returned averages equal one.

    Returns
    -------
    averages : "np.ndarray"
        Real array (G, A+1). Column a < A is the uniform slab average of
        exp(-p*abs(z-heights[a])) over z. The last column is the average of
        exp(-p*abs(z-z_prime)) over two independent uniform slab positions.
        At p=0 both kinds of average are exactly one. Results must remain
        accurate for small p*thickness, including values below 1e-8.

    Raises
    ------
    ValueError
        If inputs violate the shapes, finiteness, real-valuedness,
        nonnegativity, or allowed height interval specified above.
    """
    return 0.0
```

### Step 3

density_vertices

Goal
----
Construct point-orbital density vertices with symmetric thickness weighting.

```python
def density_vertices(
    left_vectors: "np.ndarray",
    right_vectors: "np.ndarray",
    transfer: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    orbital_averages: "np.ndarray",
) -> "np.ndarray":
    """Evaluate all band-pair density vertices at a fixed transfer.

    Parameters
    ----------
    left_vectors, right_vectors : "np.ndarray"
        Finite complex arrays (K, 2, 2), K >= 1, indexed by momentum,
        orbital and band, for the states at k and k+transfer, respectively.
        General finite coefficients are supported; physical columns are
        orthonormal Bloch eigenvectors. Inputs are not modified.
    transfer : "np.ndarray"
        Finite real Cartesian vector (2,) in inverse angstroms.
    g_vectors : "np.ndarray"
        Finite real array (G, 2), G >= 1, in inverse angstroms; order is kept.
    centres : "np.ndarray"
        Finite real array (2, 2) of in-plane orbital positions in angstroms.
    orbital_averages : "np.ndarray"
        Finite real array (G, 2), with entries in [0, 1], containing the
        one-coordinate averages from thickness_averages. Use their positive
        square roots as weights in the point-orbital density matrix element
        of exp(-i*(transfer+G).r).

    Returns
    -------
    vertices : "np.ndarray"
        Complex array (K, G, 2, 2), indexed by k, G, left band, right band.
        Band phases are inherited from the supplied coefficients.

    Raises
    ------
    ValueError
        If any documented shape, finiteness, real-input or weight bound is
        violated.
    """
    return 0.0
```

### Step 4

static_response

Goal
----
Build the static insulating response from both interband channels.

```python
def static_response(
    left_energies: "np.ndarray",
    right_energies: "np.ndarray",
    vertices: "np.ndarray",
    spin_degeneracy: int = 1,
) -> "np.ndarray":
    """Evaluate the zero-temperature static interband spectral sum.

    Parameters
    ----------
    left_energies, right_energies : "np.ndarray"
        Finite real arrays (K, 2), K >= 1, in eV, for k and k+q.
        Column 0 is occupied valence and column 1 empty conduction.
        Each row is strictly increasing, and both cross-momentum
        conduction-minus-valence gaps must be strictly positive.
    vertices : "np.ndarray"
        Finite complex array (K, G, 2, 2), G >= 1, from density_vertices.
    spin_degeneracy : int
        Positive integer multiplicity. One describes the spinless benchmark.
        It is independent of the two occupied/empty transition directions.

    Returns
    -------
    response : "np.ndarray"
        Complex Hermitian array (G, G) in inverse eV. Use the occupation
        difference divided by the left-minus-right energy difference for
        each of the two interband directions, an equal 1/K mesh weight,
        and the specified spin multiplicity. Intraband terms vanish.

    Raises
    ------
    ValueError
        If shapes, finite/real energy inputs, energy ordering, cross gaps,
        or the positive integer multiplicity violate this contract.
    """
    return 0.0
```

### Step 5

screened_interaction

Goal
----
Invert the symmetric dielectric matrix and form the screened interaction.

```python
def screened_interaction(
    response: "np.ndarray",
    magnitudes: "np.ndarray",
    area: float,
    coupling: float,
    pair_averages: "np.ndarray",
) -> "np.ndarray":
    """Return the dielectric matrix, its inverse and screened interaction.

    Parameters
    ----------
    response : "np.ndarray"
        Finite complex array (G, G), G >= 1, Hermitian to absolute entrywise
        tolerance 1e-10 and negative semidefinite to eigenvalue tolerance
        1e-10, in inverse eV. Roundoff within that tolerance is symmetrized.
        This is the weighted response from static_response.
    magnitudes : "np.ndarray"
        Finite nonnegative real vector (G,) of |q+G| in inverse angstroms.
        A zero entry invokes the exact zero-mode convention below.
    area : float
        Positive finite unit-cell area in square angstroms.
    coupling : float
        Nonnegative finite Coulomb constant C in eV angstroms; the strict-2D
        Fourier interaction at positive p is 2*pi*C/(area*p).
    pair_averages : "np.ndarray"
        Finite real vector (G,) in [0, 1], the double-coordinate averages
        from thickness_averages. Use ones for strict 2D.

    Returns
    -------
    matrices : "np.ndarray"
        Complex array (3, G, G), ordered as the symmetric dielectric matrix,
        its full matrix inverse, and W in eV. Q2D uses the source's symmetric
        construction with the supplied weighted response and pair averages.
        Each p=0 row/column of the dielectric matrix and its inverse equals
        the corresponding identity row/column; that row/column of W is zero.
        The body is computed with all remaining local-field couplings.

    Raises
    ------
    ValueError
        If shapes, finiteness, real inputs, signs, Hermiticity, response
        eigenvalue bounds, average bounds, or positive area are violated.
    """
    return 0.0
```

### Step 6

direct_kernel

Goal
----
Project screened interactions into vertical electron-hole pairs.

```python
def direct_kernel(
    vectors: "np.ndarray",
    k_points: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    interactions: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    """Contract W into the direct term for one valence and one conduction band.

    Parameters
    ----------
    vectors : "np.ndarray"
        Finite complex array (K, 2, 2), K >= 1, indexed by k, orbital, band;
        valence is band 0 and conduction band 1. General finite coefficients
        are supported; physical inputs are orthonormal Bloch eigenvectors.
    k_points : "np.ndarray"
        Finite real array (K, 2) of Cartesian momenta in inverse angstroms.
    g_vectors : "np.ndarray"
        Finite real array (G, 2), G >= 1, in inverse angstroms.
    centres : "np.ndarray"
        Finite real array (2, 2) of in-plane orbital centres in angstroms.
    interactions : "np.ndarray"
        Finite complex array (T, G, G), T >= 1, of W matrices in eV, in the
        Fourier convention in the module description. It already includes
        the selected zero-mode convention and all thickness effects.
    pair_indices : "np.ndarray"
        Integer array (K, K), with entries in [0, T), mapping ordered pair
        (i,j) to the W matrix for the unreduced transfer k_i-k_j. Boolean
        arrays are invalid. Consistency with the listed momenta is the
        caller's responsibility.

    Returns
    -------
    direct : "np.ndarray"
        Complex array (K, K) in eV. Each element is the contraction of the
        conduction matrix element of exp(+i(q+G).r), W_GG'(q), and the
        valence matrix element of exp(-i(q+G').r), with a 1/K mesh factor.
        No spin multiplier or exchange term is included. Mutual transfer
        symmetries are not enforced; physical consistent inputs yield a
        Hermitian direct kernel, covariant under independent band phases.

    Raises
    ------
    ValueError
        If a shape, finite/real input requirement, or integer index bound
        above is violated.
    """
    return 0.0
```

### Step 7

lowest_exciton_energy

Goal
----
Obtain the lowest Tamm-Dancoff excitation in the direct interaction channel.

```python
def lowest_exciton_energy(energies: "np.ndarray", direct: "np.ndarray") -> float:
    """Return the smallest eigenvalue of the direct-channel BSE Hamiltonian.

    Parameters
    ----------
    energies : "np.ndarray"
        Finite real array (K, 2), K >= 1, in eV, ordered valence/conduction
        with a strictly positive direct gap at every momentum.
    direct : "np.ndarray"
        Finite complex array (K, K) in eV, Hermitian to absolute entrywise
        tolerance 1e-10. It includes the mesh factor. Roundoff within that
        tolerance is removed by Hermitian symmetrization.

    Returns
    -------
    energy : float
        Lowest eigenvalue in eV. A degenerate lowest eigenvalue is supported.
        No additional spin factor or momentum weight is applied. Negative
        eigenvalues are returned as mathematical outputs without clipping.

    Raises
    ------
    ValueError
        If a shape, finiteness, real-energy, positive-gap or Hermiticity
        requirement above is violated.
    """
    return 0.0
```

### Step 8

head_regularization

Goal
----
Regularize the screened head at the zero transfer by the source's small-circle average.

```python
def head_regularization(
    inverse_head_x: float,
    inverse_head_y: float,
    transfer_x: float,
    transfer_y: float,
    fraction: float,
    area: float,
    coupling: float,
) -> float:
    """Return the regularized screened head W_00 at zero transfer, in eV.

    Parameters
    ----------
    inverse_head_x, inverse_head_y : float
        Finite real values in (0, 1], the head element of the full inverse
        dielectric matrix at the mesh-adjacent transfers (transfer_x, 0) and
        (0, transfer_y), respectively.
    transfer_x, transfer_y : float
        Finite strictly positive magnitudes, in inverse angstroms, of the
        mesh-adjacent transfers along the x and y axes.
    fraction : float
        Finite value in (0, 1]; the circle radius is q0 = fraction * k0 with
        k0 the smaller of the two transfers.
    area : float
        Positive finite unit-cell area in square angstroms.
    coupling : float
        Nonnegative finite Coulomb constant C in eV angstroms; the strict-2D
        Fourier interaction at positive p is 2*pi*C/(area*p).

    Returns
    -------
    head : float
        The source's small-circle regularization: the average of the strict-2D
        interaction over the circle of radius q0 about Gamma, 4*pi*C/(area*q0),
        multiplied by 1 - q0*(r0x + r0y)/4, where the screening parameters are
        the finite-difference slopes r0x = (1 - inverse_head_x)/transfer_x and
        r0y = (1 - inverse_head_y)/transfer_y of the inverse dielectric head.

    Raises
    ------
    ValueError
        If any input is not finite and real, if a transfer or the area is not
        strictly positive, if the fraction lies outside (0, 1], if the coupling
        is negative, or if an inverse head lies outside (0, 1].
    """
    return 0.0
```

### Step 9

screening_thickness_derivatives

Goal
----
Differentiate the paper's symmetric Q2D screening under slab dilation.

```python
def screening_thickness_derivatives(
    left_energies: "np.ndarray",
    right_energies: "np.ndarray",
    left_vectors: "np.ndarray",
    right_vectors: "np.ndarray",
    transfer: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    height_fractions: "np.ndarray",
    thickness: float,
    area: float,
    coupling: float,
) -> "np.ndarray":
    """Return value, first derivative and second derivative of screening.

    Parameters
    ----------
    left_energies, right_energies : "np.ndarray"
        Real arrays (K, 2), K >= 1, obeying static_response's positive
        within- and cross-momentum gap contract, with spin multiplicity one.
        Energies remain fixed while thickness varies.
    left_vectors, right_vectors : "np.ndarray"
        Finite complex arrays (K, 2, 2), indexed by k, orbital, band.
        Coefficients remain fixed while thickness varies. General finite
        coefficients are supported as in density_vertices.
    transfer : "np.ndarray"
        Finite real Cartesian vector (2,) in inverse angstroms.
    g_vectors : "np.ndarray"
        Finite real array (G, 2), G >= 1, in inverse angstroms.
    centres : "np.ndarray"
        Finite real array (2, 2) of in-plane orbital centres in angstroms.
    height_fractions : "np.ndarray"
        Finite real vector (2,) in [-0.5, 0.5]. At any thickness h >= 0
        orbital a lies at z_a(h) = h*height_fractions[a].
    thickness : float
        Finite nonnegative evaluation thickness in angstroms. At zero,
        return the right-hand derivatives of this dilation family.
    area : float
        Positive finite cell area in square angstroms.
    coupling : float
        Nonnegative finite Coulomb constant C in eV angstroms.

    Returns
    -------
    derivatives : "np.ndarray"
        Complex array (3, 3, G, G). Axis 0 contains orders 0, 1, 2 of
        differentiation with respect to thickness h, evaluated at thickness.
        Axis 1 contains epsilon, epsilon_inverse, W, in that order.
        Order 0 equals screened_interaction applied to the source's
        symmetric, orbital-weighted response. At every |q+G|=0 row and
        column epsilon and its inverse have identity order-0 entries and
        zero higher derivatives; W has zero entries at all three orders.
        The later orchestrator replaces W's zero-transfer head.
        Other entries retain all local-field couplings. Derivatives must
        include both single-coordinate and double-coordinate averages.
        Matrix values have the original units; orders 1 and 2 additionally
        carry inverse angstroms and inverse square angstroms, respectively.
        Differentiate the defining construction, including its matrix
        inverse, analytically. Thickness finite differences are excluded.
        Compose thickness_averages, density_vertices, static_response and
        screened_interaction for the order-0 construction.

    Raises
    ------
    ValueError
        If any documented shape, finiteness, realness, sign, gap or
        fractional-height bound is violated.
    """
    return 0.0
```

### Step 10

exciton_energy_derivatives

Goal
----
Compute thickness derivatives of a simple lowest exciton eigenvalue.

```python
def exciton_energy_derivatives(
    energies: "np.ndarray",
    direct_derivatives: "np.ndarray",
    gap_tolerance: float = 1e-9,
) -> "np.ndarray":
    """Return the lowest excitation energy, slope and curvature.

    Parameters
    ----------
    energies : "np.ndarray"
        Finite real array (K, 2), K >= 1, with positive rowwise direct gaps.
        The band energies do not depend on the differentiation parameter.
    direct_derivatives : "np.ndarray"
        Finite complex array (3, K, K) containing D, dD/dh and d^2D/dh^2.
        Each is Hermitian to entrywise absolute tolerance 1e-10, with
        roundoff within that bound symmetrized. Units are eV, eV/angstrom
        and eV/angstrom^2. The mesh weight is already included.
    gap_tolerance : float
        Positive finite number in eV. For K > 1, the difference of the
        lowest two eigenvalues of H must be greater than this number.

    Returns
    -------
    derivatives : "np.ndarray"
        Real array (3,) containing E0, dE0/dh and d^2E0/dh^2, with units
        eV, eV/angstrom and eV/angstrom^2. The two derivatives are ordinary
        analytic derivatives of the simple lowest eigenvalue of
        H(h)=diag(E_c-E_v)-D(h); include the eigenstate response. No
        finite differences, clipping, extra spin or extra mesh factor.
        For K=1 the eigenstate-response contribution is zero.
        The order-0 value is obtained using lowest_exciton_energy.

    Raises
    ------
    ValueError
        If any shape, finiteness, realness, positive direct gap,
        Hermiticity or positive tolerance requirement is violated, or the
        lowest excitation is not isolated by more than gap_tolerance.
    """
    return 0.0
```

### Step 11

thickness_exciton_curvature

Goal
----
Compose the microscopic screening and spectral thickness response.

```python
def thickness_exciton_curvature(
    mesh_size: int,
    lattice: "np.ndarray",
    model: "np.ndarray",
    centres: "np.ndarray",
    thickness: float,
    coupling: float,
    reciprocal_extent: int = 1,
    fraction: float = 0.5,
    gap_tolerance: float = 1e-9,
) -> float:
    """Return the analytic second thickness derivative of the exciton shift.

    Parameters
    ----------
    mesh_size : int
        Odd integer n >= 3; use K=n*n equal-weight mesh points
        (2*pi*(i-(n-1)/2)/(n*a_x), 2*pi*(j-(n-1)/2)/(n*a_y)),
        i,j=0,...,n-1, in row-major order. Booleans are invalid.
    lattice : "np.ndarray"
        Positive finite real array (2,) of lattice lengths in angstroms.
    model : "np.ndarray"
        Finite real array (7,) containing solve_bands parameters, obeying
        its strict mass bound. The Hamiltonian stays fixed under dilation.
    centres : "np.ndarray"
        Finite real array (2, 3) of orbital coordinates in angstroms at the
        evaluation thickness. Heights lie inside [-thickness/2, thickness/2].
        In-plane centres stay fixed; at variable h the heights become
        centres[:, 2]*h/thickness, so the fractional heights are constant.
    thickness : float
        Finite strictly positive evaluation thickness in angstroms.
    coupling : float
        Finite nonnegative Coulomb constant C in eV angstroms.
    reciprocal_extent : int
        Integer R >= 0, excluding booleans. Include every
        G=(2*pi*r/a_x, 2*pi*s/a_y), r,s=-R,...,R, in row-major order.
    fraction : float
        Finite real number in (0, 1]; the Gamma circle radius is this
        fraction times the nearest nonzero mesh momentum.
    gap_tolerance : float
        Positive finite lowest-exciton isolation tolerance, in eV, as in
        exciton_energy_derivatives.

    Returns
    -------
    curvature : float
        1000*d^2(E_Q2D(h)-E_2D)/dh^2 at h=thickness, in meV/angstrom^2.
        E_2D is the fixed strict-2D reference with zero heights and thickness.
        Evaluate bands at k+q with unreduced q=k_i-k_j and retain all G in
        screening and the direct channel. Use both interband directions
        with spin one, direct attraction and no exchange.
        At q=0 all orders of W's wings vanish and its screened body is
        retained. Its order-0 head is head_regularization, using this
        Q2D calculation's inverse heads at the mesh-adjacent axis transfers.
        At variable h the same head formula and fixed circle radius apply;
        differentiate its inverse-head slopes with respect to h as well.
        Use analytic screening and eigenvalue derivatives; include the
        exciton eigenstate response. Thickness finite differences are
        excluded. Compose all ten preceding public functions, transitively
        where appropriate, and use their outputs.

    Raises
    ------
    ValueError
        If an integer, shape, finite/real input, mass, lattice, thickness,
        coupling, fraction or height condition fails, or the lowest
        excitation is not isolated as required by gap_tolerance.
    """
    return 0.0
```
