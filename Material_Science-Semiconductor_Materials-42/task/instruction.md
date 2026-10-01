# Material_Science-Semiconductor_Materials-42

## Background

This constructed three-orbital benchmark uses the supplied paper’s method and Supplementary Information (SI). Its Hamiltonian, observations and discretization are synthetic. Lengths are in Å, momenta in Å⁻¹, and energies in eV. The lattice is square with \(a=3.2\), cell area \(a^2\), orthogonal point orbitals at \(\boldsymbol\tau/a=((0,0),(0.37,0.12),(0.16,0.43))\), and heights \(z_a/d=(0,0.29,-0.23)\). With \(x=ak_x,y=ak_y\), the upper triangle of the lattice-gauge Bloch Hamiltonian is
\[
H_{00}=-\mu+0.17(\cos x+\cos y),\quad H_{11}=\mu+0.11\cos x-0.08\cos y,\quad H_{22}=\mu+1.15+0.09\cos(x+y),
\]
\[
H_{01}=0.44+0.31e^{-ix}+0.23e^{-iy},\quad H_{02}=0.29-0.19e^{ix}+0.21e^{-i(x+y)},\quad H_{12}=0.16+0.13e^{-iy};\qquad H_{ba}=H_{ab}^*.
\]
Eigenvalues \(e_b\) increase with \(b=0,1,2\); eigenvector columns \(U_{ab}\) are normalized. Occupations are \((1,0,0)\), spin degeneracy is two, temperature is zero, and the vertical Tamm–Dancoff kernel is the attractive direct term. The public eigenvector interface fixes each column's largest-magnitude component to real nonnegative, with the lowest component index resolving ties.

For odd \(n\), \(\mathcal K_n=\{2\pi(i-(n-1)/2,j-(n-1)/2)/(an):0\leq i,j<n\}\), ordered by \(i\), then \(j\). Polarization always uses \(\mathcal K_7\); the exciton uses \(\mathcal K_n\). The reciprocal set is \(\mathcal G_L=\{(2\pi/a)(u,v):u,v\in\mathbb Z,\ u^2+v^2\leq L^2\}\), ordered by \((u^2+v^2,u,v)\), with the zero vector first. The same set enters dielectric inversion and the direct interaction. Every transfer is the raw difference \(\mathbf q=\mathbf k_i-\mathbf k_j\); shifted Bloch Hamiltonians are evaluated at \(\mathbf k+\mathbf q\). Brillouin-zone sums are arithmetic averages.

Write \(Q_G=|\mathbf q+\mathbf G|\), \(c=2\pi(14.3996454784255)/(a^2\kappa)\), and \(v_G=c/Q_G\). The Coulomb constant in parentheses has units eV Å. The two dimensionless slab averages are defined by
\[
F_a(Q,d)=\frac1d\int_{-d/2}^{d/2}e^{-Q|z-z_a|}\,dz,\qquad B(Q,d)=\frac1{d^2}\int_{-d/2}^{d/2}\int_{-d/2}^{d/2}e^{-Q|z-z'|}\,dz\,dz'.
\]
Both extend continuously to one at \(Qd=0\), with fixed fractional heights. The symmetric orbital prescription of SI Eq. S.25 uses vertices
\[
J^{nb}_G(\mathbf k,\mathbf q)=\sum_a U_{an}(\mathbf k)^*U_{ab}(\mathbf k+\mathbf q)e^{-i(\mathbf q+\mathbf G)\cdot\boldsymbol\tau_a}\sqrt{F_a(Q_G,d)}.
\]
The static occupation-difference response and symmetric dielectric matrix are
\[
\chi_{GG'}(\mathbf q)=\frac{2}{49}\sum_{\mathbf k\in\mathcal K_7}\sum_{n,b:f_n\ne f_b}\frac{f_n-f_b}{e_n(\mathbf k)-e_b(\mathbf k+\mathbf q)}J^{nb}_G(J^{nb}_{G'})^*,\qquad \bar\varepsilon=I-\sqrt v\,\chi\sqrt v.
\]
The effective screened potential is \(\bar W=\sqrt{vB}\,\bar\varepsilon^{-1}\sqrt{vB}\); diagonal square roots act on the reciprocal indices. The macroscopic response is \(\varepsilon_M=1/(\bar\varepsilon^{-1})_{00}\). At exactly \(Q_G=0\), use the value zero for the bare factor during matrix construction: the dielectric head is then one and its wings vanish, while the body is computed normally.

The following finite-grid origin prescription fixes the benchmark and extends the paper's first-order circular-cell rule to its pair-averaged Q2D potential. Let \(\delta=0.002/a\) and \(s_\alpha(t)=[\varepsilon_M(t\hat{\boldsymbol\alpha})-1]/t\); use \(r_\alpha=2s_\alpha(\delta/2)-s_\alpha(\delta)\) for \(\alpha=x,y\). With \(q_0=0.35(2\pi/an)\), replace \(\bar W_{00}(0)\) by \(c[2/q_0-(r_x+r_y)/2-d/3]\), set the origin wings to zero, and retain the origin body. This is the stipulated first-order cell prescription at the fixed \(\delta,q_0\); the synthetic targets refer to that prescription.

For the direct term define the unweighted point-orbital vertex
\[
I^{nb}_{ij,G}=\sum_a U_{an}(\mathbf k_j)^*e^{-i(\mathbf k_i-\mathbf k_j+\mathbf G)\cdot\boldsymbol\tau_a}U_{ab}(\mathbf k_i).
\]
With \(N=n^2\), electron bands \(c,c'=1,2\), and pair order \((i,c)\),
\[
D_{ic,jc'}=\frac1N\sum_{GG'}(I^{c'c}_{ij,G})^*\bar W_{GG'}(\mathbf k_i-\mathbf k_j)I^{00}_{ij,G'},\qquad H^X_{ic,jc'}=(e_c(\mathbf k_i)-e_0(\mathbf k_i))\delta_{ij}\delta_{cc'}-D_{ic,jc'}.
\]
\(E_0\leq E_1\leq\cdots\) are the eigenvalues of this Hermitian matrix. \(E_0^{\mathrm{head}}\) uses the same parameters and the same full dielectric inversion, with only the \(G=G'=0\) screened-potential entry retained in the direct term. Full-precision arithmetic is used except for the prescribed six-decimal rounding of fitted parameters. The exact-target equations define all admissible calibration roots; the supplied box has isolated roots. The root calculation has coordinate convergence \(10^{-8}\) in \(\kappa\) and \(d\), together with absolute energy residual below \(10^{-9}\) eV before rounding.

## Problem

Determine the splitting \(\Delta_X=E_1-E_0\) of the two lowest vertical excitons of the synthetic semiconductor defined below, using the attached paper's symmetric quasi-two-dimensional microscopic screening construction. Two exact calibration targets are \(E_0=0.7754618169\) eV at \((n,\mu,L)=(5,1.10,1)\) and \(E_0=1.6367562578\) eV at \((5,1.60,1)\), with common environmental permittivity \(6\leq\kappa\leq9\) and slab thickness \(0.6\leq d\leq3.2\) Å. Among the simultaneous fits, select the pair whose predicted macroscopic dielectric response at \(\mathbf q_w=(0.27,0.13)\) Å\(^{-1}\) is closest to the independent witness \(1.04225\); evaluate that response and the requested excitons at \((n,\mu,L)=(7,1.35,2)\). Each fitted coordinate is converged to \(10^{-8}\) in its stated units and rounded to six decimal places before selection and prediction, with equal witness discrepancies resolved by increasing \(d\), then increasing \(\kappa\). In the reasoning, report the six-value certificate \((\kappa,d,\varepsilon_M(\mathbf q_w),\overline W_{00}(0),E_0,E_0^{\mathrm{head}}-E_0)\), in dimensionless units, Å, dimensionless units, eV, eV, and eV, respectively; all seven requested numbers have absolute tolerance \(2\times10^{-5}\) in their stated units. Briefly justify the source’s off-plane inverse/average approximation, its finite-thickness cutoff convergence, the coupled dielectric/direct cutoffs, branch selection, direct-kernel conjugation and normalization, and the finite-thickness origin correction.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

bloch_frames

Goal
----
Evaluate the supplied lattice-gauge Hamiltonian and its canonical Bloch frames.

```python
import numpy as np

def bloch_frames(kpoints: np.ndarray, mass: float) -> np.ndarray:
    """Parameters
    ----------
    kpoints : finite real ndarray, shape (K,2), K>=1
        Ordered momenta (kx,ky), in inverse angstroms.
    mass : finite positive float
        Hamiltonian parameter mu, in eV, in the stated three-orbital model.
    Returns
    -------
    complex ndarray, shape (K,4,3)
        Row 0 holds the three ascending eigenenergies in eV (zero imaginary part).
        Rows 1:4 hold dimensionless eigenvector columns, in orbital order.
        Each column uses the largest-component real-nonnegative gauge; lowest
        orbital index breaks a largest-magnitude tie.
    Raises
    ------
    ValueError : malformed/nonfinite momenta or nonpositive/nonfinite mass."""
    return np.zeros((len(kpoints),4,3),dtype=complex)
```

### Step 2

slab_averages

Goal
----
Evaluate the distinct orbital and uniform-pair slab integrals in the symmetric Q2D construction (SI S.25–S.26).

```python
import numpy as np

def slab_averages(lengths: np.ndarray, thickness: float, z_fractions: np.ndarray) -> np.ndarray:
    """Parameters
    ----------
    lengths : finite nonnegative real ndarray, shape S, rank>=1, nonempty
        Q=|q+G| in inverse angstroms.
    thickness : finite nonnegative float
        Slab thickness d in angstroms.
    z_fractions : finite real ndarray, shape (A,), A>=1
        Orbital heights divided by d, each in [-0.5,0.5].
    Returns
    -------
    real ndarray, shape S+(A+1,)
        Dimensionless single-site integrals F_a in orbital order, followed by the
        uniform-pair integral B in the last entry. Continuous Q*d=0 values are one.
    Raises
    ------
    ValueError : malformed inputs, negative Q or d, or heights outside the slab."""
    return np.zeros(np.shape(lengths)+(len(z_fractions)+1,))
```

### Step 3

density_vertices

Goal
----
Form point-orbital density vertices with the symmetric orbital factors of SI S.25, keeping the Fourier and Bloch-gauge phases.

```python
import numpy as np

def density_vertices(left_vectors: np.ndarray, right_vectors: np.ndarray, phases: np.ndarray, orbital_averages: np.ndarray) -> np.ndarray:
    """Parameters
    ----------
    left_vectors : finite complex ndarray, shape (K,3,3)
        Bloch coefficients U[k,orbital,band] at the left momenta.
    right_vectors : finite complex ndarray, shape (Q,K,3,3)
        Coefficients at right momenta k+q for each transfer.
    phases : finite complex ndarray, shape (Q,G,3)
        exp[-i(q+G).tau_a] in transfer, reciprocal, orbital order.
    orbital_averages : finite nonnegative real ndarray, shape (Q,G,3)
        Single-site F_a, dimensionless; ones define the unweighted direct vertex.
    Returns
    -------
    complex ndarray, shape (Q,K,G,3,3)
        Dimensionless vertices, indexed [q,k,G,left_band,right_band], with left
        coefficients conjugated and symmetric square-root orbital factors.
    Raises
    ------
    ValueError : incompatible shapes, nonfinite inputs, or negative form factors."""
    return np.zeros((np.shape(right_vectors)[0],len(left_vectors),np.shape(phases)[1],3,3),dtype=complex)
```

### Step 4

dielectric_matrices

Goal
----
Construct the complete finite-grid static RPA response using main Eq. 5 and the symmetric Q2D dielectric prescription.

```python
import numpy as np

def dielectric_matrices(left_energies: np.ndarray, right_energies: np.ndarray, vertices: np.ndarray, bare: np.ndarray, spin: float = 2.) -> np.ndarray:
    """Parameters
    ----------
    left_energies : finite real ndarray, shape (K,3)
        Energies at k in eV; band 0 occupied and bands 1,2 empty.
    right_energies : finite real ndarray, shape (Q,K,3)
        Energies at k+q in eV. Occupied/empty spectra remain globally separated.
    vertices : finite complex ndarray, shape (Q,K,G,3,3)
        Dimensionless density vertices in left-band/right-band order.
    bare : finite nonnegative real ndarray, shape (Q,G)
        Bare Coulomb factors in eV, with zero placeholders at singular Q_G=0.
    spin : finite nonnegative float
        Spin degeneracy multiplying the k average; default 2.
    Returns
    -------
    complex ndarray, shape (Q,G,G)
        Dimensionless symmetric dielectric matrices I-sqrt(v)*chi*sqrt(v).
        chi uses the complete static occupation-difference sum, normalized by K.
    Raises
    ------
    ValueError : incompatible/nonfinite arrays, negative bare/spin, or overlapping
        occupied and empty spectra."""
    return np.zeros((len(right_energies),np.shape(bare)[1],np.shape(bare)[1]),dtype=complex)
```

### Step 5

screening_lengths

Goal
----
Extract the anisotropic microscopic screening lengths from the dielectric Schur complement (SI S.12) for the origin-cell treatment (SI S.20–S.21); the finite sampling rule is stipulated.

```python
import numpy as np

def screening_lengths(dielectric_samples: np.ndarray, delta: float) -> np.ndarray:
    """Parameters
    ----------
    dielectric_samples : finite positive Hermitian complex ndarray, shape (4,G,G)
        Dimensionless symmetric epsilon_bar at (delta,0), (delta/2,0),
        (0,delta), (0,delta/2), in exactly that order; G>=1 and head index 0.
    delta : finite positive float
        Sampling momentum in inverse angstroms.
    Returns
    -------
    real ndarray, shape (2,)
        Ordered [r_x,r_y] in angstroms, using r_alpha=2*s_alpha(delta/2)
        -s_alpha(delta), with s_alpha(t)=(epsilon_M(t)-1)/t. The macroscopic
        epsilon_M is the full inverse-head reciprocal, equivalently the Schur
        complement of the microscopic dielectric body; include local fields.
    Raises
    ------
    ValueError : malformed/nonfinite/non-positive-Hermitian samples or invalid delta."""
    return np.zeros((2,))
```

### Step 6

screened_potentials

Goal
----
Combine the independently averaged Coulomb pair potential with the inverse symmetric dielectric matrix and regularize head, wings and body according to the specified Q2D cell prescription.

```python
import numpy as np

def screened_potentials(dielectrics: np.ndarray, bare: np.ndarray, pair_averages: np.ndarray, zero_index: int, q0: float, lengths: np.ndarray, kappa: float, thickness: float) -> np.ndarray:
    """Parameters
    ----------
    dielectrics : finite positive Hermitian complex ndarray, shape (Q,G,G)
        Symmetric dimensionless Q2D dielectric matrices; reciprocal index 0 is G=0.
    bare : finite nonnegative real ndarray, shape (Q,G)
        Bare Coulomb factors in eV.
    pair_averages : finite nonnegative real ndarray, shape (Q,G)
        Dimensionless uniform-pair slab factors B.
    zero_index : int in {-1,0,...,Q-1}
        Transfer index of q=0; -1 means that this batch has no zero transfer.
    q0 : finite positive float
        Circular-cell radius in inverse angstroms.
    lengths : finite real ndarray, shape (2,)
        Screening lengths [r_x,r_y], in angstroms.
    kappa : finite positive float
        Dimensionless environmental permittivity.
    thickness : finite nonnegative float
        Slab thickness d, in angstroms.
    Returns
    -------
    complex ndarray, shape (Q,G,G)
        W in eV. Use the full dielectric inverse between square roots of v*B.
        At zero_index>=0, the head is c*(2/q0-(r_x+r_y)/2-d/3), origin wings
        are zero, and the body retains its matrix-inverse value; a=3.2 angstroms.
    Raises
    ------
    ValueError : incompatible/nonfinite data, invalid scalar domains, or a
        dielectric matrix that is not positive Hermitian."""
    return np.zeros(np.shape(dielectrics),dtype=complex)
```

### Step 7

exciton_hamiltonian

Goal
----
Apply the paper’s microscopic direct kernel with reciprocal local fields and both conduction channels; the real-space Eq. 20 fixes the stated conjugation convention.

```python
import numpy as np

def exciton_hamiltonian(energies: np.ndarray, pair_vertices: np.ndarray, screened: np.ndarray, q_indices: np.ndarray, head_only: bool = False) -> np.ndarray:
    """Parameters
    ----------
    energies : finite real ndarray, shape (N,3), N>=1
        Bloch energies in eV, band 0 valence and bands 1,2 conduction.
    pair_vertices : finite complex ndarray, shape (N,N,G,3,3)
        Unweighted dimensionless I[i,j,G,n,b], with left Bloch state at k_j
        and right state at k_i, and transfer k_i-k_j.
    screened : finite complex ndarray, shape (Q,G,G)
        Screened potentials in eV, with the origin prescription already applied.
    q_indices : integer ndarray, shape (N,N), values in [0,Q)
        Lookup q_indices[i,j] for each raw momentum difference.
    head_only : bool
        If true, use only screened[:,0,0] in the direct contraction. The supplied
        full dielectric inversion underlying that entry is retained.
    Returns
    -------
    complex ndarray, shape (2*N,2*N)
        Exciton Hamiltonian in eV, ordered by momentum i then conduction c=1,2.
        It equals the vertical-gap diagonal minus the direct attraction with 1/N
        normalization and the stated I^{c'c} conjugation. Physically consistent
        paired transfers and vertices yield a Hermitian matrix.
    Raises
    ------
    ValueError : incompatible/nonfinite arrays or invalid transfer indices."""
    return np.zeros((2*len(energies),2*len(energies)),dtype=complex)
```

### Step 8

exciton_levels

Goal
----
Obtain the lowest exciton eigenenergies while retaining physical degeneracies.

```python
import numpy as np

def exciton_levels(hamiltonian: np.ndarray, count: int = 3) -> np.ndarray:
    """Parameters
    ----------
    hamiltonian : finite Hermitian complex ndarray, shape (D,D)
        Exciton Hamiltonian in eV, D>=1.
    count : int, 1<=count<=D
        Number of lowest levels requested, including multiplicities.
    Returns
    -------
    real ndarray, shape (count,)
        Exciton energies in ascending order, in eV.
    Raises
    ------
    ValueError : malformed/nonfinite/non-Hermitian matrix or invalid count."""
    return np.zeros((count,))
```

### Step 9

calibration_branches

Goal
----
Resolve the inverse calibration’s isolated branches rather than depending on one optimizer starting point; this is a constructed inverse use of the paper’s forward model.

```python
import numpy as np

def calibration_branches(forward, observed: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Parameters
    ----------
    forward : callable
        Maps a finite real [kappa,d] array of shape (2,) to two finite ground-state
        energies (eV) in a real array of shape (2,). On the supplied rectangle,
        the first energy is strictly increasing in kappa. Its feasible first-
        energy contour has simple isolated second-energy crossings separated
        by at least 0.15 angstrom; each crossing has a feasible neighborhood of
        radius (d_max-d_min)/32 relative to the rectangle. The supplied experiment and analytical unit
        fixtures satisfy this class. Interior tangent roots are outside this class.
    observed : finite real ndarray, shape (2,)
        Exact target energies in the same order, in eV.
    bounds : finite real ndarray, shape (2,2)
        Rows [kappa_min,kappa_max] and [d_min,d_max], each strictly increasing.
    Returns
    -------
    real ndarray, shape (R,2)
        All admissible root pairs [kappa,d], with absolute coordinate
        convergence 1e-8 and with both energy residuals below 1e-9 eV before rounding. Coordinates are rounded to 6 decimal places, and
        rows sorted by increasing d then kappa. Empty shape (0,2) means no fit.
    Raises
    ------
    ValueError : malformed/nonfinite observations or bounds."""
    return np.zeros((0,2))
```

### Step 10

predict_exciton_splitting

Goal
----
Orchestrate the microscopic screening, complete inverse branch set, dielectric witness selection, and full versus head-projected exciton prediction.

```python
import numpy as np

def predict_exciton_splitting(observed: np.ndarray, witness: float, bounds = ((6.,9.),(.6,3.2)), prediction = (7,1.35,2)) -> np.ndarray:
    """Parameters
    ----------
    observed : finite real ndarray, shape (2,)
        Exact calibration ground energies in eV at (n,mu,L)=(5,1.1,1) and
        (5,1.6,1), respectively. The isolated-root class of calibration_branches
        applies whenever roots exist.
    witness : finite positive float
        Dimensionless dielectric witness at q_w=(0.27,0.13) inverse angstroms,
        evaluated at the prediction mass and reciprocal cutoff.
    bounds : real array-like, shape (2,2)
        Rows [kappa_min,kappa_max], [d_min,d_max], a nondegenerate subrectangle
        of [6,9] x [0.6,3.2], in dimensionless units and angstroms.
    prediction : tuple (n,mu,L)
        n is 3,5 or 7; mu is in [0.8,2] eV; reciprocal cutoff L is 1 or 2.
    Returns
    -------
    real ndarray, shape (7,)
        Ordered [E1-E0,kappa,d,epsilon_M(q_w),W00(0),E0,E0_head-E0]. Units are
        eV, dimensionless, angstrom, dimensionless, eV, eV, eV. The chosen root
        minimizes witness error after six-decimal rounding, then d, then kappa.
        An empty admissible fit set returns seven entries equal to -1.
        This final public function must call and combine every preceding public
        function, directly or through helpers/callbacks, using their returned
        scientific quantities. Constants, Hamiltonian, grids, approximations,
        origin prescription and certificate are those in the task background.
    Raises
    ------
    ValueError : invalid observation/witness, bounds, or prediction configuration."""
    return np.zeros((7,))
```
