# Material_Science-Molecular_Modeling-47

## Background

*Linear response as a self-consistent problem.* Density-functional perturbation theory computes the first-order change of the ground-state density under a perturbation of the external potential. The independent-particle susceptibility $\chi_0$ gives the density change a *fixed* effective potential would produce; but the induced density itself changes the Hartree and exchange-correlation potential through the kernel $K$, which changes the density again. Summing that geometric series is the Dyson equation $\delta n = \chi_0[\delta V_{\mathrm{ext}} + K \delta n]$, equivalently $(1 - \chi_0 K)\,\delta n = \chi_0[\delta V_{\mathrm{ext}}]$. The operator $\mathcal{E} = 1 - \chi_0 K$ is the (adjoint) dielectric operator, and screening in a metal is a large effect: the interacting response is typically several times smaller than the bare one, so treating the bare response as an approximation to the physical one is qualitatively wrong, not merely inaccurate.

*Metals, smearing and particle-number conservation.* At zero temperature with integer occupations, only occupied-to-empty transitions contribute to $\chi_0$ and the occupations are inert. At finite electronic temperature, states near the Fermi level carry fractional occupations, and two things change. Transitions between two partially occupied states contribute in proportion to the divided difference $(f_n - f_m)/(\varepsilon_n - \varepsilon_m)$, which is finite everywhere and tends to $f'(\varepsilon_n)$ as the two levels approach one another, so a vanishing denominator must be resolved as a limit and not treated as a singularity. And the occupations themselves respond: a perturbation that raises a level pushes electrons off it, and in a closed system they must go somewhere, so the Fermi level moves. The shift is fixed by the constraint that the total electron number does not change. Omitting it does not produce a small error - it describes a different physical system, one exchanging electrons with a reservoir, and the difference is of leading order whenever any state is fractionally occupied. The cleanest statement of this is that a spatially constant perturbation, which shifts every level by the same amount, must leave the density exactly unchanged at fixed particle number.

*Why the susceptibility is never assembled.* Applying $\chi_0$ to a potential formally requires, for each occupied state, a sum over the entire unoccupied spectrum weighted by inverse energy differences. In a plane-wave calculation that spectrum is enormous and computing it is precisely what one cannot afford. The standard alternative, due to Sternheimer, is to obtain the same quantity by solving a linear system for each occupied state: the shifted Hamiltonian acting on the unknown response equals minus the perturbation acting on the state, with both sides projected onto the subspace orthogonal to the states treated explicitly. That projection does two jobs at once - it removes the null direction that would otherwise make the system singular, and it removes exactly the contributions accounted for separately by the retained-pair term, so that nothing is double counted. Because the projected operator is Hermitian, a conjugate-gradient iteration solves it, and every iterate is kept inside the subspace.

*Inexact Krylov methods.* An outer Krylov iteration whose matrix-vector product is computed only approximately is not simply a perturbed iteration: the theory of inexact Krylov methods shows that the tolerable inexactness *grows* as the outer iteration converges. Early on the outer iterate is far from the solution and an accurate operator application is wasted work; late in the iteration the same relative error would destroy the outer accuracy. The practical consequence is a rule in which the inner tolerance is inversely proportional to the current outer residual norm, with a prefactor that accounts for the conditioning of the projected problem and for how strongly each inner solve enters the quantity of interest. The source paper derives that rule for the DFPT setting, bounds the resulting error in $\delta n$, and reports that the inner solves may be relaxed by orders of magnitude - a saving of roughly forty per cent of the total time on realistic systems - without measurable loss in the outer answer. The property that makes such a scheme gradeable is that the converged answer is a property of the *equations*, not of the path taken to them: the same number must emerge whether the inner solves were run loose or tight, or the system solved densely and directly.

*Consistency of the construction.* The decomposition of $\chi_0$ into a retained-pair block, a projected Sternheimer block and an occupation-shift block is an identity, not an approximation, provided the neglected states carry no occupation. It can therefore be checked against the closed form obtained by diagonalising the Hamiltonian completely and summing the divided differences over all pairs, which is the standard density-matrix expression for the Frechet derivative of the Fermi-Dirac matrix function. Agreement at machine precision is the correct verification that the blocks have been assembled with the right weights and signs; any disagreement indicates a structural error rather than a tolerance issue.

## Problem

Perturbing a self-consistent electronic-structure calculation is not the same as perturbing a fixed Hamiltonian: a change in the external potential shifts the electron density, the shifted density changes the Hartree and exchange-correlation potential, and the changed potential shifts the density again. The physically meaningful response is the fixed point of that loop. In a metal the problem is harder still, because states at the Fermi level are only fractionally occupied, so the occupations themselves respond and the total electron number must be held fixed while they do. This task runs one such calculation to completion on a fully specified model and asks for the response of a single observable.

*Model.* A one-dimensional periodic model solid is discretised on a uniform real-space grid of $N_g = 20$ points spanning a cell of length $L = 8.0$, in atomic units throughout. The grid points are $x_j = jL/N_g$ for $j = 0,\dots,N_g-1$, and the plane waves are $G_n = 2\pi k_n/L$ with the signed integers $k_n = n$ for $n < N_g/2$ and $k_n = n - N_g$ otherwise, which is the ordering a standard discrete Fourier transform produces. The unperturbed Kohn-Sham Hamiltonian is a kinetic operator diagonal in reciprocal space plus a local potential diagonal in real space,

$$H_{jk} \;=\; \frac{1}{N_g}\sum_{n} \frac{G_n^{2}}{2}\, e^{\,\mathrm{i}G_n (x_j - x_k)} \;+\; \delta_{jk}\, V(x_j), \qquad V(x) = 0.9\cos\frac{2\pi x}{L} + 0.35\sin\frac{4\pi x}{L} + 0.15\cos\frac{6\pi x}{L}.$$

Electrons occupy its eigenstates by Fermi-Dirac statistics, $f(\varepsilon) = \big[1 + e^{\beta(\varepsilon - \mu)}\big]^{-1}$, at inverse temperature $\beta = 10.0$ and chemical potential $\mu = 0.55$. The system is metallic: the third state is roughly half filled, so the occupations respond to the perturbation and the Fermi level moves. The lowest $n_{\mathrm{occ}} = 8$ states are retained explicitly, every neglected state having occupation below $10^{-15}$.

*Response.* Perturb the external potential by $\delta V_{\mathrm{ext}}(x) = 0.4\cos(2\pi x/L) - 0.25\sin(6\pi x/L) + 0.1$. The independent-particle susceptibility $\chi_0$ maps a local potential change to the first-order density change it induces at fixed electron number. Writing $B_{mn} = \langle \phi_m | \delta V | \phi_n \rangle$ and $f'(\varepsilon) = -\beta f(1-f)$, it has three contributions: pairs of retained states couple through the divided difference $(f_n - f_m)/(\varepsilon_n - \varepsilon_m)$ of their occupations, which tends to $f'(\varepsilon_n)$ when the two eigenvalues coincide; each retained state couples to the entire unretained spectrum through the projected Sternheimer equation

$$Q\,(H - \varepsilon_n)\,Q\,\delta\phi_n \;=\; -\,Q\,(\delta V\,\phi_n), \qquad Q = 1 - \sum_m \phi_m \phi_m^{\dagger},$$

whose solution enters as $2 f_n \operatorname{Re}(\phi_n^{*}\delta\phi_n)$; and the occupations shift by $\delta f_n = f'(\varepsilon_n)\big(B_{nn} - \delta\varepsilon_F\big)$, entering as $\delta f_n |\phi_n|^2$. The Fermi-level shift $\delta\varepsilon_F$ is not free: it is fixed by conservation of the total electron number, $\sum_n \delta f_n = 0$, whence $\delta\varepsilon_F = \sum_n f'_n B_{nn} \big/ \sum_n f'_n$. If no retained state has a non-zero occupation derivative - the zero-compressibility limit of a gapped system at low temperature - the occupations cannot respond at all and every $\delta f_n$ vanishes.

*Screening.* A density change induces a potential change through the Hartree and exchange-correlation kernel

$$(K\,\delta n)(x_j) \;=\; \frac{1}{N_g}\sum_{n} v(G_n)\, e^{\,\mathrm{i}G_n x_j} \sum_{k} e^{-\mathrm{i}G_n x_k}\, \delta n(x_k) \;+\; c_{\mathrm{xc}}\,\delta n(x_j),$$

with the Coulomb factor $v(G) = (28.0/N_g)\cdot 4\pi/G^{2}$ for $G \neq 0$ and $v(0) = 0$, and the local exchange-correlation contact term $c_{\mathrm{xc}} = -1.4$. The interacting response is then the solution of the Dyson equation

$$\mathcal{E}\,\delta n = \chi_0[\delta V_{\mathrm{ext}}], \qquad \mathcal{E} := 1 - \chi_0 K.$$

*Method.* The whole calculation is carried out in the regime the source paper targets: $\chi_0$ is never formed as a matrix, only applied, and each application costs one Sternheimer solve per retained state. Two further ingredients of the paper are required. The Dyson equation is **preconditioned** on both sides by the Kerker operator $\mathbf{T}$, which multiplies each reciprocal-space component by $|G|^2/(|G|^2 + \alpha^2)$ with $\alpha = 1.5$, modified in the way the source paper prescribes so that it leaves the mean untouched and therefore conserves total charge. Preconditioning does not change the solution, only the path to it. And the outer method is **restarted** GMRES with restart size $m = 5$, carrying a running estimate $s$ of the smallest singular value of the Hessenberg matrix, updated at each restart to the smallest value encountered and accepted only when it is genuinely a lower bound to the current cycle's value. Run each inner Sternheimer solve no more accurately than the outer accuracy demands, following the source paper's adaptive per-state rule rather than a fixed inner tolerance, and take the *guaranteed* variant of that rule - the most conservative of the paper's three adaptive strategies, the one that retains both of the terms the cheaper two drop. A state of vanishing occupation never enters the density, so no inner solve is performed for it. Use $\tau = 10^{-12}$ with restart size $m = 5$, and stop the outer iteration when the estimated least-squares residual falls to $\tau/3$ or below, the remaining two thirds of the error budget being spent on the initial residual and on the inexact operator applications.

*Quantity asked.* With the observable $O(x) = \cos(2\pi x/L) + 0.4\sin(2\pi x/L)$, compute the single number

$$A \;=\; \sum_j O(x_j)\,\delta n_{\mathrm{interacting}}(x_j),$$

the screened first-order response of $O$ to $\delta V_{\mathrm{ext}}$ at fixed total electron number. Report $A$ together with the occupations, the bare (unscreened) response $O\cdot\chi_0[\delta V_{\mathrm{ext}}]$ of the same observable, and the condition number of $\mathcal{E}$.

Output Format Requirements: Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure. Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: `0.4847`, `12.6`, `1.05`). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep `<reasoning>` short (a few hundred words): the diagnostics listed above are its required content and must all appear, and beyond them show only the few scalars that determine the final number. Do not paste the Hamiltonian, the local potential, the field arrays, or per-iteration tables.

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

plane_wave_grid

Goal
----
Build the reciprocal-space grid of the periodic cell: the signed plane-wave wavevectors in the ordering a discrete Fourier transform produces, their squared magnitudes, and the Coulomb factor of the Hartree kernel with its divergent component removed.

```python
import numpy as np


def plane_wave_grid(n_grid, cell_length, coulomb_scale):
    """n_grid: int >= 2, number of grid points. cell_length: float > 0, the cell length L.
    coulomb_scale: float, the scale s multiplying the Coulomb factor.
    Return an (n_grid, 3) real array whose columns are the signed wavevectors G, their
    squared magnitudes |G|^2, and the Coulomb factor v(G) with its G = 0 entry zeroed."""
    # Implement per the formulas above.
    grid = None
    return grid
```

### Step 2

ground_state_orbitals

Goal
----
Diagonalize the Kohn-Sham Hamiltonian and evaluate the Fermi-Dirac occupations of its eigenstates. Returns the eigenvectors, the ascending eigenvalues and the occupations packed into a single array, because every later step needs all three and the Studio driver compares one numeric object rather than a tuple.

```python
import numpy as np

def ground_state_orbitals(hamiltonian, mu, beta):
    """hamiltonian: (Nb,Nb) complex Hermitian. mu: float, chemical potential.
    beta: float > 0, inverse electronic temperature.
    Return one (Nb, Nb+2) complex array packing the eigenvectors in columns 0..Nb-1,
    the ascending eigenvalues in column Nb, and the occupations in column Nb+1.
    Unpack later with n = packed.shape[0]; U = packed[:, :n];
    w = packed[:, n].real; f = packed[:, n+1].real."""
    # Implement per the formulas above.
    packed = None
    return packed
```

### Step 3

sternheimer_solve

Goal
----
Solve the projected Sternheimer equation for the first-order response of one retained state to a local potential perturbation, by conjugate gradients run to a caller-supplied tolerance. This is the primitive whose cost the adaptive rule of the later steps controls.

```python
import numpy as np

def sternheimer_solve(hamiltonian, orbitals, eigenvalues, band_index, delta_V,
                      tol=1e-10, max_iter=200):
    """hamiltonian: (Nb,Nb) complex Hermitian. orbitals: (Nb,Nocc) complex, the retained
    states as orthonormal columns. eigenvalues: (Nocc,) float. band_index: int, which
    retained state to solve for. delta_V: (Nb,) real local potential perturbation.
    tol: float > 0, residual-norm stopping tolerance. max_iter: int, iteration cap.
    Return the (Nb,) complex projected response of that state."""
    # Implement per the formulas above.
    dphi = None
    return dphi
```

### Step 4

apply_chi0

Goal
----
Apply the independent-particle susceptibility to a local potential perturbation, at finite electronic temperature and at fixed total electron number, without ever assembling it as a matrix.

```python
import numpy as np

def apply_chi0(hamiltonian, orbitals, eigenvalues, occupations, delta_V, beta,
               tol=1e-10, max_iter=200):
    """hamiltonian: (Nb,Nb) complex Hermitian. orbitals: (Nb,Nocc) complex retained states.
    eigenvalues: (Nocc,) float. occupations: (Nocc,) float in [0,1]. delta_V: (Nb,) real.
    beta: float > 0. tol: float or (Nocc,) array of tolerances for the inner solves.
    max_iter: int, cap for each inner solve.
    Return the (Nb,) real first-order density response at fixed electron number."""
    # Implement per the formulas above.
    delta_rho = None
    return delta_rho
```

### Step 5

kerker_precondition

Goal
----
Apply the charge-conserving Kerker preconditioner to a vector. This is the operator the source paper puts in front of the Dyson equation to stop the condition number growing with system size, and the charge-conserving correction is the part that is easy to miss.

```python
import numpy as np


def kerker_precondition(vector, g_squared, alpha):
    """vector: (Ng,) real vector to precondition. g_squared: (Ng,) real non-negative squared
    wavevector magnitudes, as returned in column 1 of the grid step. alpha: float > 0, the
    Kerker screening parameter.
    Return the (Ng,) real preconditioned vector."""
    # Implement per the formulas above.
    preconditioned = None
    return preconditioned
```

### Step 6

adaptive_cg_tolerance

Goal
----
Compute the tolerance to which each inner Sternheimer solve may be run at the current outer iteration, using the source paper's guaranteed prefactor. This rule is the paper's contribution reduced to one function.

```python
import numpy as np


def adaptive_cg_tolerance(target_tol, residual_norm, sigma_min, iteration, restart_size,
                          occupations, orbitals, kernel_vector, cell_volume, n_grid, n_occ):
    """target_tol: float > 0, requested outer accuracy. residual_norm: float > 0, previous
    outer residual norm. sigma_min: float >= 0, running estimate of the smallest singular
    value of the Krylov Hessenberg matrix. iteration: int >= 0. restart_size: int > 0, the
    restart size m. occupations: (Nocc,) float in [0,1]. orbitals: (Ng,Nocc) complex retained
    states. kernel_vector: (Ng,) real, the kernel applied to the current Krylov vector.
    cell_volume: float > 0. n_grid: int. n_occ: int.
    Return the (Nocc,) array of per-state tolerances, np.inf where the occupation vanishes."""
    # Implement per the formulas above.
    tolerances = None
    return tolerances
```

### Step 7

preconditioned_gmres

Goal
----
Solve the preconditioned Dyson equation by restarted inexact GMRES: apply the Kerker preconditioner to both sides, run the Arnoldi process with an operator that is deliberately applied only approximately, maintain the running singular-value estimate the tolerance rule needs, and restart when the cycle is exhausted. The screening kernel is built here.

```python
import numpy as np


def preconditioned_gmres(hamiltonian, orbitals, eigenvalues, occupations, rhs,
                         coulomb_gspace, xc_local, g_squared, alpha, beta, target_tol,
                         cell_volume, restart_size=5, max_cycles=12, cg_max_iter=400):
    """hamiltonian, orbitals, eigenvalues, occupations: as in the susceptibility step.
    rhs: (Ng,) real right-hand side. coulomb_gspace: (Ng,) real Coulomb factor.
    xc_local: float contact coefficient. g_squared: (Ng,) real squared wavevectors.
    alpha: float > 0, Kerker parameter. beta: float > 0. target_tol: float > 0, requested
    outer accuracy. cell_volume: float > 0. restart_size: int > 0, the restart size m.
    max_cycles: int > 0, cap on restarts. cg_max_iter: int > 0, inner cap.
    Return the (Ng,) real interacting density response. The screening kernel is built here."""
    # Implement per the formulas above.
    delta_n = None
    return delta_n
```

### Step 8

response_observable

Goal
----
Run the whole calculation on the stated configuration and return the screened response of the observable: build the grid, diagonalise, form the bare response, and solve the preconditioned Dyson equation.

```python
import numpy as np


def response_observable(hamiltonian, delta_V_ext, observable, mu, beta, n_occ, n_grid,
                        cell_length, coulomb_scale, xc_local, alpha, target_tol,
                        restart_size=5):
    """hamiltonian: (Ng,Ng) complex Hermitian. delta_V_ext: (Ng,) real perturbation.
    observable: (Ng,) real. mu: float. beta: float > 0. n_occ: int in [1, Ng].
    n_grid: int. cell_length: float > 0. coulomb_scale: float. xc_local: float.
    alpha: float > 0, Kerker parameter. target_tol: float > 0. restart_size: int > 0.
    Return the screened response of the observable as a float."""
    # Implement per the formulas above.
    A = None
    return A
```
