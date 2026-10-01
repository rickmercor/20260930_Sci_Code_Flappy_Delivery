# Physics-Computational_Physics-18

## Background

*Phase-field crystal models.* A PFC free energy is minimised not by a constant but by a periodic field, because its quadratic part $\tfrac12\phi(\Delta+a)^2\phi$ has its Fourier symbol $\tfrac12(a-k^2)^2$ vanishing on the whole set $|k|^2=a$ - a circle in the two dimensions used here. That single feature is what makes the model resolve atomic-scale periodicity while evolving on diffusive time scales, and it is also why every discrete operator in the scheme is a fourth-order one: the symbol contains $k^4$, so an explicit treatment of the linear part would demand a time step scaling like the fourth power of the mesh size. A binary model carries two such fields and an interspecies term with its own length scale, which couples the two density equations at the highest order.

*Conserved Allen-Cahn dynamics.* Taking the gradient flow in $L^2$ rather than in $H^{-1}$ lowers the spatial order by two, but loses mass conservation. Subtracting the spatial mean of the chemical potential restores it: the correction is orthogonal to the constants, so it does not spoil the energy law, and it is nonlocal only in the weakest possible sense - it is constant in space, so its Fourier representation is as sparse as a nonlocal term can be.

*Invariant energy quadratization.* If the nonlinear part of an energy density is bounded from below, adding a large enough positive constant makes it positive, and its square root can be promoted to an independent unknown. The energy then becomes quadratic in the enlarged set of variables, the variational derivative becomes a product of a known coefficient and that unknown, and a semi-implicit discretisation that treats the coefficient explicitly and the unknown implicitly is linear and unconditionally energy stable by construction. The price is that the auxiliary unknown obeys its own evolution equation and slowly drifts away from the square root it started as, so the energy the scheme dissipates is a modified one.

*Zero-energy contribution.* A quadratized system is linear but still couples all its unknowns through the auxiliary variable. The trick that removes the coupling is to introduce one further unknown, a scalar function of time only, obeying an ODE whose right-hand side is a sum of pairs of $L^2$ inner products that cancel identically, so that the scalar is exactly one at the continuous level; multiplying selected terms by it therefore changes nothing. Discretely, the scalar is treated implicitly and its multiplicative factors explicitly, so every unknown becomes an affine function of that one scalar, each of the two parts solves a constant-coefficient problem, and one scalar equation closes the system at the end of the step.

*Fourier spectral discretisation of a constant-coefficient problem.* Once a scheme has been arranged so that each unknown satisfies a constant-coefficient equation on a periodic box, the solve is a division by the operator's symbol, mode by mode. What has to be checked is that the symbol never vanishes; for an operator assembled from a positive mass term, a bi-Laplacian and a shifted bi-Laplacian entering with either sign, the check is a one-line inequality in the wavenumber, and its worst case need not be at high frequency.

## Problem

A phase-field-crystal model resolves the atomic-scale periodicity of a crystal with a coarse-grained density field, so that dislocations, elasticity and grain boundaries emerge from a free energy rather than being inserted by hand. A published study extends that framework to a binary alloy in an applied magnetic field, and then builds for it a fully discrete time-marching scheme that is at once linear, fully decoupled - every unknown is obtained from its own constant-coefficient equation - second-order accurate in time and unconditionally energy stable, by quadratizing the nonlinear part of the energy and introducing a nonlocal auxiliary scalar whose exact value is one. This task asks you to run that scheme, exactly, on a fully specified deterministic instance. Nothing is fitted and nothing is random.

*The model.* Two density fields $\phi_1,\phi_2$ and a two-component magnetization $\boldsymbol M=(M_x,M_y)$ live on a periodic square $\Omega$. The total free energy is $E=\int_\Omega\big(F_B+F_{GL}+F_{Reg}\big)\,d\boldsymbol x$ with

$$F_B = \tfrac{\phi_1}{2}(\Delta+a_1)^2\phi_1 + \tfrac{\phi_2}{2}(\Delta+a_2)^2\phi_2 + \tfrac{\phi_1}{2}(\Delta+a_{12})^2\phi_2 + \tfrac14\phi_1^4 - \tfrac{\epsilon}{2}\phi_1^2 + \tfrac14\phi_2^4 - \tfrac{\epsilon}{2}\phi_2^2 + \tfrac{\eta}{3}\big(|\phi_1|^3+|\phi_2|^3-\phi_1^3-\phi_2^3\big) + \tfrac{\gamma_{12}}{2}\phi_1^2\phi_2^2,$$

$$F_{GL} = \tfrac{\omega_0}{2}|\nabla\boldsymbol M|^2 - \tfrac{\alpha}{2}|\boldsymbol M|^2 + \tfrac{\beta}{4}|\boldsymbol M|^4 - \boldsymbol M\cdot\boldsymbol H - \gamma_1|\boldsymbol M|^2\phi_1 - \gamma_2|\boldsymbol M|^2\phi_2 - \tfrac{\eta_1}{2}(\boldsymbol M\cdot\nabla\phi_1)^2 - \tfrac{\eta_2}{2}(\boldsymbol M\cdot\nabla\phi_2)^2,$$

$$F_{Reg} = \tfrac{\theta}{6}|\boldsymbol M|^6 + \tfrac{\theta_1}{4}|\nabla\phi_1|^4 + \tfrac{\theta_2}{4}|\nabla\phi_2|^4,$$

where $|\nabla\boldsymbol M|^2=\sum_{i,j}(\partial_i M_j)^2$ and $\boldsymbol H$ is a fixed applied field. The dynamics are the two mass-conserved Allen-Cahn equations and the one Allen-Cahn equation obtained by the energetic variational approach,

$$\partial_t\phi_i = -\mathcal M_\phi\Big(\mu_i - \tfrac{1}{|\Omega|}\int_\Omega\mu_i\,d\boldsymbol x\Big),\quad \mu_i=\frac{\delta E}{\delta\phi_i}, \qquad \partial_t\boldsymbol M = -\mathcal M_m\,\mu_3,\quad \mu_3=\frac{\delta E}{\delta\boldsymbol M},$$

with periodic boundary conditions; the nonlocal multipliers make $\int_\Omega\phi_1$ and $\int_\Omega\phi_2$ constant in time.

*Parameters.* $\epsilon=0.1$, $\mathcal M_\phi=1$, $\mathcal M_m=10^{-2}$, $\eta=10$, $\gamma_{12}=0.5$, $\omega_0=1$, $\theta=\theta_1=\theta_2=10^{-9}$, $\alpha=1$, $\beta=10$, $a_1=a_2=1$, $a_{12}=1.2$, $\gamma_1=\gamma_2=0.01$, $\eta_1=\eta_2=-0.1$. The scheme you are to run also needs two stabilization constants and one positive shift constant, whose roles you must find in the source; take $S_\phi=S_m=10$ and $B=10^7$.

*The instance.* $\Omega=[0,L]^2$ with $L=64$, discretized by a uniform $N\times N$ Fourier collocation grid with $N=64$: $x_i=i\,L/N$ and $y_j=j\,L/N$ for $i,j=0,\dots,N-1$, built with `numpy.meshgrid(..., indexing='ij')`. Write $\kappa=2\pi/L$. The initial data and the applied field are

$$\phi_1(\cdot,0)=\cos(8\kappa x)\sin(8\kappa y),\quad \phi_2(\cdot,0)=\cos(8\kappa x)\cos(8\kappa y),\quad \boldsymbol M(\cdot,0)=\big(\sin(2\kappa x)\sin(2\kappa y),\ \cos(2\kappa x)\cos(2\kappa y)\big),$$

$$\boldsymbol H=\big(\sin(2\kappa x)\cos(\kappa y),\ \cos(\kappa y)\big).$$

Advance **ten** steps of size $\delta t=2\times10^{-3}$, to $T=0.02$.

*Discretization conventions, fixed here so that the graded number is reproducible.* All spatial operators are evaluated spectrally: transform, multiply by the symbol built from the wavenumbers `2*numpy.pi*numpy.fft.fftfreq(N, d=L/N)` in each direction, with $N$ the number of grid points and $L$ the edge length in that direction, transform back and take the real part; the Nyquist wavenumber is kept exactly as that expression returns it and **no dealiasing of any kind is applied**. All nonlinear products are formed pointwise on the grid. Every $L^2$ inner product and every integral is the plain grid sum times the cell area $h_xh_y$. Where the source's scheme calls for a Crank-Nicolson average of a quantity, use $\tfrac12(\psi^{n+1}+\psi^{n})$; where it calls for a second-order extrapolation of a field, use $\psi^{*}=\tfrac32\psi^{n}-\tfrac12\psi^{n-1}$, and this same rule is applied to the auxiliary variable of the source's quadratization; where it calls for a second-order extrapolation of a time derivative to the half step, use $\psi^{*}_t=\big(2\psi^{n}-3\psi^{n-1}+\psi^{n-2}\big)/\delta t$, which is the unique three-point backward combination consistent at $t^{n+1/2}$. The same second-order formulas are used at every step including the first two, the two backward layers being supplied by the **exact initial velocities**: for each of $\phi_1,\phi_2,\boldsymbol M$ and the auxiliary variable, $\psi^{-k}=\psi^{0}-k\,\delta t\,\partial_t\psi(0)$ for $k=1,2$, where $\partial_t\phi_i(0)$ and $\partial_t\boldsymbol M(0)$ are the right-hand sides of the governing system evaluated on the initial data and $\partial_t U(0)$ comes from the auxiliary variable's own evolution equation. The initial chemical potentials are the source's reformulated ones evaluated on the initial data with the nonlocal scalar equal to its initial value; the chemical potentials of the two backward layers are never referenced by the scheme and need not be constructed.

*What to report.* Run the scheme for the ten steps and **report the total free energy $E$ of the model above, evaluated on the numerical solution at $T=0.02$, divided by the area of the domain**. **Find the source paper and take from it**: how the nonlinear part of the free energy is quadratized, that is, the auxiliary variable it introduces, what sits under its square root, why a positive shift constant is needed there, and the ordinary differential equation the auxiliary variable satisfies; the three coefficient functions that multiply it in the reformulated chemical potentials, and the reformulated chemical potentials themselves, including where the two stabilization constants enter and cancel; the nonlocal auxiliary scalar the decoupling introduces, the ordinary differential equation it satisfies, why its exact value is one, and which terms of the reformulated system are multiplied by it; the fully discrete scheme, that is, which quantities are Crank-Nicolson averaged, which are extrapolated, and how the nonlocal scalar is treated; the splitting of every unknown into a part independent of the nonlocal scalar and a part multiplying it, and the subsystems and explicit right-hand sides that result - four for the densities, two for the magnetization and two explicit updates of the quadratization's auxiliary variable; how the two subsystems that remain coupled through the interspecies length scale are decoupled, and how the implicit nonlocal integral term is then eliminated; the two scalars whose ratio gives the nonlocal scalar at the half step, and the relation between the half-step value and the new one; the modified discrete energy the scheme dissipates, including the two gradient-difference terms in $\phi_i^{n+1}-\phi_i^{n}$ that its Eq. (3.57) carries; and the lower bound the stabilization constant must satisfy. Do not substitute a plausible reconstruction: the graded number depends on the sign with which the shifted bi-Laplacian enters the difference system, on the nonlocal multiplier being retained rather than replaced by its exact continuous value, on the explicit right-hand sides being those the source's own conversion from Crank-Nicolson averages to new-layer unknowns produces - which old-layer quantities they carry, and with which coefficients - and on both auxiliary variables being advanced rather than frozen.

State in your reasoning: the auxiliary variable and the three coefficient functions you fetched, and the ordinary differential equation the nonlocal scalar satisfies; the Fourier symbol of the constant-coefficient operator you invert for the sum system and for the difference system, and the argument that the second is nonsingular for every wavenumber; the lower bound the stabilization constant must satisfy for this parameter set, and whether the stated value satisfies it; how you obtained the mean of the sum unknown before solving, and how the implicit nonlocal term is eliminated in the Fourier representation; the initial and final values of the free-energy density and of the modified discrete energy density, and the fact that the modified energy agrees with the original one at $t=0$ - **for the reported initial modified energy only, set both previous-density arguments of the modified energy equal to the current initial densities, so that its two gradient-difference correction terms are identically zero; retain the prescribed backward ghost layers for the time stepping itself**, which is what makes that agreement a diagnostic of the two energy expressions rather than of the start-up layers; whether both energies decreased at every step; the final value of the nonlocal scalar and how far it is from one; the two mass drifts; and the values the same pipeline returns after zero steps, after five steps, and after twenty steps of half the size.

Output Format Requirements: Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure. Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: `0.4847`, `12.6`, `1.05`). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep `<reasoning>` short (a few hundred words): the diagnostics listed above are its required content and must all appear, and beyond them show only the few scalars that determine the final number. Do not paste the field arrays, the Fourier coefficients, or per-step tables of more than the two energies.

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

pfc_spectral_derivatives

Goal
----
Evaluate, by the Fourier spectral method on a uniform periodic grid, five spatial operators applied to one real periodic field $u$: the two first partial derivatives $\\partial_x u$ and $\\partial_y u$, the Laplacian $\\Delta u$, the bi-Laplacian $\\Delta^2 u$, and the ****shifted bi-Laplacian**** $(\\Delta+a)^2u$ for a real shift $a$. Return all five stacked along a new leading axis, in that order, so that the shape is `(5,) + u.shape`. Every operator is applied by transforming, multiplying by its symbol, transforming back and taking the real part; ****no dealiasing of any kind is applied**** and the Nyquist wavenumber is kept exactly as `numpy.fft.fftfreq` returns it. The shifted bi-Laplacian is graded separately from the bi-Laplacian because the whole interspecies coupling of the model rides on it, and because expanding it as $\\Delta^2+a^2$ - dropping its two cross terms - is the single most tempting simplification in the pipeline.

```python
import numpy as np

def pfc_spectral_derivatives(u, cell, a=0.0):
    """u: real periodic field on a uniform grid, shape (Nx, Ny).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    a: real shift of the shifted bi-Laplacian (Delta + a)^2.
    Return the real array [du/dx, du/dy, lap u, bilap u, (Delta+a)^2 u]
    of shape (5,) + u.shape."""
    # Implement per the specification above.
    return None
```

### Step 2

pfc_free_energy

Goal
----
Evaluate the ****total free energy**** of the magnetic-coupled binary phase-field-crystal model on the grid: the sum of the binary PFC block, the Ginzburg-Landau magnetic block and the higher-order regularization block, integrated over the periodic domain. Return one float. Every integral is the plain grid sum times the cell area $h_xh_y$, and every spatial operator is the spectral one of step 1. This is the quantity the whole benchmark finally reports, and it is also the quantity the modified discrete energy of step 7 must reproduce at $t=0$, so an error here shows up twice.

```python
import numpy as np

def pfc_free_energy(phi1, phi2, M, cell, params=None, H=None):
    """phi1, phi2: density fields of equal shape (Nx, Ny).
    M: magnetization, shape (2, Nx, Ny). H: applied field, same shape as M;
       None means no applied field.
    cell: domain edge lengths, a scalar or a length-2 sequence.
    params: dict of model-parameter overrides; None means the fixed set
       eps, M_phi, M_m, eta, gamma12, omega0, theta, theta1, theta2, alpha,
       beta, S_phi, S_m, a1, a2, a12, B, gamma1, gamma2, eta1, eta2.
    Return the total free energy of the model, a float.
    Raise ValueError for mismatched or non-finite fields, invalid domain
    lengths, or invalid model parameters. H=None means a zero applied field."""
    # Implement per the specification above.
    return None
```

### Step 3

pfc_nonlinear_terms

Goal
----
Evaluate the ****nonlinear part**** $N$ of the free-energy density - everything except the three blocks that are quadratic in derivatives of the fields - together with its three variational derivatives $N_1=\\delta N/\\delta\\phi_1$, $N_2=\\delta N/\\delta\\phi_2$ and $\\boldsymbol N_3=\\delta N/\\delta\\boldsymbol M$. Return them stacked along a new leading axis in the order $[N,\\ N_1,\\ N_2,\\ N_{3x},\\ N_{3y}]$, so that the shape is `(5,) + phi1.shape`. These four objects are what the quadratization of the next step is built from: $N$ sits under the square root and the three derivatives sit in the numerators of the three coefficient functions.

```python
import numpy as np

def pfc_nonlinear_terms(phi1, phi2, M, cell, params=None, H=None):
    """Arguments exactly as in pfc_free_energy.
    Return the real array [N, dN/dphi1, dN/dphi2, (dN/dM)_x, (dN/dM)_y]
    of shape (5,) + phi1.shape.
    Raise ValueError for mismatched or non-finite fields, invalid domain
    lengths, or invalid model parameters. H=None means a zero applied field."""
    # Implement per the specification above.
    return None
```

### Step 4

pfc_ieq_coefficients

Goal
----
Evaluate the ****invariant-energy-quadratization auxiliary variable**** $U$ and the three coefficient functions $H_1$, $H_2$ and $\\boldsymbol R$ that multiply it in the reformulated chemical potentials. Return them stacked along a new leading axis in the order $[U,\\ H_1,\\ H_2,\\ R_x,\\ R_y]$, so that the shape is `(5,) + phi1.shape`. The radicand must be strictly positive: raise a `ValueError` if it is not, since a non-positive radicand means the shift constant $B$ was chosen too small for the state, and that is a modelling failure rather than a number to be returned.

```python
import numpy as np

def pfc_ieq_coefficients(phi1, phi2, M, cell, params=None, H=None):
    """Arguments exactly as in pfc_free_energy.
    Return the real array [U, H1, H2, R_x, R_y] of shape (5,) + phi1.shape.
    Raise ValueError if the radicand is not strictly positive everywhere.
    Raise ValueError for mismatched or non-finite input fields, invalid
    domain lengths or parameters, or a non-finite IEQ radicand."""
    # Implement per the specification above.
    return None
```

### Step 5

pfc_split_solve

Goal
----
Solve ****one**** of the four decoupled constant-coefficient density subsystems of the scheme. After the sum-and-difference transformation and after the chemical potential has been eliminated, each subsystem is a single biharmonic-type equation in one unknown $\\varphi$, driven by two explicit right-hand sides $G_{13}$ and $G_{24}$ and carrying a sign $s=\\pm1$ in front of the shifted bi-Laplacian - $s=+1$ for a sum system, $s=-1$ for a difference system. Return the solution and the corresponding potential stacked along a new leading axis in the order $[\\varphi,\\ \\mu]$, so that the shape is `(2,) + G13.shape`. The equation contains one ****implicit nonlocal term****, and part of the task is to see that its value is available in closed form before the solve.

```python
import numpy as np

def pfc_split_solve(G13, G24, sign, cell, dt, params=None):
    """G13, G24: the two explicit right-hand sides, shape (Nx, Ny).
    sign: +1 for a sum system, -1 for a difference system.
    cell: domain edge lengths, a scalar or a length-2 sequence.
    dt: the time step. params: as in pfc_free_energy.
    Return the real array [phi, mu] of shape (2,) + G13.shape.
    Raise ValueError if G13 and G24 are not finite two-dimensional arrays
    of equal shape, if sign is not +1 or -1, if dt is not a finite
    positive scalar, or if the split operator is singular for the given
    parameters."""
    # Implement per the specification above.
    return None
```

### Step 6

pfc_time_step

Goal
----
Advance the whole system by ****one**** step of the fully discrete Fourier-IEQ-ZEC scheme. The state is the three most recent layers, each holding the nine fields $[\\phi_1,\\ \\phi_2,\\ M_x,\\ M_y,\\ \\mu_1,\\ \\mu_2,\\ \\mu_{3x},\\ \\mu_{3y},\\ U]$ in that order, so `state[0]` is layer $n$, `state[1]` is layer $n-1$ and `state[2]` is layer $n-2$; the nonlocal scalar $Q^n$ is passed separately. Build the second-order extrapolations, form the three coefficient functions at the extrapolated state, assemble the six explicit right-hand sides, solve the four density subsystems of step 5 and the two magnetization subsystems, evaluate the two nonlocal scalars, close the step for the nonlocal scalar and recombine. Return the ten planes $[\\phi_1,\\ \\phi_2,\\ M_x,\\ M_y,\\ \\mu_1,\\ \\mu_2,\\ \\mu_{3x},\\ \\mu_{3y},\\ U,\\ Q^{n+1}]$ of the new layer, the last plane being constant and equal to the new nonlocal scalar, so that the shape is `(10, Nx, Ny)`.

```python
import numpy as np

def pfc_time_step(state, q_n, cell, dt, params=None, H=None):
    """state: array of shape (3, 9, Nx, Ny); state[k] is layer n-k and holds
       [phi1, phi2, M_x, M_y, mu1, mu2, mu3_x, mu3_y, U] in that order.
    q_n: the nonlocal scalar at layer n, a float.
    cell: domain edge lengths. dt: the time step.
    params: as in pfc_free_energy. H: applied field, shape (2, Nx, Ny).
    Return the real array of shape (10, Nx, Ny) holding the nine fields of
    the new layer followed by a constant plane equal to Q^{n+1}.
    Raise ValueError if state is not a finite array of shape
    (3, 9, Nx, Ny), if q_n is not a finite scalar, if dt is not a finite
    positive scalar, if H is given with a shape other than (2, Nx, Ny),
    or if the equation for the nonlocal scalar is singular."""
    # Implement per the specification above.
    return None
```

### Step 7

pfc_modified_energy

Goal
----
Evaluate the ****modified discrete energy**** that the scheme dissipates - the quantity the source's discrete energy law bounds, which is the original free energy with its nonlinear part replaced by the squared norm of the auxiliary variable, plus the two quadratic stabilizer terms, plus the square of the nonlocal scalar, minus the two constants those replacements introduce, plus the two gradient-difference terms that the second-order time discretisation contributes. Return one float. The previous-layer densities are optional and default to the current ones, in which case the two gradient-difference terms vanish; that default is what the initial layer needs. ****Every squared gradient norm in this expression - the two $\\|\\nabla\\phi_i\\|^2$, the $\\|\\nabla\\boldsymbol M\\|^2$ and the two gradient-difference terms - is to be evaluated as $-(u,\\Delta u)$, not as the integral of the squared pointwise first derivatives.**** The two agree in the continuum and differ on the grid, and the choice is graded.

```python
import numpy as np

def pfc_modified_energy(phi1, phi2, M, U, q, cell, params=None,
                        phi1_prev=None, phi2_prev=None):
    """phi1, phi2, U: fields of shape (Nx, Ny). M: shape (2, Nx, Ny).
    q: the nonlocal scalar, a float. cell: domain edge lengths.
    params: as in pfc_free_energy.
    phi1_prev, phi2_prev: the previous-layer densities; None means equal to
       the current ones, so the two gradient-difference terms vanish.
    Return the modified discrete energy, a float.
    Raise ValueError for mismatched or non-finite fields (including optional
    previous densities), a non-finite or non-scalar q, invalid domain lengths,
    or invalid model parameters."""
    # Implement per the specification above.
    return None
```

### Step 8

pfc_ieq_zec_run

Goal
----
Run the whole benchmark. Build the initial data and the applied field from the closed form below; evaluate the auxiliary variable and the three coefficient functions on them; assemble the initial chemical potentials of the reformulated system with the nonlocal scalar at its initial value $Q^0=1$; build the two backward ghost layers from the exact initial velocities so that the second-order formulas apply from the very first step; advance $n_steps$ steps of size $dt$ with the step of step 6; and return the ****total free energy of the final layer divided by the area of the domain****. Setting `quantity` selects a diagnostic instead: `'energy0'` the initial free-energy density, `'emod'` and `'emod0'` the modified discrete energy density at the final and the initial layer, `'q'` the final nonlocal scalar, `'mass'` the total drift of the two spatial means over the run, and `'drift'` the sup-norm distance between the auxiliary variable carried by the scheme and the square root recomputed from the final fields.

```python
import numpy as np

def pfc_ieq_zec_run(n_steps=10, dt=2.0e-3, grid=(64, 64), cell=(64.0, 64.0), params=None, quantity="energy"):
    """n_steps: number of time steps. dt: the time step.
    grid: (Nx, Ny). cell: domain edge lengths (Lx, Ly).
    params: dict of model-parameter overrides; None means the fixed set.
    quantity: 'energy' (default, the final free-energy density), 'energy0',
       'emod', 'emod0', 'q', 'mass' or 'drift'.
    Return the requested scalar, a float.
    Raise ValueError if n_steps is not a non-negative integer, if dt is
    not a finite positive scalar, if either grid dimension is below four,
    or if quantity is not one of the names listed above.
    Integer-valued real scalars are accepted for n_steps and grid dimensions;
    other values, missing dimensions and non-finite values raise ValueError."""
    # Implement per the specification above.
    return None
```
