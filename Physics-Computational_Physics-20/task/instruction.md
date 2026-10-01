# Physics-Computational_Physics-20

## Background

*Phase-field crystal models.* A phase-field-crystal free energy is minimised not by a constant but by a periodic field, because its quadratic part $\tfrac12\phi(\Delta+1)^2\phi$ has Fourier symbol $\tfrac12(1-|k|^2)^2$, which vanishes on the whole set $|k|=1$ - a circle in the two dimensions used here. That single feature is what makes the model resolve atomic-scale periodicity while evolving on diffusive time scales, and it is also why the quadratic block controls nothing on that circle: whatever selects the amplitude of the crystal has to come from the quartic term and from the temperature parameter. It is also why every discrete operator here is fourth order in the wavenumber, so an explicit treatment of the linear part would demand a time step scaling like the fourth power of the mesh size, and a sixth-order one once the $H^{-1}$ mobility is included.

*The long-range block.* Adding a term quadratic in the inverse Laplacian of the deviation from the mean penalises long-wavelength segregation, which is the standard device for microphase separation in block-copolymer melts. On a periodic box it is the weakest possible nonlocality: the operator is diagonal in Fourier space, the solvability condition is exactly that the right-hand side has zero mean, and the solution is unique only after its own mean is fixed - so a convention has to be chosen, and the natural one is to set the zero mode to zero.

*Conserved gradient-flow dynamics.* Taking the gradient flow in $H^{-1}$ rather than in $L^2$ raises the spatial order by two but makes the flow conservative for free: the time derivative is a Laplacian, whose integral over a periodic box vanishes identically. Discretely that is a statement about one Fourier mode - the mobility symbol is zero at $k=0$ - so a spectral scheme inherits exact mass conservation without any correction, provided every increment it forms carries that symbol as a factor.

*Quadratization by a scalar auxiliary variable.* If the nonlinear part of an energy is bounded from below, adding a large enough positive constant makes it positive and its square root can be promoted to an independent scalar unknown. The energy then becomes quadratic in the enlarged set of variables, the corresponding part of the variational derivative becomes a product of a known field and that unknown, and a discretisation which treats the field explicitly and the unknown implicitly is linear and unconditionally energy stable by construction. The price is that the auxiliary unknown obeys its own evolution equation and slowly drifts away from the square root it started as, so the energy the scheme provably dissipates is a modified one, equal to the true energy at the initial time and not afterwards.

*Runge-Kutta methods with two tableaux.* A problem split into a stiff linear part and a nonstiff nonlinear part can be integrated by carrying one tableau for each. If the tableau of the linear part is lower triangular including its diagonal, each stage is a constant-coefficient implicit solve - a division by a symbol, on a periodic box - and the method is stable at large step sizes; if the tableau of the nonlinear part is strictly lower triangular, the nonlinear terms are always evaluated on stage values already known, so no nonlinear iteration is needed anywhere. Accuracy is then not a property of either tableau but a set of polynomial identities coupling the two, more numerous than for a single tableau because a split problem has more elementary differentials at each order. Energy stability is a separate and far simpler condition on one symmetric matrix built from the implicit tableau and the weights. Because the two conditions are independent, they are typically satisfied by a whole family of coefficient sets rather than by an isolated one - and different members of such a family are genuinely different schemes, agreeing in order and in stability while producing different trajectories at any finite step size.

## Problem

A phase-field-crystal model resolves the atomic-scale periodicity of a crystal with a coarse-grained density field, so that elasticity, dislocations and grain boundaries emerge from a free energy rather than being inserted by hand. A published study takes the *modified* phase-field-crystal equation - the one whose free energy carries, on top of the usual crystal block, a long-range interaction of the kind that drives microphase separation in block-copolymer melts - and builds for it a four-stage Runge-Kutta time-marching method that is at once linear at every stage, exactly mass conserving, unconditionally energy stable and **third-order** accurate in time, by linearising the nonlinear part of the energy through a scalar auxiliary variable and by carrying a **different Runge-Kutta coefficient tableau for the linear and for the nonlinear part** of the equation. This task asks you to run that scheme, exactly, on a fully specified deterministic instance. Nothing is fitted and nothing is random.

*The model.* One atomic density field $\phi$ lives on a periodic rectangle $\Omega$. Its free energy is

$$E(\phi)=\int_\Omega\Big(\tfrac14\phi^4+\tfrac{1-\epsilon}{2}\phi^2-|\nabla\phi|^2+\tfrac12(\Delta\phi)^2\Big)\,d\boldsymbol x+\frac{\alpha}{2}\int_\Omega\!\!\int_\Omega\big(\phi(\boldsymbol x)-\bar\phi\big)\,\Gamma(\boldsymbol x-\boldsymbol y)\,\big(\phi(\boldsymbol y)-\bar\phi\big)\,d\boldsymbol y\,d\boldsymbol x,$$

where $\bar\phi=\int_\Omega\phi\,d\boldsymbol x/|\Omega|$ is the mean density, $\epsilon>0$ is the atomic thickness parameter, $\alpha>0$ measures the long-range interaction, and $\Gamma$ is the Green function of the periodic Poisson problem $-\Delta\Gamma=\delta$. Equivalently, writing $\chi$ for the **mean-free** periodic solution of $-\Delta\chi=\phi-\bar\phi$, the long-range block is $\tfrac{\alpha}{2}\int_\Omega|\nabla\chi|^2\,d\boldsymbol x$. The dynamics are the $H^{-1}$ gradient flow of that energy,

$$\frac{\partial\phi}{\partial t}=\mathcal M\,\Delta\mu,\qquad \mu=\phi^3-\epsilon\phi+(\Delta+1)^2\phi+\alpha\chi,$$

with periodic boundary conditions and a positive constant mobility $\mathcal M$. This flow conserves $\int_\Omega\phi$ and decreases $E$.

*Parameters.* $\mathcal M=10$, $\epsilon=0.4$, $\alpha=0.25$. The scheme you are to run also needs one positive shift constant, whose role you must find in the source; take $C_{SAV}=400$.

*Coefficients - read this carefully.* The source displays one explicit pair of Runge-Kutta coefficient tableaux, and then, separately, the **general family** of coefficient pairs that its energy-stability and third-order-accuracy proofs actually permit, parameterised by five free real constants - in the source's own notation $\tilde a_{11},\tilde a_{32},\tilde a_{33},a_{31},a_{43}$, of which three must be nonzero. **Do not use the displayed pair.** Instantiate the family at

$$\tilde a_{11}=\tfrac12,\qquad \tilde a_{32}=-\tfrac14,\qquad \tilde a_{33}=\tfrac32,\qquad a_{31}=\tfrac13,\qquad a_{43}=\tfrac23,$$

and run that member. It is a different third-order, unconditionally energy-stable scheme, and it gives a different trajectory: using the displayed pair instead moves the graded number by about one part in a thousand.

*The instance.* $\Omega=[0,L_x]\times[0,L_y]$ with $L_x=32$ and $L_y=48$, discretized by a uniform $N_x\times N_y$ Fourier collocation grid with $N_x=48$ and $N_y=64$, so that the two mesh spacings differ: $x_i=i\,L_x/N_x$ and $y_j=j\,L_y/N_y$ for $i=0,\dots,N_x-1$ and $j=0,\dots,N_y-1$, built with `numpy.meshgrid(..., indexing='ij')`. Write $\kappa_x=2\pi/L_x$ and $\kappa_y=2\pi/L_y$. The initial density is

$$\phi(\cdot,0)=0.15+0.30\cos(6\kappa_xx)\cos(4\kappa_yy)+0.20\sin(2\kappa_xx)\sin(6\kappa_yy).$$

Advance **sixteen** steps of size $\delta t=0.125$, to $T=2$.

*Discretization conventions, fixed here so that the graded number is reproducible.* All spatial operators are evaluated spectrally: transform, multiply by the symbol built from the wavenumbers `2*numpy.pi*numpy.fft.fftfreq(N, d=L/N)` in each direction - with $N$ the number of grid points and $L$ the edge length **in that direction**, so the two wavenumber grids are different - transform back and take the real part; the Nyquist wavenumber is kept exactly as that expression returns it and **no dealiasing of any kind is applied**. All nonlinear products are formed pointwise on the grid. Every $L^2$ inner product and every integral over $\Omega$ is the plain grid sum times the cell area $h_xh_y$, with $h_x=L_x/N_x$ and $h_y=L_y/N_y$. The field $\chi$ is obtained by dividing the transform of $\phi$ by $|k|^2$ at every nonzero wavenumber and **setting its zero Fourier mode to zero**, which is what makes it mean-free and simultaneously performs the subtraction of $\bar\phi$. Every squared gradient norm appearing in an energy is read as **minus the inner product with the discrete Laplacian** rather than as the integral of the squared pointwise first derivatives - for the long-range block this is the identity $-(\chi,\Delta_h\chi)_h=(\chi,\phi-\bar\phi)_h$, and it is the reading under which the scheme's discrete energy law is an identity. The auxiliary variable of the source's reformulation is initialised as the exact square root it is defined to be, evaluated on the initial density. The same coefficient family member is used at every step, including the first: the scheme is a one-step method and needs no start-up procedure.

*What to report.* Run the scheme for the sixteen steps and **report the total free energy $E$ of the model above, evaluated on the numerical solution at $T=2$, divided by the area of the domain**. **Find the source paper and take from it**: how the nonlinear part of the free energy is separated from the quadratic part, and which of the two the scheme keeps implicit; the scalar auxiliary variable the reformulation introduces, what sits under its square root, why a positive shift constant is needed there, and the ordinary differential equation it satisfies; the nonlinear map that multiplies it in the reformulated chemical potential; the fully discrete four-stage scheme in its two-tableau form, that is, which quantities are accumulated with the implicit tableau, which with the explicit one, and how far each accumulation runs, together with the single weight vector the two tableaux share and whether the abscissae enter the scheme at all; the per-stage elimination that turns each implicit stage into a single division by a constant-coefficient Fourier symbol, the two fields of the resulting affine decomposition, and the scalar equation that closes the stage; the modified discrete energy the scheme dissipates, the matrix condition on the implicit tableau and the weights that makes it dissipate, and the algebraic identity behind that theorem, which splits the change in the quadratic part of the energy over one step into a term linear in the stage increments and a quadratic form in them; the nine order conditions that make the method third order; and the general coefficient family, from which you are to instantiate the member specified above, together with the reason such a family exists at all. Do not substitute a plausible reconstruction. Several of the choices in that list have a near miss that reads just as naturally, differs in one index, one constant or one term, and moves the graded number; the value you report is the only place that difference will show.

State in your reasoning: the source paper you used and at least one primary source for the model itself, each identified precisely enough to be located unambiguously, and what each supplies; the split of the free energy into the block the scheme keeps implicit and the nonlinear remainder, and how you fixed the nonlocal potential at the zero Fourier mode; the auxiliary variable and the nonlinear map you fetched, and the ordinary differential equation the auxiliary variable satisfies; the two instantiated tableaux and the weight vector, and the fact that two entries which look free are determined by the other constants; the largest of the nine order-condition residuals you computed, and the four eigenvalues of the energy-stability matrix, together with the observation that neither depends on the five constants; the two Fourier symbols you multiply by and the symbol you divide by at each stage, and the one-line argument that it never vanishes for a positive diagonal entry and a positive step; the three per-stage accumulations, which tableau each of them uses and how far each of them runs, and the constant that stands in the denominator of the stage's scalar equation; why every stage increment has exactly zero spatial mean, and what that implies for the mass; the initial and final values of the free-energy density and of the modified discrete energy density, and the fact that the two energies agree at $t=0$ and why; whether both energies decreased at every step, and which of the two the source's theorem actually guarantees; the final value of the auxiliary variable and its distance from the square root recomputed on the final density; the relative mass drift; and the values the same pipeline returns after zero steps, after eight steps, and after thirty-two steps of half the size.

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

mpfc_spectral_operators

Goal
----
Evaluate, by the Fourier spectral method on a uniform periodic grid, a fixed list of spatial operators applied to one real periodic field $u$. The field is $d$-dimensional for ****any**** $d\\ge1$, not two-dimensional only, and the list is: the $d$ first partial derivatives $\\partial_{x_0}u,\\dots,\\partial_{x_{d-1}}u$ in axis order, then the Laplacian $\\Delta u$, the bi-Laplacian $\\Delta^2u$, the ****shifted bi-Laplacian**** $(\\Delta+1)^2u$, and the ****mean-free inverse Laplacian**** $\\chi[u]$, defined as the unique periodic solution of $-\\Delta\\chi = u-\\bar u$ with $\\int_\\Omega\\chi = 0$, where $\\bar u$ is the spatial mean of $u$. Return them all stacked along a new leading axis, in that order, so that the shape is `(d + 4,) + u.shape` - six planes when $d=2$, five when $d=1$, seven when $d=3$. Each operator is applied by transforming, multiplying by its symbol, transforming back and taking the real part; ****no dealiasing of any kind is applied**** and the Nyquist wavenumber is kept exactly as `numpy.fft.fftfreq` returns it, which matters on every axis with an even number of points. The last two entries are graded separately from the derivatives because the whole model rides on them: the shifted bi-Laplacian is the operator whose symbol is degenerate on a sphere of wavenumbers, and the mean-free inverse Laplacian is the only operator here whose symbol has to be defined by hand at $k=0$.

```python
def mpfc_spectral_operators(u: "np.ndarray", cell: "float | tuple") -> "np.ndarray":
    """u: real periodic field on a uniform grid of ANY dimension d >= 1,
       shape (N_0, ..., N_{d-1}).
    cell: domain edge lengths, a scalar or a length-d sequence.
    Return the real array holding, in order, the d first partial
    derivatives of u in axis order, then lap u, bilap u, (Delta+1)^2 u
    and chi, of shape (d + 4,) + u.shape, where chi is the mean-free
    periodic solution of -lap chi = u - mean(u).
    Raise ValueError if u is not a finite real array of at least one
    dimension with at least two points along every axis, or if cell is
    not positive, or is a sequence whose length is not u.ndim."""
    # Implement per the specification above.
    return None
```

### Step 2

mpfc_free_energy

Goal
----
Evaluate the discrete free energy of the model on one density field and split it into the two parts the scheme treats differently. Return the real array $[E,\\;E_{\\rm quad},\\;E_1,\\;\\bar\\phi]$ of shape `(4,)`, where $E_{\\rm quad}=\\tfrac12(\\phi,(\\Delta+1)^2\\phi)_h$ is the quadratic block, $E_1$ is everything else - the quartic bulk term, the temperature term and the long-range term - $E=E_{\\rm quad}+E_1$ is the total free energy of the model as printed in the problem statement, and $\\bar\\phi$ is the spatial mean of $\\phi$. The split is not cosmetic: the scheme of the source treats $E_{\\rm quad}$ implicitly through its Fourier symbol and $E_1$ through an auxiliary variable, so an implementation that puts one term on the wrong side of the split is running a different scheme.

```python
def mpfc_free_energy(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """phi: real periodic density field, of ANY dimension d >= 1.
    cell: domain edge lengths, a scalar or a length-d sequence.
    params: dict of overrides for the model parameters
       M (mobility), eps, alpha and C_sav; None means the fixed set.
       M must be strictly positive and alpha must be non-negative;
       eps and C_sav may take any finite value, the positivity of
       E1 + C_sav being checked only where a square root is taken.
    Return the real array [E, E_quad, E1, mean(phi)] of shape (4,).
    Raise ValueError if phi is not a finite real array of rank at least one, if
    cell is not a positive scalar or one length per axis, or if
    params carries an
    unknown key, a non-finite value, a non-positive M or a negative
    alpha."""
    # Implement per the specification above.
    return None
```

### Step 3

mpfc_sav_terms

Goal
----
Evaluate the two field-valued quantities the time-stepping needs at a given density layer. Return the real array $[\\,\\mathcal H(\\phi),\\;\\mu(\\phi)\\,]$ of shape `(2,) + phi.shape`, where $\\mu$ is the ****chemical potential of the original model****, the variational derivative of the free energy of step 2, and $\\mathcal H$ is the ****nonlinear map of the source's scalar-auxiliary-variable reformulation****: the variational derivative of $E_1$ alone, divided by the square root of $E_1$ shifted by the positive constant $C_{SAV}$. Both must be evaluated at the **same** argument: in particular the radicand is $E_1$ of the field passed in, never $E_1$ of some earlier layer. Raise `ValueError` if that radicand is not strictly positive, which is the condition on $C_{SAV}$ the source states and the only thing that makes the reformulation legitimate.

```python
def mpfc_sav_terms(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """phi: real periodic density field, of ANY dimension d >= 1.
    cell: domain edge lengths, a scalar or a length-d sequence.
    params: as in mpfc_free_energy.
    Return the real array [H(phi), mu(phi)] of shape (2,) + phi.shape.
    Raise ValueError if phi is not a finite real array of rank at least one, if
    cell is not a positive scalar or one length per axis, if params
    is invalid, or if
    E1(phi) + C_sav is not strictly positive."""
    # Implement per the specification above.
    return None
```

### Step 4

mpfc_butcher

Goal
----
Build the pair of Butcher tableaux this scheme runs on from the source's ****general coefficient family****, and certify it. The source displays one explicit pair of tableaux and then, separately, the family of pairs its stability and accuracy proofs actually permit, parameterised by five free real constants - in the source's own notation $\\tilde a_{11},\\tilde a_{32},\\tilde a_{33},a_{31},a_{43}$, of which three must be nonzero. Given those five constants, return the concatenated real array of shape `(49,)` holding, in this order: the $4\\times4$ implicit tableau $\\tilde A$ flattened row by row (16 entries), the $4\\times4$ explicit tableau $A$ flattened row by row (16 entries), the weight vector $b$ (4 entries), the residuals of the source's ****nine**** order conditions for third-order temporal accuracy, each written as left-hand side minus right-hand side and taken in the source's own order (9 entries), and the four eigenvalues of the energy-stability matrix $P=D\\tilde A+\\tilde A^{T}D-bb^{T}$ with $D=\\operatorname{diag}(b)$, sorted ascending (4 entries). Raise `ValueError` if any of the three constants the family requires to be nonzero is zero.

```python
def mpfc_butcher(a11t: "float | None" = None, a32t: "float | None" = None,
        a33t: "float | None" = None, a31: "float | None" = None,
        a43: "float | None" = None) -> "np.ndarray":
    """a11t, a32t, a33t, a31, a43: the five free real constants of the
       source's general coefficient family; None means this task's
       prescribed value for that constant.
    Return the real array of shape (49,) holding, in order: Atilde
    flattened (16), A flattened (16), b (4), the nine order-condition
    residuals (9) and the four eigenvalues of P sorted ascending (4).
    Raise ValueError if any constant is not finite, or if any of
    a11t, a33t and a43 - the three the family requires to be nonzero -
    is zero.  a32t and a31 may be zero."""
    # Implement per the specification above.
    return None
```

### Step 5

mpfc_stage_solve

Goal
----
Carry out one Runge-Kutta stage of the source's scheme in Fourier space. The stage is implicit in two coupled unknowns - a field increment and a scalar increment - and the source eliminates them by writing the field increment as an affine function of the scalar. Given the ****accumulated density**** $\\psi$ that the implicit tableau builds for this stage, the nonlinear map $\\mathcal H$ of step 3 already evaluated at this stage's explicit argument, the ****accumulated auxiliary variable**** $\\rho$, the diagonal entry $\\tilde a_{ll}$ of the implicit tableau and the step $\\delta t$, return the real array $[\\mathcal A,\\;\\mathcal B,\\;\\phi_{k},\\;r_{k}]$ of shape `(4,) + acc.shape` - the argument named `acc` in the signature is that $\\psi$ - where $\\mathcal A$ and $\\mathcal B$ are the two fields of the source's affine decomposition, $\\phi_k$ is this stage's density increment and the last plane is ****constant****, equal to this stage's scalar increment $r_k$.

```python
def mpfc_stage_solve(acc: "np.ndarray", Hl: "np.ndarray", rho: float, a_ll: float,
        dt: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """acc: the accumulated density psi of this stage, any dimension.
    Hl: the nonlinear map H evaluated at this stage's explicit
       argument, same shape.
    rho: the accumulated auxiliary variable of this stage, a float.
    a_ll: the diagonal entry of the implicit tableau for this stage.
    dt: the time step. cell: domain edge lengths, a scalar or one length per axis.
    params: as in mpfc_free_energy.
    Return the real array [A, B, phi_k, r_k] of shape (4,) + acc.shape,
    whose last plane is constant and equal to r_k.
    Raise ValueError if acc and Hl are not finite real arrays of rank at least one
    of the same shape, if rho or a_ll is not a finite scalar, if dt is
    not a finite positive scalar, if params is invalid, or if either the
    stage operator or the scalar equation is singular."""
    # Implement per the specification above.
    return None
```

### Step 6

mpfc_time_step

Goal
----
Advance the pair (density, auxiliary variable) by one step of the source's four-stage scheme. Build the two tableaux and the weights with step 4; for each of the four stages assemble the three arguments step 5 needs - the density accumulated with the ****implicit**** tableau, the nonlinear map $\\mathcal H$ of step 3 evaluated at the density accumulated with the ****explicit**** tableau, and the auxiliary variable accumulated with the ****implicit**** tableau - solve the stage, and combine the four stage increments with the weights. Return the real array $[\\phi^{n+1},\\;r^{n+1}]$ of shape `(2,) + phi.shape`, whose second plane is ****constant**** and equal to the new auxiliary variable. Which tableau feeds which argument, and whether the diagonal entry is included in each accumulation, are the two things this step is graded on.

```python
def mpfc_time_step(phi: "np.ndarray", r: float, dt: float, cell: "float | tuple",
        params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    """phi: the density at layer n, any dimension d >= 1.
    r: the auxiliary variable at layer n, a float.
    dt: the time step. cell: domain edge lengths, a scalar or one length per axis.
    params: as in mpfc_free_energy.
    coef: dict of overrides for the five free constants a11t, a32t,
       a33t, a31, a43 of the coefficient family; None means this task's
       prescribed set.
    Return the real array [phi_new, r_new] of shape (2,) + phi.shape,
    whose second plane is constant and equal to r_new.
    Raise ValueError if phi is not a finite real array of rank at least one, if r
    is not a finite scalar, if dt is not a finite positive scalar, or if
    cell, params or coef is invalid."""
    # Implement per the specification above.
    return None
```

### Step 7

mpfc_modified_energy

Goal
----
Evaluate the ****modified discrete energy**** the scheme actually dissipates, and compare it with the original free energy of step 2 on the same layer. Return the real array $[E_{\\rm mod},\\;E,\\;E_{\\rm mod}-E,\\;r-\\sqrt{E_1(\\phi)+C_{SAV}}\\,]$ of shape `(4,)`. The modified energy is the quadratic block of step 2 plus the square of the auxiliary variable minus the shift constant; it is a function of the **pair** $(\\phi,r)$, and it coincides with the original free energy exactly when $r$ is the square root it was initialised as. The last entry is that discrepancy, the quadratization's own consistency error, and it is what makes the two energies separate along a run.

```python
def mpfc_modified_energy(phi: "np.ndarray", r: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """phi: real periodic density field, of ANY dimension d >= 1.
    r: the auxiliary variable carried by the scheme, a float.
    cell: domain edge lengths, a scalar or a length-d sequence.
    params: as in mpfc_free_energy.
    Return the real array [E_mod, E, E_mod - E, r - sqrt(E1 + C_sav)]
    of shape (4,).
    Raise ValueError if phi is not a finite real array of rank at least one, if r
    is not a finite scalar, if cell or params is invalid, or if
    E1(phi) + C_sav is not strictly positive."""
    # Implement per the specification above.
    return None
```

### Step 8

mpfc_energy_identity

Goal
----
Evaluate, term by term, the exact algebraic identity that the source's energy-dissipation theorem is proved from, for one step taken from the layer $(\\phi^n,r^n)$. Writing $\\mathcal L=(\\Delta+1)^2$ for the quadratic operator and $(\\cdot,\\cdot)_h$ for the grid inner product of step 2, the theorem rewrites the change in the quadratic part of the energy as a ****stage-work**** term and a ****quadratic-form**** term built from the same matrix $P$ that step 4 returns the eigenvalues of. Return the real array of shape `(4,)` holding, in order: $(\\phi^{n+1},\\mathcal L\\phi^{n+1})_h$, $(\\phi^{n},\\mathcal L\\phi^{n})_h$, the stage-work term and the quadratic-form term. The four are not independent: the first equals the second plus the third minus the fourth, to round-off, and that is the identity this step certifies. Raise `ValueError` on the same invalid inputs as the single step of step 6.

```python
def mpfc_energy_identity(phi: "np.ndarray", r: float, dt: float,
        cell: "float | tuple", params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    """phi: the density at layer n, any dimension d >= 1.
    r: the auxiliary variable at layer n, a float.
    dt: the time step. cell: domain edge lengths, a scalar or one
       length per axis.
    params: as in mpfc_free_energy. coef: as in mpfc_time_step.
    Return the real array [(phi^{n+1}, L phi^{n+1})_h,
    (phi^n, L phi^n)_h, W, S] of shape (4,), with L the shifted
    bi-Laplacian, W the stage-work term and S the quadratic-form term.
    Raise ValueError if phi is not a finite real array of rank at least
    one, if r is not a finite scalar, if dt is not a finite positive
    scalar, or if cell, params or coef is invalid."""
    # Implement per the specification above.
    return None
```

### Step 9

mpfc_rksav_run

Goal
----
Run the whole benchmark. Build the initial density from the closed form below on the requested grid and box; initialise the auxiliary variable as the exact square root of the shifted nonlinear energy of that initial density; advance $n_steps$ steps of size $dt$ with the step of step 6 and the coefficient family member of step 4. The energy of the final layer is ****assembled from two halves****: its quadratic block is read off the energy identity of step 8, evaluated on the layer the ****last**** step starts from, whose first entry is exactly that block; the nonlinear block, bulk and long-range together, comes from step 2. Return their sum ****divided by the area of the domain****. With $n_steps$ zero no step is taken, there is no identity to read, and the quadratic block is the initial one. Setting `quantity` selects a diagnostic instead: `'energy0'` the initial free-energy density, `'emod'` and `'emod0'` the modified discrete energy density of the final and of the initial layer, `'quad'` the quadratic block of the final layer divided by the area, `'r'` the final auxiliary variable, `'mass'` the relative drift of the spatial mean over the run, `'drift'` the absolute distance between the auxiliary variable carried by the scheme and the square root recomputed from the final density, and `'work'` and `'qform'` the stage-work term and the quadratic-form term that step 8 returns for that last step, each divided by the area and each zero when $n_steps$ is zero.

```python
def mpfc_rksav_run(n_steps: int = 16, dt: float = 0.125, grid: tuple = (48, 64),
        cell: "float | tuple" = (32.0, 48.0), params: "dict | None" = None,
        coef: "dict | None" = None, quantity: str = "energy") -> float:
    """n_steps: number of time steps. dt: the time step.
    grid: (Nx, Ny), two integers; this step is two-dimensional.
    cell: domain edge lengths (Lx, Ly), a scalar or a pair.
    params: dict of model-parameter overrides; None means the fixed set.
    coef: dict of overrides for the five free constants of the
       coefficient family; None means this task's prescribed set.
    quantity: 'energy' (default, the final free-energy density),
       'energy0', 'emod', 'emod0', 'quad', 'r', 'mass', 'drift',
       'work' or 'qform'.
    Return the requested scalar, a float.
    Raise ValueError if n_steps is not a non-negative integer, if dt is
    not a finite positive scalar, if grid is not two integers each at
    least four, if quantity is not one of the names listed above, or if
    cell, params or coef is invalid."""
    # Implement per the specification above.
    return None
```
