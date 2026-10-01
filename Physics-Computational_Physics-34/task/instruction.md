# Physics-Computational_Physics-34

## Background

*The phase-field crystal model.* PFC describes a crystalline solid by a periodic atomic density field rather than a homogeneous order parameter, so it resolves lattice periodicity, elasticity and defects while evolving on diffusive rather than vibrational time scales. Its free energy is a Swift-Hohenberg functional whose linear operator $(1+\Delta)^2$ is minimised at a finite wave number, which is what makes the minimisers periodic; the dynamics is the conserved ($H^{-1}$) gradient flow of that functional, so the equation is sixth order in space and mass conserving.

*Stiffness and stabilised splitting.* The sixth-order linear part has Fourier symbol growing like $|\boldsymbol{k}|^6$, so explicit stepping is unusable and stability comes from treating a chosen linear operator implicitly. Adding and subtracting $A\phi$ moves a constant from the nonlinear potential into the implicit operator: the linear operator becomes $\Delta^2+2\Delta+(\alpha+A)\mathcal{I}$ and the potential becomes $\tfrac14\phi^4-\tfrac{A}{2}\phi^2$. The split changes nothing about the continuous problem - the two shifts cancel identically in the total energy - but it makes the implicit operator positive definite, which is what the solvability and stability proofs need.

*Auxiliary-variable methods and their failure mode.* Invariant energy quadratization, the scalar auxiliary variable method and the Lagrange multiplier approach all linearise the nonlinearity by carrying an extra scalar or field unknown, so each step reduces to constant-coefficient linear solves. SAV-type schemes dissipate a *modified* energy that can drift from the physical one; the Lagrange multiplier variant enforces the original energy law exactly, at the price of a scalar nonlinear equation whose solvability degrades as the physical dissipation rate goes to zero - precisely at the equilibrium states long simulations spend most of their time near. A global quadratic penalty on the multiplier decouples solvability from dissipation and repairs this, at the price of one more parameter to choose.

*Variable-step BDF2.* Second-order backward differentiation on a non-uniform grid replaces the constant BDF2 coefficients by ratio-dependent ones and requires a bound on the adjacent step ratio $\gamma_n$ for the discrete energy argument to close; different equations and different proof techniques give different admissible bounds. Because BDF2 damps high frequencies strongly, it is the natural workhorse for adaptive stepping in stiff gradient flows, where the step must be small during a fast transition and can grow by orders of magnitude during the subsequent relaxation.

## Problem

The phase-field crystal (PFC) equation is a sixth-order gradient flow whose stiffness makes fully implicit stepping expensive and fully explicit stepping useless, and whose dynamics span decades in time - fast defect rearrangement followed by slow coarsening - so that a fixed time step is either wasteful or inaccurate. A modern answer combines a variable-step second-order backward differentiation formula with an auxiliary-variable treatment of the nonlinearity, so that each step costs only constant-coefficient linear solves plus one scalar root-find. This task asks you to **evaluate one such complete scheme**, exactly, on a fully specified deterministic instance. Nothing is random and nothing is fitted; the initial state is a closed-form field, so the whole computation is a single reproducible number.

*Model.* Let $\phi(\boldsymbol{x},t)$ be the atomic density field on the periodic square $\Omega=[0,L)^2$. The PFC equation is the $H^{-1}$ gradient flow

$$\partial_t\phi=\Delta\mu,\qquad \mu=\frac{\delta E}{\delta\phi}=\phi^3+\alpha\phi+2\Delta\phi+\Delta^2\phi,$$

of the classical PFC free energy

$$E[\phi]=\int_\Omega\Big(\tfrac14\phi^4+\tfrac{\alpha}{2}\phi^2-|\nabla\phi|^2+\tfrac12(\Delta\phi)^2\Big)\,d\boldsymbol{x},\qquad \alpha=1-\varepsilon,$$

with $\varepsilon$ the quench depth. All spatial derivatives are evaluated spectrally on the periodic grid.

*Stabilised splitting.* An artificial stabilisation parameter $A\ge0$ is introduced, splitting the chemical potential as $\mu=\mathcal{L}\phi+F'(\phi)$ with the constant-coefficient linear operator and the modified bulk potential

$$\mathcal{L}=\Delta^2+2\Delta+(\alpha+A)\,\mathcal{I},\qquad F'(\phi)=\phi^3-A\phi,\qquad F(\phi)=\tfrac14\phi^4-\tfrac{A}{2}\phi^2 .$$

**Use these exactly as printed** and derive everything else from them. Work out for yourself how $E[\phi]$, $\tfrac12(\mathcal{L}\phi,\phi)$ and $\int_\Omega F(\phi)$ are related, and what that implies about where $A$ can and cannot appear in a quantity that is claimed to be a discrete energy.

*The scheme.* The time grid is non-uniform: steps $\tau_n=t_n-t_{n-1}$ and ratios $\gamma_n=\tau_n/\tau_{n-1}$. The scheme to evaluate is a **published variable-step BDF2 scheme for this equation in which the nonlinear term is carried by a scalar Lagrange multiplier $\eta$ that is anchored near unity by a global artificial penalty proportional to $(\eta^{n+1})^2-(\eta^n)^2$**, so that the scalar equation determining $\eta^{n+1}$ remains solvable at equilibrium, where the physical dissipation rate vanishes and unpenalised Lagrange multiplier methods fail outright. The same source proves unconditional dissipation of a modified discrete energy under a mild bound on the step ratio, and reports a sensitivity study of the penalty parameter. **Find that source and take the scheme from it exactly.** State in your reasoning: the three equations of the scheme, including the second-order variable-step difference operator $D_2\phi^{n+1}$ and the explicit extrapolation $\bar\phi^{\,n+1}$ that the nonlinearity is evaluated at; the scaling function $S(\eta)$ the source adopts and the two normalisation conditions it satisfies; the first-order initial step used to start the recursion; the decomposition of $\phi^{n+1}$ into two fields obtained from constant-coefficient linear solves that do not involve $\eta^{n+1}$, and the resulting scalar nonlinear equation for $\eta^{n+1}$; the modified discrete energy $\tilde E^{n+1}$ whose dissipation is proved, term by term; and the numerical value of the maximal admissible step ratio $\gamma_{\max}$, together with the scalar equation it is the root of. Do not substitute a plausible reconstruction: the graded number depends on the exact form of $S$, on the presence of the penalty inside the scalar equation, on the extrapolated argument of the nonlinearity, on the exact coefficients of $D_2$, and on the exact value of $\gamma_{\max}$.

**One caution about the source.** It prints the scalar equation for $\eta^{n+1}$ twice, and the two printings are not the same equation. Only one of them is consistent with the discrete energy law that the stability proof consumes. **Decide between them on that ground alone**: derive the energy law the proof requires, test each printing against it, and adopt the one that reproduces it exactly. Do not simply take whichever printing you encounter first, and state in your reasoning which you adopted and why. Verify numerically that your implementation satisfies that energy law to round-off at every step, and report the largest residual you observe over the run. Either form of the law is accepted: the full identity, in which the $H^{-1}$ inner product of $D_2\phi^{n+1}$ with the increment, the change in $E$, one half of the stabilised quadratic form of the increment and the penalty increment sum to zero, or the equivalent scalar identity that the multiplier equation itself expresses. Quote a residual **value**; anything at or below $10^{-10}$ counts as round-off here, since the residual is a cancellation between several $O(1)$-$O(10)$ quantities and its floor depends on summation order, while every wrong convention misses by $10^{-5}$ or more.

*Discrete conventions.* Six conventions, which the source leaves generic or omits, are fixed here.

(i) Space is Fourier pseudo-spectral on the periodic mesh $x_a[i]=i\,h_a$, $h_a=L_a/N_a$, $i=0,\dots,N_a-1$, with wave numbers $k_a=2\pi p_a/L_a$ in `numpy.fft.fftfreq` order; $\Delta\to-|\boldsymbol{k}|^2$, $\Delta^2\to|\boldsymbol{k}|^4$, $\Delta^3\to-|\boldsymbol{k}|^6$. Every volume integral is $h_xh_y\sum_{\Omega_h}$.

(ii) In the free energy, the quartic and quadratic terms are evaluated **pointwise** and summed, while the two derivative terms are evaluated as the **spectral quadratic forms** $\int|\nabla\phi|^2=-\int\phi\,\Delta\phi$ and $\int(\Delta\phi)^2=\int\phi\,\Delta^2\phi$, i.e. as $h_xh_y(N_xN_y)^{-1}\sum_{\boldsymbol{k}}|\boldsymbol{k}|^2|\hat\phi|^2$ and $h_xh_y(N_xN_y)^{-1}\sum_{\boldsymbol{k}}|\boldsymbol{k}|^4|\hat\phi|^2$. This is not cosmetic: forming $|\nabla\phi|^2$ pointwise from $\mathrm{i}k_a\hat\phi$ breaks Parseval at the Nyquist mode of an even grid, and with it the exact discrete energy law.

(iii) $\|v\|_{-1}^2=\int_\Omega v\,(-\Delta)^{-1}v$, evaluated in Fourier space with the zero mode removed from $v$ and excluded from the sum. The increment $\phi^{n+1}-\phi^n$ is mean-zero to round-off.

(iv) The scalar equation for $\eta^{n+1}$ is solved by **Newton's iteration started from $\eta^n$**, with the analytic derivative, stopping when the Newton correction is $\le10^{-13}$ in absolute value (at most 100 iterations). **The starting point is a branch-selection convention, not a convenience.** The residual is a degree-eight polynomial in $\eta$ and need not have a single root: the source's solvability theorem carries a smallness assumption on the step, and outside it - weak penalty at $\tau=O(1)$ - the interval $[\,1-\sqrt{\tau_{n+1}},\,1+\sqrt{\tau_{n+1}}\,]$ can hold more than one root, so bracketing on that whole interval is not a well-defined substitute. Take in every case the root Newton reaches from $\eta^n$, i.e. the branch continuously connected to $\eta=1$. This is a question of solvability and is independent of the scheme's unconditional energy dissipation, which rests on the step-ratio bound instead. If the iteration cannot deliver a root - the cap is exhausted, the derivative vanishes, or an iterate goes non-finite - treat it as a solver breakdown and report it; do not accept the last iterate as a root.

(v) The recursion is started by the source's own first-order initial step, from $\phi^0$ with $\eta^0=1$ and $\tau_1=\tau_{\min}$; the same penalised scalar equation determines $\eta^1$.

(vi) **Adaptive step-size rule** (fixed here, since the source specifies only that its steps adapt). Having taken the step to $t_n$ with step $\tau_n$, form the energy rate $\dot E=(E[\phi^n]-E[\phi^{n-1}])/\tau_n$ from the **original** free energy $E$ above, and set

$$\tau_{n+1}=\max\Big\{\tau_{\min},\ \min\big\{\tau_{\max},\ \tau_{\mathrm{ad}},\ \gamma_{\max}\tau_n\big\}\Big\},\qquad \tau_{\mathrm{ad}}=\frac{\tau_{\max}}{\sqrt{1+\beta\,\dot E^{\,2}}},$$

with $\gamma_{\max}$ the value taken from the source. The last entry of the inner minimum is the step-ratio clamp: it is what keeps the scheme inside the regime in which the modified energy is proved to dissipate. **The convention of this task is the PRINTED value: take $\gamma_{\max}$ as the source prints it, the rounded decimal it states, rather than re-solving its defining equation to higher precision.** That is the same convention stated in step 7 and used in every other field of this task. The two choices differ by far less than the grading band - re-solving the root exactly moves the reported total by about $2\times10^{-7}$, which is invisible at the graded precision - so a response that re-solves the root is not penalised; the printed value is nevertheless the reference.

*Benchmark instance.* Square box $L_x=L_y=32\pi$, grid $N_x=N_y=64$, $\alpha=0.75$ and $A=0.37$ (so $\varepsilon=0.25$), exactly the physical parameters the source uses for its dynamical runs. Writing $s=2\pi x/L_x$ and $t=2\pi y/L_y$, the **seed field** is the deterministic stand-in for the source's randomly perturbed constant state,

$$\phi^{0}=-0.27+0.06\cos(16s+0.3)\cos(9t)+0.05\sin(11s)\cos(13t+0.7)+0.04\cos(7s-14t+1.1)$$
$$\qquad\quad+\;0.03\sin(19s+5t)\sin(6t)+0.02\cos(23s)\sin(21t+0.4),$$

which has mean exactly $-0.27$ and puts most of its power near the PFC band $|\boldsymbol{k}|\approx1$. The controller uses $\tau_{\min}=10^{-4}$, $\tau_{\max}=1$, $\beta=4$, and the run is exactly $N=60$ steps (the initial step plus 59 BDF2 steps). The **penalty sweep** is the source's own, $\theta\in\{0.5,\,5.0,\,50.0,\,500.0\}$, each value run independently from the same $\phi^0$.

*What to report.* For each $\theta$, evaluate the source's **modified discrete energy** $\tilde E^{N}$ at the final state. Its leading coefficient involves the step ratio one step beyond the end of the run: take $\gamma_{N+1}=\tau_{N+1}/\tau_N$ with $\tau_{N+1}$ the step the controller of (vi) proposes after the final step, computed but not used to advance. **Report the sum of $\tilde E^{N}$ over the four penalty parameters.**

The reasoning section has one required content, listed here; report all of it, and little else, so that the pipeline can be seen to have run: the relation between $E$, $\tfrac12(\mathcal{L}\phi,\phi)$ and $\int F$, and what it means for $\tilde E$; the real Fourier symbol of the implicit operator inverted at each step; $S(\eta)$ and $\gamma_{\max}$; the free energy of the seed field and the conserved mass; for each of the four $\theta$, the final free energy $E[\phi^{N}]$, the final modified energy $\tilde E^{N}$, the final multiplier $\eta^{N}$, the largest deviation $\max_n|\eta^n-1|$ over the run, the final time $\sum_n\tau_n$ and the ratio $\gamma_{N+1}$; whether $E$ and whether $\tilde E$ decrease at every step of every run, and the largest residual of the discrete energy identity of the caution paragraph above; how many steps of each run are clamped at $\gamma_{\max}$; the value the same pipeline returns with the multiplier frozen at $\eta\equiv1$; what happens to the run when the penalty term is removed from the scalar equation; and, in one line each, the conventions you used for the energy quadratic forms, the $H^{-1}$ norm, the Newton start, and the initial step.

Output Format Requirements: Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure. Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: `0.4847`, `12.6`, `1.05`). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep `<reasoning>` short (a few hundred words): the diagnostics listed above are its required content and must all appear, and beyond them show only the few scalars that determine the final number. Do not paste the seed field on the grid, the step-size sequence, per-iteration Newton paths, or per-step tables.

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

step_01_spectral_wavenumbers

Goal
----
Return the array of squared Fourier wave numbers $|\\boldsymbol{k}|^2$ on a periodic **rectangular** two-dimensional box, in `numpy` FFT mode order. Two things are general here and both are graded: $N$ is either an `int` or a length-2 sequence $(N_x, N_y)$, and $L$ is either a scalar or a length-2 sequence $(L_x, L_y)$, so direction $a$ carries its own mode count $N_a$ *and* its own box length $L_a$, and the two are not interchangeable. Every spectral operator in the rest of the pipeline - the Laplacian, the biharmonic and tri-Laplacian operators inside the implicit symbol, the inverse Laplacian of the $H^{-1}$ norm, and the two quadratic forms of the free energy - is built from this one array.

```python
def spectral_wavenumbers(N: "int | Sequence[int]",
                         L: "float | Sequence[float]") -> "np.ndarray":
    """N: int, or length-2 sequence (Nx, Ny), the per-direction mode counts.
    L: float, or length-2 sequence (Lx, Ly), the per-direction box lengths.
    Return the real array of squared Fourier wave numbers |k|^2 in numpy FFT
    mode order, shape (Nx, Ny)."""
    # Implement per the formulas above.
    return None
```

### Step 2

step_02_pfc_free_energy

Goal
----
Return the discrete phase-field-crystal free energy of a state $\\phi$ on the periodic rectangular box. Two of the four terms are evaluated pointwise and two as spectral quadratic forms, and **the split is graded**: the quartic and quadratic terms are summed on the grid, while $\\int|\\nabla\\phi|^2$ and $\\int(\\Delta\\phi)^2$ are evaluated by Parseval as $-\\int\\phi\\,\\Delta\\phi$ and $\\int\\phi\\,\\Delta^2\\phi$. Forming $|\\nabla\\phi|^2$ pointwise from $\\mathrm{i}k_a\\hat\\phi$ instead is a different discretisation: on an even grid the Nyquist column carries a first derivative that is not the adjoint of the second, so the two disagree by about $5\\times10^{-9}$ on an evolved $64^2$ state - enough to destroy the exact discrete energy law that the whole scheme rests on. This is the *original* energy of the model, with no stabilisation parameter in it.

```python
def pfc_free_energy(phi: "np.ndarray", L: "float | Sequence[float]",
                    alpha: float) -> float:
    """phi: real 2-D array of shape (Nx, Ny), the atomic density field.
    L: float, or length-2 sequence (Lx, Ly), the per-direction box lengths.
    alpha: float, the temperature parameter 1 - eps.
    Return the discrete PFC free energy E[phi] as a float."""
    # Implement per the formulas above.
    return None
```

### Step 3

step_03_h_minus1_norm

Goal
----
Return the squared $H^{-1}$ norm $\\|v\\|_{-1}^2=\\int_\\Omega v\\,(-\\Delta)^{-1}v$ of a mean-zero field on the periodic rectangular box, evaluated spectrally. The inverse Laplacian is defined only up to a constant and only on mean-zero data, so the zero Fourier mode must be **removed from the input and excluded from the sum**; the function must therefore return the same value for $v$ and for $v+c$. This norm carries the leading term of the modified discrete energy whose dissipation the scheme guarantees, and it is the norm in which the variable-step BDF2 difference operator is tested against the increment.

```python
def h_minus1_norm_sq(v: "np.ndarray",
                     L: "float | Sequence[float]") -> float:
    """v: real 2-D array of shape (Nx, Ny).
    L: float, or length-2 sequence (Lx, Ly), the per-direction box lengths.
    Return the squared H^{-1} norm of v as a float, with the zero Fourier mode
    removed from v and excluded from the sum."""
    # Implement per the formulas above.
    return None
```

### Step 4

step_04_decoupled_fields

Goal
----
Return the three fields that make one step of the scheme independent of the multiplier: the explicit extrapolation $\\bar\\phi^{\\,n+1}$ at which the nonlinearity is evaluated, and the two fields $p^{n+1}$ and $q^{n+1}$ obtained from the same constant-coefficient elliptic solve. The new state is then $\\phi^{n+1}=p^{n+1}+S(\\eta^{n+1})\\tau\\,q^{n+1}$, so **neither $p$ nor $q$ may depend on $\\eta$** - that is exactly what makes the scheme fully decoupled and reduces the step to two elementwise divisions in Fourier space plus one scalar root-find. Setting $\\gamma=0$ must reproduce the first-order backward-Euler starting step: the leading coefficient becomes $1$, the extrapolation collapses to $\\phi^n$, and $\\phi^{n-1}$ drops out. That degenerate case is graded.

```python
def bdf2_decoupled_fields(phi_n: "np.ndarray", phi_nm1: "np.ndarray", gamma: float,
                          tau: float, L: "float | Sequence[float]", alpha: float,
                          A: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """phi_n, phi_nm1: real 2-D arrays of the same shape, the states at t_n and
    t_{n-1}.  gamma: float >= 0, the step ratio tau_{n+1}/tau_n (use 0.0 for the
    backward-Euler starting step).  tau: float > 0, the step tau_{n+1}.
    L: float, or length-2 sequence (Lx, Ly).  alpha, A: floats.
    Return the tuple (phi_bar, p, q) of three real arrays with the shape of
    phi_n, neither p nor q depending on the Lagrange multiplier."""
    # Implement per the formulas above.
    return None
```

### Step 5

step_05_solve_multiplier

Goal
----
Return the Lagrange multiplier $\\eta^{n+1}$ of one step, by solving the penalised scalar nonlinear equation. This is the whole nonlinear content of the step: everything else is linear. The equation is the **discrete energy law itself** - the change in bulk energy plus the change in the penalty term equals the multiplier-weighted work of the extrapolated nonlinearity over the increment - so an implementation is correct if and only if that identity holds to round-off once $\\eta^{n+1}$ is substituted back. Three things are graded and each of them changes the answer: the scaling function is $S(\\eta)=\\eta(2-\\eta)$, not $\\eta$; the penalty term $\\theta(\\eta^2-(\\eta^{n})^2)$ sits **inside** this equation, not only in the energy functional, and removing it makes the iteration break down; and there is **no extra factor of $\\tau$** multiplying $S(\\eta)$ outside the bracket, only the one inside $p+S(\\eta)\\tau q$. Newton is started at $\\eta^{n}$: that is the prescribed **branch-selection convention**, because $g$ may have more than one root and only the branch continuously connected to $\\eta=1$ is the discrete solution. If the iteration cannot deliver such a root - the cap is exhausted, the derivative vanishes, or an iterate goes non-finite - **report the failure**; do not return the last iterate.

```python
def solve_multiplier(p: "np.ndarray", q: "np.ndarray", phi_n: "np.ndarray",
                     phi_bar: "np.ndarray", tau: float, theta: float,
                     eta_n: float, A: float, L: "float | Sequence[float]",
                     tol: float=1e-13, max_iter: int=100) -> float:
    """p, q, phi_n, phi_bar: real 2-D arrays of the same shape, from step 4.
    tau: float > 0, the step tau_{n+1}.  theta: float > 0, the penalty.
    eta_n: float, the previous multiplier (also the Newton starting guess, which
        selects the root branch and is part of the scheme's specification).
    A: float, the stabilisation parameter.  L: float or length-2 sequence.
    tol: float, stopping tolerance on the Newton correction.
    max_iter: int, iteration cap.
    Return the new Lagrange multiplier eta^{n+1} as a float.
    Raise RuntimeError if the iteration cannot produce a root: max_iter is
    exhausted before the correction falls to tol, the derivative vanishes, or
    the residual, derivative or iterate becomes non-finite."""
    # Implement per the formulas above.
    return None
```

### Step 6

step_06_bdf2_step

Goal
----
Advance the scheme by one step: given $\\phi^{n-1}$, $\\phi^{n}$, $\\eta^{n}$, the step ratio $\\gamma_{n+1}$ and the step $\\tau_{n+1}$, return the pair $(\\phi^{n+1},\\eta^{n+1})$. This is steps 4 and 5 composed with the reconstruction $\\phi^{n+1}=p^{n+1}+S(\\eta^{n+1})\\,\\tau_{n+1}\\,q^{n+1}$, using $S(\\eta)=\\eta(2-\\eta)$. Passing $\\gamma=0$ (with $\\phi^{n-1}$ ignored) must give the first-order backward-Euler starting step, so the whole run is this one function called once with $\\gamma_1=0$ and then repeatedly with the controller's ratio. The step must conserve the spatial mean of $\\phi$ exactly, to round-off, for every $\\gamma$, $\\tau$ and $\\theta$. The multiplier solve of step 5 is the only stage that can fail, and its failure is **propagated, not absorbed**: if Newton cannot deliver a root the step reports the breakdown instead of returning a state built from an unconverged iterate.

```python
def pfc_bdf2_step(phi_n: "np.ndarray", phi_nm1: "np.ndarray", eta_n: float,
                  gamma: float, tau: float, L: "float | Sequence[float]",
                  alpha: float, A: float,
                  theta: float) -> "tuple[np.ndarray, float]":
    """phi_n, phi_nm1: real 2-D arrays of the same shape.
    eta_n: float, the previous Lagrange multiplier.
    gamma: float >= 0, the step ratio (0.0 for the backward-Euler start).
    tau: float > 0, the step tau_{n+1}.  L: float or length-2 sequence.
    alpha, A, theta: floats.
    Return the tuple (phi_next, eta_next): the new state as a real array with
    the shape of phi_n, and the new multiplier as a float.
    Raise RuntimeError if the step-5 multiplier solve breaks down - the Newton
    cap is exhausted, the derivative vanishes, or a residual, derivative or
    iterate becomes non-finite. The failure propagates out of this function; do
    not substitute the last iterate, eta_n, or any fallback value."""
    # Implement per the formulas above.
    return None
```

### Step 7

step_07_adaptive_time_step

Goal
----
Return the next time step chosen by the energy-controlled adaptive rule. The step shrinks where the free energy is falling fast and relaxes towards $\\tau_{\\max}$ once the flow is quiet, and it is then held back by two hard limits: the floor $\\tau_{\\min}$ and the **step-ratio clamp** $\\gamma_{\\max}\\tau_n$, with $\\gamma_{\\max}=4.86454$ the maximal ratio under which the modified discrete energy of this scheme is proved to dissipate. The clamp is the only place in the pipeline where that constant enters, and it is what keeps an adaptive run inside the proved-stable regime; the value must be taken from the source rather than assumed, since the corresponding bound for Cahn-Hilliard is the much tighter $1.534$. Use the constant **as the source prints it**, $4.86454$, rather than re-solving its defining equation $\\gamma^{3/2}=1+2\\gamma$ to higher precision; that is the convention every field of this task uses.

```python
def adaptive_time_step(tau_n: float, E_n: float, E_nm1: float,
                       tau_min: float, tau_max: float,
                       beta: float) -> float:
    """tau_n: float > 0, the step just taken.
    E_n, E_nm1: floats, the free energies after and before that step.
    tau_min, tau_max: floats with 0 < tau_min <= tau_max.
    beta: float >= 0, the controller gain.
    Return the next time step tau_{n+1} as a float."""
    # Implement per the formulas above.
    return None
```

### Step 8

step_08_penalized_lm_answer

Goal
----
**Final orchestrator.** Run the complete penalised Lagrange multiplier variable-step BDF2 scheme once for each penalty parameter and return the sum, over those runs, of the modified discrete energy at the final state. Each run starts from the same analytic seed field with $\\eta^{0}=1$, takes its first step by backward Euler with $\\tau_{1}=\\tau_{\\min}$ (i.e. step 6 at $\\gamma=0$), and then takes $n_steps - 1$ BDF2 steps whose ratios come from the controller of step 7 driven by the *original* free energy of step 2. After the last step the controller is called **once more**, purely to supply the ratio $\\gamma_{N+1}$ that the modified energy's leading coefficient needs; that proposed step is not taken. Solvability of the scalar multiplier equation is *conditional* - unlike the energy dissipation, which is not - so a breakdown of the step-5 solve at any step of any run propagates out of this function as a reported failure; it is never absorbed, retried with a fallback, or replaced by a substituted multiplier. On the default arguments this returns the graded number.

```python
def penalized_lm_pfc_answer(
        thetas: "Sequence[float]"=(0.5, 5.0, 50.0, 500.0),
        N: "int | Sequence[int]"=64,
        L: "float | Sequence[float]"=100.53096491487338,
        alpha: float=0.75, A: float=0.37, n_steps: int=60,
        tau_min: float=1e-4, tau_max: float=1.0, beta: float=4.0) -> float:
    """thetas: sequence of penalty parameters, one independent run each.
    N: int or length-2 sequence, the grid.  L: float or length-2 sequence.
    alpha, A: floats, the material and stabilisation parameters.
    n_steps: int >= 2, the total number of steps per run (backward-Euler start
    plus n_steps - 1 BDF2 steps).
    tau_min, tau_max, beta: floats, the adaptive controller parameters.
    Return the sum over thetas of the final modified discrete energy, a float.
    Raise RuntimeError if the multiplier solve of step 5 breaks down at any step
    of any run. Solvability of that scalar equation is conditional - a penalty
    too weak for the step size loses it - and the failure propagates out of this
    function rather than being absorbed or replaced by a fallback value."""
    # Implement per the formulas above.
    return None
```
