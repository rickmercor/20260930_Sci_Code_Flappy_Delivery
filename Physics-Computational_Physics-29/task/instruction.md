# Physics-Computational_Physics-29

## Background

*Why a gradient quartic.* A phase-field-crystal free energy is a Swift-Hohenberg functional: the operator $(1+\Delta)^2$ penalises every wavenumber except $|\boldsymbol k|=1$, so the minimiser is periodic with wavelength $2\pi$, and the nonlinear term decides which periodic lattice wins. With the classical quartic $\tfrac14\phi^4$ the three-mode resonance $\boldsymbol k_1+\boldsymbol k_2+\boldsymbol k_3=0$ among unit wavevectors is available, and the hexagonal lattice - whose reciprocal vectors realise it - is selected. Replacing $\tfrac14\phi^4$ by $\tfrac14|\nabla\phi|^4$ removes that cubic resonance from the amplitude equations and leaves the four-mode combination $\boldsymbol k_1+\boldsymbol k_2-\boldsymbol k_3-\boldsymbol k_4=0$ dominant, which two orthogonal pairs of unit wavevectors realise: the square lattice. The price is analytical. The variational derivative of $\tfrac14\int|\nabla\phi|^4$ is $-\nabla\cdot(|\nabla\phi|^2\nabla\phi)$, a 4-Laplacian, so the chemical potential is quasilinear rather than semilinear and the flow $\partial_t\phi=\Delta\mu$ is a sixth-order equation with a degenerate-at-$\nabla\phi=0$ nonlinearity.

*Convex splitting and why an order above two needs help.* Write the energy as convex plus concave: $\tfrac14|\nabla\phi|^4+\tfrac12\phi(1+\Delta)^2\phi$ is convex in $\phi$, and $-\tfrac{\varepsilon}{2}\phi^2$ is concave. Treating the convex part implicitly and the concave part explicitly makes the step the minimisation of a strictly convex functional, hence uniquely solvable for every $\tau$ with no step-size restriction, and at first order it also dissipates the energy outright, because a convex functional lies above its tangent plane and a concave one below. Above first order the argument breaks: the concave part is no longer evaluated at a single old level but extrapolated, and the extrapolation error can inject energy. The repair is an artificial diffusivity - a stabilization term whose sole job is to dominate that injection. Choosing it as $S\tau^q\Delta_h\mathcal{B}_q\phi^n$, i.e. built from the same multi-level difference operator as the time derivative, keeps it $O(\tau^q)$ so that it does not degrade the order, while giving it a quadratic decomposition of exactly the same shape as the one the time derivative already has.

*Discrete gradient structures.* Both the BDF$q$ operator and the extrapolation are multi-step, so neither is a plain difference of two levels and neither telescopes on its own. The standard device is a quadratic decomposition: for a $q$-step kernel one seeks a symmetric positive-semidefinite $q\times q$ matrix $\mathbf G$ and a positive constant $\kappa$ with $v^n\sum_j\beta_jv^{n-j}=(\vec v^{\,n})^T\mathbf G\vec v^{\,n}-(\vec v^{\,n-1})^T\mathbf G\vec v^{\,n-1}+(\vec v^{\,n})^T\mathbf R\vec v^{\,n}+\tfrac{\kappa}{2}|v^n|^2$, with $\mathbf R$ positive semidefinite. The first two terms telescope, the third is a sign-definite remainder and the fourth is the dissipation the argument spends. Such decompositions exist for BDF$q$ up to $q=5$ and only up to $q=5$, which is why the family stops there and not because of implementation convenience: BDF6 is not even zero-stable. The same construction applied to the extrapolation kernel yields $\mathbf J$, $\mathbf X$ and a constant $\eta$, this time with the wrong sign - $-\eta|v^n|^2$ - so it consumes rather than supplies dissipation, and it is precisely this deficit that the stabilization has to cover. Adding the three telescoping pieces to the original energy defines a modified energy that decreases monotonically - subject to the same adjointness hypothesis the argument below rests on, so the statement covers data band-limited strictly below Nyquist and has to be checked rather than quoted on anything else; because the added pieces are quadratic in the first differences, they vanish whenever the solution is stationary, so the modified and original energies agree in equilibrium and at any start-up whose levels coincide.

*Where the step-size independence comes from.* The stabilization contributes $\tfrac{1}{2\tau} S\tau^{q}\kappa_q\|\nabla_h\delta_\tau\phi^n\|^2$ to the per-step budget, and the deficit to be covered is $O(1)$, so a naive comparison would force $S$ to grow like $\tau^{-(q-1)}$ - a step-size-dependent condition, which is what "unconditional" is supposed to rule out. The way out is to spend the constant $\tfrac12$ that the $(1+\Delta_h)$ term already provides: the identity $\|\nabla_hv\|^2=\|v\|^2-\langle(1+\Delta_h)v,v\rangle$, which holds wherever $\nabla_h$ and $\Delta_h$ are exact adjoints, with Young's inequality gives $\|\nabla_hv\|^2\le\tfrac54\|v\|^2+\|(1+\Delta_h)v\|^2$, and then a weighted arithmetic-geometric-mean inequality applied to $\tfrac{q-2}{q-1}\cdot\tfrac{q-1}{q-2}+\tfrac{1}{q-1}\cdot(q-1)S\kappa_q\tau^{q-1}$ converts the coefficient from $O(\tau^{q-1})$ to $O(\tau)$ with the constant $C_q=[(\tfrac{q-1}{q-2})^{q-2}S\kappa_q(q-1)]^{1/(q-1)}$. That single factor of $\tau$ is then absorbed against the $\kappa_q/(2\tau)$ in front of the $H^{-1}$ term by the interpolation inequality $\|v\|^2\le\|\nabla_hv\|\,\|v\|_{-1}$, which is available because mass conservation puts every first difference in the mean-zero subspace. The surviving requirement, $\sqrt{\kappa_qC_q}\ge\varepsilon\eta_q+\varepsilon/2+5/8$, contains no $\tau$ and no mesh size.

*Reading the resulting bound.* Because that chain spends several inequalities generously, the resulting threshold on $S$ is a sufficient condition and a weak one; it is not a prediction of the $S$ a practitioner should use. In practice $S$ is chosen for two competing reasons the analysis does not see: a larger $S$ makes the implicit operator more diagonally dominant, so the fixed-point iteration converges in fewer sweeps, while a larger $S$ also enlarges the artificial diffusivity and hence the error constant. Reporting the threshold without that reading is reporting half the result.

## Problem

A phase-field-crystal model resolves the atomic-scale periodicity of a crystal with a coarse-grained density field, so that elasticity, plasticity and defects emerge from a free energy instead of being inserted by hand. The classical functional, whose quartic term is $\tfrac14\phi^4$, selects a hexagonal lattice in two dimensions. Replacing that term by $\tfrac14|\nabla\phi|^4$ selects a square lattice instead, and the resulting sixth-order gradient flow - a 4-Laplacian sitting inside an $H^{-1}$ flow - is much harder to integrate stably than the classical one. This task asks you to run a high-order, unconditionally energy-stable time-marching scheme for that model, exactly, on a fully specified deterministic instance. Nothing is fitted and nothing is random.

*The model.* On a periodic rectangle $\Omega=(0,L_x)\times(0,L_y)$ the phase variable $\phi$ obeys the conserved gradient flow $\partial_t\phi=\Delta\mu$, $\mu=-\nabla\cdot(|\nabla\phi|^2\nabla\phi)-\varepsilon\phi+(1+\Delta)^2\phi$, which is the $H^{-1}$ gradient flow of $E_s[\phi]=\int_\Omega\big(\tfrac14|\nabla\phi|^4+\tfrac12\phi(1+\Delta)^2\phi-\tfrac{\varepsilon}{2}\phi^2\big)\,d\boldsymbol x$. Here $0\le\varepsilon\le1$ is a dimensionless parameter measuring the deviation from the melting temperature. The first two terms of $E_s$ are convex in $\phi$ and the third is concave; that split is the one the scheme below exploits.

*Space.* Discretise with the Fourier pseudo-spectral method on the uniform grid $x_i=iL_x/M_x$, $y_j=jL_y/M_y$ with $M_x,M_y$ even, spacings $h_x=L_x/M_x$ and $h_y=L_y/M_y$, and the signed index set $-M_x/2\le k\le M_x/2-1$, $-M_y/2\le m\le M_y/2-1$ - exactly the set `numpy.fft.fftfreq` produces, in its storage order. Write $\lambda_{k,m}=4\pi^2\big(k^2/L_x^2+m^2/L_y^2\big)$, so that $-\Delta_h$ has symbol $\lambda_{k,m}$ and $\Delta_h$ has symbol $-\lambda_{k,m}$. The first derivatives $D_x$ and $D_y$ have symbols $2\pi ik/L_x$ and $2\pi im/L_y$, and $\nabla_h v=(D_xv,D_yv)^T$, $\Delta_h v=D_{xx}v+D_{yy}v$ where $D_{xx},D_{yy}$ carry the second-derivative multipliers $-4\pi^2k^2/L_x^2$ and $-4\pi^2m^2/L_y^2$ directly. $\Delta_h$ is therefore **not** the composition $D_x\circ D_x+D_y\circ D_y$: the two differ at the Nyquist row, because of the convention fixed next. Two conventions are fixed and are graded. First, in the **odd**-order multipliers $D_x$ and $D_y$ the entry at the Nyquist index ($k=-M_x/2$, respectively $m=-M_y/2$) is set to zero before the inverse transform; in the even-order multipliers, including $\lambda_{k,m}$, it is kept. Second, every field is real, so the real part is taken after each inverse transform. These two are the same convention, and the reason is a statement about the Nyquist row as a whole rather than about its individual entries. For a real field the conjugate symmetry $\tilde v_{-k,-m}=\overline{\tilde v_{k,m}}$ maps the row $k=-M_x/2$ onto itself with $m\mapsto-m$, so the row is Hermitian in $m$; its entries are forced real only at the self-paired $m=0$ and $m=-M_y/2$, and are complex in general. An odd multiplier is purely imaginary and constant along that row, so it turns the row's combined contribution to the inverse transform into a purely imaginary one, which is exactly what taking the real part discards - and zeroing the row explicitly therefore gives the identical array. The same holds for the column $m=-M_y/2$ in $D_y$. For the pure Nyquist oscillation $v_{ij}=(-1)^i$, which is constant in $y$, the convention returns $D_xv=0$ outright. The discrete inner product and norm are $\langle u,w\rangle=h_xh_y\sum_{i,j}u_{ij}w_{ij}$ and $\|u\|=\sqrt{\langle u,u\rangle}$, and $\|v\|_4^4=h_xh_y\sum_{i,j}|v_{ij}|^4$; for the vector field $\nabla_h\phi$ the modulus is the pointwise Euclidean length, so $\|\nabla_h\phi\|_4^4=h_xh_y\sum_{i,j}\big((D_x\phi)_{ij}^2+(D_y\phi)_{ij}^2\big)^2$.

*Time.* Fix an order $q\in\{3,4,5\}$ and a step $\tau$. For a sequence $\{v^n\}$ let $\delta_\tau v^n=v^n-v^{n-1}$. The BDF$q$ operator is written as a discrete convolution over first differences, $\mathcal{B}_qv^n=\tfrac1\tau\sum_{j\ge0}\beta_j^{(q)}\delta_\tau v^{n-j}$, and $\beta_j^{(q)}=0$ for $j\ge q$. The kernel $\beta^{(q)}$ is fixed by the single requirement that this convolution reproduce the BDF$q$ approximation of $\partial_tv(t_n)$ exactly; its entries are **not** listed anywhere in this task, and recovering them for $q=3,4,5$ is the first thing to be derived. **They are convolution weights on first differences, not the direct BDF$q$ coefficients $c_j^{(q)}$ acting on $v^{n-j}$;** the two are different vectors, and both are needed. The $q$-order extrapolation of $v(t_n)$ is $\hat v_q^{\,n}=v^n-\delta_\tau^qv^n=v^n-\sum_{j\ge0}\alpha_j^{(q)}\delta_\tau v^{n-j}$, where $\delta_\tau^m$ is the $m$-fold first difference; expanding that $m$-fold difference over the levels, and so obtaining $\alpha^{(q)}$, is likewise left to the task and is not written out here. The stability theory also uses the discrete orthogonal convolution (DOC) kernel $\gamma^{(q)}$, the convolution inverse of $\beta^{(q)}$, fixed by $\sum_{i=0}^{m}\gamma_i^{(q)}\beta_{m-i}^{(q)}=\delta_{m0}$ for all $m\ge0$; its entries are not listed here either, and it satisfies the published decay bound $|\gamma_j^{(q)}|\le\tfrac{r_q}{4}(q/7)^j$ with $r_3=10/3$, $r_4=6$, $r_5=96/5$.

*The scheme.* Apply BDF$q$ to $\partial_t\phi$, treat the convex part implicitly, extrapolate the concave part, and add a multi-time-level stabilization term: find $\phi^n$ with $\mathcal{B}_q\phi^n=\Delta_h\mu^n$ and $\mu^n=-\nabla_h\cdot\big(|\nabla_h\phi^n|^2\nabla_h\phi^n\big)-\varepsilon\hat\phi_q^{\,n}+(1+\Delta_h)^2\phi^n-S\tau^q\Delta_h\mathcal{B}_q\phi^n$, where $S\ge0$ is the stabilization parameter. Write $N(\phi)=-\nabla_h\cdot(|\nabla_h\phi|^2\nabla_h\phi)$ for the quartic term, and eliminate $\mu^n$ between the two equations. How that elimination goes through - where the stabilization ends up once it is done, what separates out of $\mathcal{B}_q\phi^n$, which terms are linear in $\phi^n$ and what operator is left to invert - is not given here and is part of what has to be got right. Whatever rearrangement you arrive at, resolve the nonlinearity that is left by the fixed-point iteration that rearrangement defines, started from $\phi^{(0)}=\hat\phi_q^{\,n}$ and stopped when $\max_{ij}|\phi^{(s+1)}-\phi^{(s)}|<10^{-13}$, with a hard cap of $200$ iterations. Start-up is prescribed, not computed: **take $\phi^0=\phi^1=\dots=\phi^{q-1}=\phi_0$**, the given initial field repeated, which is mass-conservative as the scheme requires, and step from $n=q$ onward.

*The two energies.* The original discrete energy is $E^n=\tfrac14\|\nabla_h\phi^n\|_4^4-\tfrac{\varepsilon}{2}\|\phi^n\|^2+\tfrac12\|(1+\Delta_h)\phi^n\|^2$. The quantity the scheme actually dissipates is a modified energy built from the quadratic decompositions of the BDF$q$ kernel and of the extrapolation kernel. Writing $\vec v^{\,n}=(v^{n-q+1},\dots,v^n)^T$ and $\langle\vec a,\vec b\rangle=\sum_{i=0}^{q-1}\langle a^{n-i},b^{n-i}\rangle$, and given symmetric positive-semidefinite $q\times q$ matrices $\mathbf G_q$ and $\mathbf J_q$ - which act on the $q$-vector of grid functions before the inner product is taken, so that $\langle\mathbf G\vec a,\vec a\rangle=\sum_{i,j}\mathbf G_{ij}\langle a^{(j)},a^{(i)}\rangle$ with $a^{(i)}$ the $i$-th slot of $\vec a$ - it is $\hat E^n=E^n+\tfrac1\tau\Big[\big\langle\mathbf G_q\delta_\tau\vec\phi^{\,n},\delta_\tau\vec\phi^{\,n}\big\rangle_{-1}+S\tau^{q}\big\langle\mathbf G_q\delta_\tau\nabla_h\vec\phi^{\,n},\delta_\tau\nabla_h\vec\phi^{\,n}\big\rangle\Big]+\varepsilon\big\langle\mathbf J_q\delta_\tau\vec\phi^{\,n},\delta_\tau\vec\phi^{\,n}\big\rangle$ - the two $\mathbf G$ terms share the single prefactor $\tfrac1\tau$, and how that leaves each of them scaling in $\tau$ is part of what has to be got right, in which $\langle u,w\rangle_{-1}=\langle(-\Delta_h)^{-1}u,w\rangle$ is taken on mean-zero fields with the zero mode of $(-\Delta_h)^{-1}$ set to zero. **The numerical entries of $\mathbf G_q$ and $\mathbf J_q$, and the values of the two constants $\kappa_q$ and $\eta_q$ that accompany them, are deliberately not printed here.** They are the published decomposition data of the BDF$q$ convolution kernel $\beta^{(q)}$ and of the extrapolation kernel $\alpha^{(q)}$ respectively, in the sense made precise in the background section, and only their $q=3$ instances are used below; retrieve them from the primary literature on discrete gradient structures for BDF$q$ and on stabilized convex-splitting BDF$k$ schemes. Read them in the slot ordering fixed just above, in which the last slot of $\vec v^{\,n}$ is the newest level $v^n$; in that ordering the first row and the first column of both matrices vanish, both are symmetric positive semidefinite, and $\kappa_3,\eta_3>0$, which is enough to detect a transposed, reversed or differently normalised copy. There is a theorem giving $\hat E^n\le\hat E^{n-1}$ for $n\ge q$ under the stabilization condition below. That proof runs through $\|\nabla_hv\|^2=\|v\|^2-\langle(1+\Delta_h)v,v\rangle$ and through the interpolation inequality $\|v\|^2\le\|\nabla_hv\|\,\|v\|_{-1}$, and both identities need $\nabla_h$ and $\Delta_h$ to be exact adjoints. Under the Nyquist convention fixed above they are not adjoint on the Nyquist rows, so the guarantee is a statement about fields band-limited strictly below Nyquist and not about every grid function; on a field with Nyquist content $\hat E$ can and does increase even when the condition holds. Whether it decreases on the data below is therefore something to measure, not to quote.

*The stabilization condition.* The dissipation proof needs $C_q\ge(\varepsilon\eta_q+\varepsilon/2+5/8)^2/\kappa_q$, where $C_q:=\Big[\big(\tfrac{q-1}{q-2}\big)^{q-2}S\,\kappa_q\,(q-1)\Big]^{1/(q-1)}$ is independent of $\tau$. Write $C_{\mathrm{req}}$ for the right-hand side of that inequality, and $S_{\min}$ for the smallest $S$ at which the inequality holds.

*The instance.* Five configurations are to be run, every one of them at $q=3$. That is the order for which the complete set of decomposition data - $\mathbf G_3$, $\mathbf J_3$, $\kappa_3$ and $\eta_3$ - is available in the literature, and therefore the only order at which $\hat E$ and the certificate can both be formed from published constants; the kernels for $q=4$ and $q=5$ are given above because the scheme is stated for all three, and they are not used on this instance. Two initial fields are used. The **trigonometric** seed is $\phi_0=0.07+0.60\big(\cos x\cos y+0.40\sin(2x)\cos y+0.30\cos(x-2y)\big)$, evaluated at the grid points in physical coordinates; the **nucleus** seed is $\phi_0=2.5\big(1-\tanh\big(\tfrac12(r-2)\big)\big)$ with $r=\sqrt{(x-L_x/2)^2+(y-L_y/2)^2}$. The stabilization parameter is **not** given directly. Each configuration carries a dimensionless **safety factor** $s$, and the parameter actually used is $S=s\,S_{\min}(\varepsilon)$, with $S_{\min}$ the value the certificate above returns at $q=3$, at the retrieved $\kappa_3$ and $\eta_3$, and at that configuration's $\varepsilon$. Every run therefore sits at a known multiple of the smallest stabilization the theory admits, and the certificate has to be evaluated before any stepping can begin. The configurations are

| tag | seed | $(L_x,L_y)$ | $(M_x,M_y)$ | $\varepsilon$ | $s$ | $\tau$ | steps |
|---|---|---|---|---|---|---|---|
| P1 | trigonometric | $(8\pi,8\pi)$ | $(32,32)$ | $0.25$ | $200$ | $0.05$ | $12$ |
| P2 | nucleus | $(25,25)$ | $(32,32)$ | $0.50$ | $100$ | $0.10$ | $20$ |
| P3 | trigonometric | $(6\pi,10\pi)$ | $(24,40)$ | $0.40$ | $250$ | $0.04$ | $15$ |
| P4 | nucleus | $(32,20)$ | $(48,24)$ | $0.30$ | $50$ | $0.08$ | $18$ |
| P5 | trigonometric | $(10\pi,4\pi)$ | $(40,24)$ | $0.20$ | $120$ | $0.06$ | $14$ |

where "steps" is the number of scheme steps taken after the prescribed start-up, so the run ends at $\phi^{2+\text{steps}}$.

*What to report.* Compute, for each configuration, the **modified** discrete energy $\hat E$ of the terminal state - it is built from the last $q+1=4$ levels of that run's history, which the prescribed start-up supplies for every configuration, including one taking zero steps - and report as the single final answer the **sum of those five terminal modified energies**.

State in your reasoning, and nothing else beyond the few scalars that determine the final number: (i) the symbol $\lambda_{k,m}$ you used, its signed index set, and the two Nyquist conventions; (ii) that each axis carries its own edge length and its own point count - in the symbol, in both first-derivative multipliers and in the cell area that weights every inner product; (iii) the direct BDF$q$ coefficients you obtained from the convolution kernel $\beta^{(q)}$, and the extrapolation coefficients $\alpha_j^{(q)}$; (iv) the discrete energy expression you evaluated, and in particular how the quartic term is taken; (v) how you evaluated $N(\phi)$, and why expanding it by the chain rule is not equivalent; (vi) where the stabilization term ends up once $\mu^n$ has been eliminated; (vii) the Fourier symbol $A_{k,m}$ you inverted, why it is strictly positive at every wavenumber including $k=m=0$, and the two wavenumbers at which one of its terms vanishes; (viii) why the $\mathbf J$ term of $\hat E$ carries the factor $\varepsilon$; (ix) the original discrete energy $E$ of each of the five initial fields; (x) the original discrete energy $E$ of each of the five terminal fields; (xi) whether $E$ and $\hat E$ decreased at every step of every configuration, and whether $E$ still does in the $S=0$ control; (xii) the relative drift of the discrete mass over each run, what you attribute that drift to, and the mechanism in the scheme that makes the discrete mass exactly conserved; (xiii) the terminal $\hat E$ of all five configurations, and that they sum to your final answer; (xiv) the value $\hat E$ takes at $n=q-1$ under the prescribed start-up, the value of $E$ at that same level for at least one configuration, and the reason the two must agree; (xv) $S_{\min}$ and the resulting $S=sS_{\min}$ for each of the five configurations; (xvi) $C_3$ and $C_{\mathrm{req}}$ for P1, and whether the $S$ actually used satisfies the condition; (xvii) the value the identical pipeline returns with zero steps, and the value it returns with $S=0$ on every configuration; and (xviii) the numerical $\mathbf G_3$, $\mathbf J_3$, $\kappa_3$ and $\eta_3$ you used, and where you obtained them.

Output Format Requirements: Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure. Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: `0.4847`, `12.6`, `1.05`). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep `<reasoning>` short (a few hundred words): the diagnostics listed above are its required content and must all appear, and beyond them show only the few scalars that determine the final number. Do not paste the field arrays, the Fourier coefficients, or per-step tables of more than the energies.

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

laplacian_symbol

Goal
----
Build the Fourier symbol of the negative Laplacian $-\\Delta_h$ on a periodic rectangle discretised by the Fourier pseudo-spectral method. The rectangle is $(0,L_x)\\times(0,L_y)$ and carries $M_x\\times M_y$ equispaced points with $M_x$ and $M_y$ even; the admissible wavenumber indices are the signed set $-M_x/2\\le k\\le M_x/2-1$ and $-M_y/2\\le m\\le M_y/2-1$, stored in the order `numpy.fft.fftfreq` produces, so that the returned array can be multiplied elementwise against `numpy.fft.fft2` output without any reordering. Return the real array $\\lambda_{k,m}$ of shape `(Mx, My)`. The two axes are independent: each carries its own number of points and its own edge length, and neither the grid spacing nor a single scalar $L$ may be substituted for the pair.

```python
def sqpfc_laplacian_symbol(mesh: "int | Sequence[int]",
                           cell: "float | Sequence[float]") -> "np.ndarray":
    """mesh: number of grid points, a scalar or a length-2 sequence (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    Return the real array lambda_{k,m} = 4 pi^2 (k^2/Lx^2 + m^2/Ly^2) of shape
    (Mx, My), with the signed wavenumber indices in numpy fft storage order.
    Raise ValueError if either argument is a sequence whose length is not
    2, if mesh is not a positive whole number in both directions, or if a
    cell edge length is not positive and finite."""
    # Implement per the specification above.
    return None
```

### Step 2

bdf_kernels

Goal
----
Return the two multistep kernels of order $q\\in\\{3,4,5\\}$ that the time discretisation needs, stacked as a `(2, q)` array. Row $0$ is the BDF$q$ **convolution kernel** $\\beta^{(q)}$, the weights that act on FIRST DIFFERENCES in $\\mathcal{B}_qv^n=\\tfrac1\\tau\\sum_{j\\ge0}\\beta_j^{(q)}\\delta_\\tau v^{n-j}$ where $\\delta_\\tau v^n=v^n-v^{n-1}$. Row $1$ is the extrapolation kernel $\\alpha^{(q)}$ appearing in $\\hat v_q^{\\,n}=v^n-\\delta_\\tau^qv^n=v^n-\\sum_{j\\ge0}\\alpha_j^{(q)}\\delta_\\tau v^{n-j}$, where $\\delta_\\tau^m$ denotes the $m$-fold first difference. Both kernels vanish for $j\\ge q$, so only the first $q$ entries are returned. **Row $0$ is not the direct BDF$q$ coefficient vector** acting on $v^n,\\dots,v^{n-q}$: the two vectors share only their leading entry, and the conversion between them belongs to the step that uses them, not here.

```python
def sqpfc_bdf_kernels(q: int) -> "np.ndarray":
    """q: BDF order, one of 3, 4, 5.
    Return the real array [beta^{(q)}, alpha^{(q)}, gamma^{(q)}] of shape
    (3, q): beta the BDFq convolution kernel acting on first differences,
    alpha the extrapolation kernel, and gamma the leading q entries of the
    discrete orthogonal convolution (DOC) kernel, the convolution inverse
    of beta.
    Raise ValueError if q is not one of 3, 4, 5."""
    # Implement per the specification above.
    return None
```

### Step 3

spectral_gradient

Goal
----
Apply the two first-derivative operators $D_x$ and $D_y$ of the Fourier pseudo-spectral discretisation to one real periodic field, and return them stacked as $\\nabla_hv=(D_xv,D_yv)^T$ with shape `(2,) + v.shape`. Each is applied by transforming, multiplying by that direction's symbol, transforming back and taking the real part. **The Nyquist entry is dropped**: the multiplier is set to zero at $k=-M_x/2$ for $D_x$ and at $m=-M_y/2$ for $D_y$ before the inverse transform. No dealiasing of any other kind is applied. Each axis carries its own edge length and its own point count.

```python
def sqpfc_spectral_gradient(phi: "np.ndarray",
                            cell: "float | Sequence[float]") -> "np.ndarray":
    """phi: real periodic field of shape (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    Return the real array [D_x phi, D_y phi] of shape (2,) + phi.shape, with
    the Nyquist entry of each first-derivative multiplier set to zero.
    Raise ValueError if phi is not two-dimensional, or if cell is not a
    positive scalar or length-2 sequence."""
    # Implement per the specification above.
    return None
```

### Step 4

discrete_energy

Goal
----
Evaluate the original discrete free energy of the square phase field crystal model, $E=\\tfrac14\\|\\nabla_h\\phi\\|_4^4-\\tfrac{\\varepsilon}{2}\\|\\phi\\|^2+\\tfrac12\\|(1+\\Delta_h)\\phi\\|^2$, and return it as a single float. All three norms carry the discrete inner product $\\langle u,w\\rangle=h_xh_y\\sum_{ij}u_{ij}w_{ij}$, so every sum is weighted by the cell area $h_xh_y=(L_x/M_x)(L_y/M_y)$. The quartic is a **gradient** quartic and the modulus inside it is the pointwise Euclidean length of the vector $\\nabla_h\\phi$, so the summand is $\\big((D_x\\phi)^2+(D_y\\phi)^2\\big)^2$: it is neither $\\phi^4$ nor the square of the squared $L^2$ norm of the gradient. The concave term enters with a minus sign.

```python
def sqpfc_discrete_energy(phi: "np.ndarray", cell: "float | Sequence[float]",
                          eps: float) -> float:
    """phi: real periodic field of shape (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    eps: the parameter epsilon of the model.
    Return the float E = (1/4)||grad_h phi||_4^4 - (eps/2)||phi||^2
    + (1/2)||(1 + Lap_h) phi||^2, every norm carrying the cell area hx*hy.
    Raise ValueError if phi is not two-dimensional."""
    # Implement per the specification above.
    return None
```

### Step 5

quartic_divergence

Goal
----
Evaluate the nonlinear term of the chemical potential, $N(\\phi)=-\\nabla_h\\cdot\\big(|\\nabla_h\\phi|^2\\nabla_h\\phi\\big)$, and return it as a real array with the same shape as $\\phi$. Build it in the order the expression is written: form the vector $\\nabla_h\\phi=(D_x\\phi,D_y\\phi)$, form the scalar $|\\nabla_h\\phi|^2=(D_x\\phi)^2+(D_y\\phi)^2$ pointwise, multiply it into each component to get the flux $\\boldsymbol F=|\\nabla_h\\phi|^2\\nabla_h\\phi$, then take $-\\big(D_xF_x+D_yF_y\\big)$. Both the gradient and the divergence use the first-derivative operators of the earlier step, with their Nyquist entries dropped. **Do not expand by the chain rule**: $-\\nabla_h\\cdot(|\\nabla_h\\phi|^2\\nabla_h\\phi)$ is not $-3|\\nabla_h\\phi|^2\\Delta_h\\phi$, because $|\\nabla_h\\phi|^2$ is not constant, and the discrete product rule fails for spectral differences in any case.

```python
def sqpfc_quartic_divergence(phi: "np.ndarray",
                             cell: "float | Sequence[float]") -> "np.ndarray":
    """phi: real periodic field of shape (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    Return the real array N(phi) = -div_h(|grad_h phi|^2 grad_h phi) of the
    same shape as phi.
    Raise ValueError if phi is not two-dimensional."""
    # Implement per the specification above.
    return None
```

### Step 6

convex_split_step

Goal
----
Advance the fully discrete BDF$q$ convex-splitting scheme by one time step and return the new field $\\phi^n$. The input `hist` is the array $[\\phi^{n-q},\\phi^{n-q+1},\\dots,\\phi^{n-1}]$ of shape `(q, Mx, My)`, oldest first, and $q$ is read from its leading axis. The scheme is $\\mathcal{B}_q\\phi^n=\\Delta_h\\mu^n$ with $\\mu^n=N(\\phi^n)-\\varepsilon\\hat\\phi_q^{\\,n}+(1+\\Delta_h)^2\\phi^n-S\\tau^q\\Delta_h\\mathcal{B}_q\\phi^n$, where $N$ is the quartic divergence of the previous step. Eliminate $\\mu^n$, move every term linear in $\\phi^n$ to the left, and invert the resulting constant-coefficient operator in Fourier space; the remaining nonlinearity is resolved by the Picard iteration this rearrangement defines, started from $\\phi^{(0)}=\\hat\\phi_q^{\\,n}$ and stopped when $\\max_{ij}|\\phi^{(s+1)}-\\phi^{(s)}|<10^{-13}$, with a hard cap of $200$ iterations.

```python
def sqpfc_convex_split_step(hist: "np.ndarray", cell: "float | Sequence[float]",
                            eps: float, S: float, tau: float,
                            tol: float = 1e-13,
                            max_iter: int = 200) -> "np.ndarray":
    """hist: array [phi^{n-q}, ..., phi^{n-1}] of shape (q, Mx, My), oldest
    first; q is read from the leading axis and must be 3, 4 or 5.
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    eps: the parameter epsilon.  S: stabilization parameter.  tau: time step.
    tol: increment at which the Picard iteration is stopped.
    max_iter: hard cap on the number of Picard sweeps.
    Return the real array phi^n of shape (Mx, My), the fixed point of the
    Picard iteration, stopped when the max-norm increment falls below tol.
    Raise ValueError if hist does not have shape (q, Mx, My) with q in
    {3, 4, 5}, if tau or tol is not positive, if S is negative, or if
    max_iter is below 1.  Raise RuntimeError if the increment has not
    fallen below tol within max_iter sweeps: the last iterate is not the
    step and must not be returned in its place."""
    # Implement per the specification above.
    return None
```

### Step 7

modified_energy

Goal
----
Evaluate the modified discrete energy $\\hat E^n$, the quantity whose decay the scheme's stability theorem establishes - for data band-limited strictly below Nyquist, a hypothesis this task's convention does not make automatic and which the background states in full. The input `hist` is $[\\phi^{n-q},\\dots,\\phi^{n}]$ of shape $(q+1, Mx, My)$, oldest first, so that the $q$ first differences $\\delta_\\tau\\vec\\phi^{\\,n}=(\\delta_\\tau\\phi^{n-q+1},\\dots,\\delta_\\tau\\phi^{n})$ can be formed; $q$ is read from the shape of the supplied matrix $\\mathbf G$. The quantity to evaluate is the modified discrete energy defined in the problem statement, with $\\mathbf G$ and $\\mathbf J$ supplied as arguments and $\\langle\\vec a,\\vec b\\rangle$ the sum of the componentwise discrete inner products over the $q$ slots; how each of its terms ends up scaling in $\\tau$ is part of what is graded. where $E^n$ is the original discrete energy of the LAST slot, $\\langle u,w\\rangle_{-1}=\\langle(-\\Delta_h)^{-1}u,w\\rangle$ with the zero Fourier mode of $(-\\Delta_h)^{-1}$ set to zero, and the middle term contracts the two components of each gradient as well as the $q$ slots. **Note the factor $\\varepsilon$ on the $\\mathbf J$ term.** Return a single float.

```python
def sqpfc_modified_energy(hist: "np.ndarray", cell: "float | Sequence[float]",
                          eps: float, S: float, tau: float,
                          G: "np.ndarray", J: "np.ndarray") -> float:
    """hist: array [phi^{n-q}, ..., phi^n] of shape (q+1, Mx, My), oldest first.
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    eps, S, tau: the parameter epsilon, the stabilization parameter, the step.
    G, J: the (q, q) quadratic-decomposition matrices; q = G.shape[0].
    Return the float Ehat^n defined above, with the factor eps on the J term.
    Raise ValueError if G and J are not square matrices of the same shape,
    if hist does not have shape (q+1, Mx, My) with q = G.shape[0], if tau
    is not positive, or if S is negative."""
    # Implement per the specification above.
    return None
```

### Step 8

stabilization_certificate

Goal
----
Evaluate the stabilization condition that the energy-dissipation proof requires, and return the four diagnostics $[C_q,\\;C_{\\mathrm{req}},\\;S_{\\min},\\;C_q-C_{\\mathrm{req}}]$ as a length-4 array. Here $C_q$ is the step-size-independent constant the argument produces from the stabilization parameter $S$, $C_{\\mathrm{req}}$ is the value it has to reach, $S_{\\min}$ is the smallest $S$ for which it does, and the last entry is the margin, non-negative exactly when the condition holds. The inputs are the order $q$, the two constants $\\kappa_q>0$ and $\\eta_q$ of the two quadratic decompositions, the model parameter $\\varepsilon$, and the stabilization parameter $S\\ge0$. The formulas are valid for $q\\ge3$.

```python
def sqpfc_stabilization_certificate(q: int, kappa_q: float,
                                    eta_q: float, eps: float,
                                    S: float) -> "np.ndarray":
    """q: BDF order, at least 3.
    kappa_q: the positive constant of the BDF quadratic decomposition.
    eta_q: the constant of the extrapolation quadratic decomposition.
    eps: the parameter epsilon.  S: the stabilization parameter, >= 0.
    Return np.array([C_q, C_required, S_min, C_q - C_required]).
    Raise ValueError if q is not a whole number of at least 3, if kappa_q
    is not positive, or if S is negative."""
    # Implement per the specification above.
    return None
```

### Step 9

terminal_energy_sum

Goal
----
**Final orchestrator.** Run the BDF$3$ convex-splitting scheme on each of five fixed configurations and return the sum of the five terminal **modified** discrete energies. Nothing about a configuration is hard-wired except the numbers in its row: in particular the stabilization parameter is derived, not given. For each configuration, in order: call the certificate of the earlier sub-problem at $q=3$, at the published $q=3$ decomposition constants $\\kappa_3$ and $\\eta_3$ - given in the formulas below - and at that configuration's $\\varepsilon$, read $S_{\\min}$ off it and set $S=s\\,S_{\\min}$ with $s$ the configuration's safety factor; build the initial field $\\phi_0$ from its seed; set the start-up layers $\\phi^0=\\phi^1=\\phi^2=\\phi_0$ and carry a fourth copy, so that even a run of zero steps holds the $q+1=4$ levels the modified energy needs; take the prescribed number of steps with the step of the earlier sub-problem; and evaluate the modified discrete energy of the last four levels with $\\mathbf G_3$, $\\mathbf J_3$ and that configuration's $S$ and $\\tau$. The five configurations, in this order, are P1: trigonometric seed, $(L_x,L_y)=(8\\pi,8\\pi)$, $(M_x,M_y)=(32,32)$, $\\varepsilon=0.25$, $s=200$, $\\tau=0.05$, $12$ steps; P2: nucleus seed, $(25,25)$, $(32,32)$, $\\varepsilon=0.50$, $s=100$, $\\tau=0.10$, $20$ steps; P3: trigonometric, $(6\\pi,10\\pi)$, $(24,40)$, $\\varepsilon=0.40$, $s=250$, $\\tau=0.04$, $15$ steps; P4: nucleus, $(32,20)$, $(48,24)$, $\\varepsilon=0.30$, $s=50$, $\\tau=0.08$, $18$ steps; P5: trigonometric, $(10\\pi,4\\pi)$, $(40,24)$, $\\varepsilon=0.20$, $s=120$, $\\tau=0.06$, $14$ steps. The keyword arguments override, for every configuration at once, the mesh, the step size, the number of steps and the stabilization parameter - an $S$ passed explicitly bypasses the certificate; `configs` restricts the sum to the named subset. With no arguments the function returns the graded answer.

```python
def sqpfc_terminal_energy_sum(configs: "Sequence[str] | None" = None,
                              mesh: "int | Sequence[int] | None" = None,
                              tau: "float | None" = None,
                              n_steps: "int | None" = None,
                              S: "float | None" = None) -> float:
    """configs: list of configuration tags to include; None means all five.
    mesh, tau, n_steps, S: if given, override that setting for every
    configuration; None means use each configuration's own value.
    Return the float sum of the terminal modified discrete energies.
    Raise ValueError if configs names a tag that is not one of the five,
    if n_steps is negative, if tau is not positive, or if S is negative."""
    # Implement per the specification above.
    return None
```
