# Material_Science-Semiconductor_Materials-70

## Background

The two parameters do different jobs and their errors have opposite signs in the same variable. More sample points improve the trapezoidal estimate of every Fourier coefficient and do nothing else, so their benefit saturates once the quadrature error has been driven below the truncation error. More retained modes let the represented field follow the material interface more sharply, which helps, but the coefficients they admit are the ones whose quadrature estimates are worst, because that error grows with the frequency index, and they are also the ones that ring near a discontinuity. Where the two effects balance is a property of the microstructure and of the contrast rather than of the algorithm, and it moves towards keeping everything as the contrast falls towards one, since a homogeneous material has nothing to ring at.

A field represented by a truncated Fourier series is a function defined at every point of the cell, not a table of values at the points that produced it. Reading it only at those points is a choice, and at high truncation it is a choice that hides the thing being measured.

## Problem

Spectral solvers for the transport properties of a heterogeneous material replace the heterogeneous problem by a homogeneous reference medium carrying a polarisation field, and iterate: the constitutive law is applied point by point in real space, where heterogeneity costs no more than homogeneity, and equilibrium is imposed mode by mode in Fourier space, where a homogeneous reference medium makes the operator diagonal. Two numbers control the discretisation. One is the number of grid points, which fixes how accurately each Fourier coefficient is estimated by the trapezoidal rule. The other is the number of Fourier modes the represented field is allowed to keep. They are habitually set equal, so habitually only one of them is ever named, and the habit is a convention rather than a result: estimating a coefficient requires at least as many sample points as there are coefficients to estimate, and nothing requires the reverse. Your task is to find, for one prescribed two-phase semiconductor film on one prescribed grid, how many modes should actually be kept.

The film is a periodic square array of heavily doped square regions in a lightly doped host, both the same crystal, carrying steady ohmic conduction with $\operatorname{div} J = 0$ and $J = c(x) E$, and $E$ the gradient of a potential whose fluctuating part is periodic. Each region is extrinsic and n-type with every donor ionised, so its conductivity is the elementary charge times the donor density times the electron mobility at that donor density. The two factors move against each other, since heavier doping supplies more carriers and scatters them harder, and the conductivity contrast is therefore not the doping ratio. Within the cell $[0, L] \times [0, L]$ the heavily doped phase occupies $x_1 < L/2$ and $x_2 < L/2$, a square of half the cell edge and a quarter of its area. Every sample point is assigned to a phase by that strict inequality and by nothing else, the point at the origin included, which puts it inside the heavily doped square; no sample point lies on the interface at $L/2$, since with $N$ even the number $N + 1$ is odd and $L/2$ falls midway between two sample coordinates. The points on the cell edges $x_1 = 0$ and $x_2 = 0$, through which the periodic image of the other face of the square passes, are assigned to the heavily doped phase by the same strict inequality, so there is no tie to break and no averaging to do anywhere. Repeated periodically the microstructure is a square lattice of squares, and the corners at which four quadrant boundaries meet make it a severe test of any representation by a finite series.

Discretise as follows. The cell carries $N + 1$ sample points along each edge at spacing $\Delta x = L / (N + 1)$, the point of index $a$ at $a \Delta x$, so the discrete transform has odd length $N + 1$ and its frequency indices $i = -N/2, \dots, N/2$ are one complete residue system, symmetric about zero with no unpaired folding mode to special-case. The wavevector of index $i$ is $\xi_i = 2 \pi i / L$. Independently of that, the potential fluctuation is represented by a partial Fourier series of order $M + 1$ per direction, retaining the modes with $|i| \le M/2$ and $|j| \le M/2$ and no others, with $M$ even and $M \le N$; the mean mode carries the applied loading. Every sweep of the iteration must leave the represented field inside that space, so whatever the update produces above the truncation is discarded rather than kept. Start the iteration from the uniform field equal to the applied mean. The Green operator is the one the standard spectral derivative rule gives,

$$\hat{\Gamma}(\xi) = \frac{\xi \otimes \xi}{c_0 |\xi|^{2}},$$

with $c_0$ the reference conductivity. Stop the iteration when the equilibrium residual

$$\frac{L}{2 \pi} \frac{\langle \|\operatorname{div} J\|^{2} \rangle^{1/2}}{\| \langle J \rangle \|}$$

falls below the stated tolerance, the divergence being summed over the retained modes and no others, since the represented field spans exactly those and no admissible field can cancel a divergence above them.

Measure the quality of a converged field in the energy norm the problem supplies,

$$\mathrm{err} = \frac{\int_{\Omega} (E_{\mathrm{ex}} - E_{\mathrm{num}}) \cdot c \cdot (E_{\mathrm{ex}} - E_{\mathrm{num}}) \mathrm{d}\Omega}{\int_{\Omega} E_{\mathrm{ex}} \cdot c \cdot E_{\mathrm{ex}} \mathrm{d}\Omega},$$

with $E_{\mathrm{ex}}$ the exact field, meaning the solution of the conduction problem posed on the real microstructure rather than on any discretisation of it. Three things about that integral are part of the specification. The conductivity in it is the conductivity of the real material, whose interface lies exactly at $L/2$ and whose area fraction is exactly one quarter, and not the array of values the grid sampled from it; the grid sampling is used to apply the constitutive law inside the iteration, and the only other thing it is used for is the macroscopic conductivity reported below, which is read off the mean Fourier mode of the current the iteration itself produces. The numerical field in it is the trigonometric polynomial the solver produced, evaluated everywhere in the cell and not only at the points it was computed on, because a truncated series near a discontinuity oscillates on the scale of its own highest mode and a grid of that spacing can sit on the nodes of the ripple and miss it entirely. And the integral is to be evaluated exactly, in the sense that the reported figures must not move at the tenth significant figure under any refinement of any quadrature used; the integrand is a trigonometric polynomial multiplied by a piecewise-constant indicator, so exact evaluation is available. This microstructure has an exact effective conductivity in closed form; it is not any mean-field estimate, it is not given here, and it is required.

Use the following deterministic configuration.

- surrounding phase: ionised donor density $1.0 \times 10^{22}$ per cubic metre and electron mobility $0.1200$ square metre per volt second
- embedded phase: ionised donor density $5.0 \times 10^{24}$ per cubic metre and electron mobility $0.0240$ square metre per volt second
- elementary charge $1.602176634 \times 10^{-19}$ coulomb
- cell edge $L = 2.0 \times 10^{-6}$ m, the embedded square occupying the quarter of the cell with both coordinates below $L/2$
- applied mean electric field $(1.0 \times 10^{5}, 0)$ volt per metre
- grid parameter $N = 96$, so $97$ sample points along each edge
- reference conductivity $c_0$ the arithmetic mean of the two phase conductivities
- residual tolerance $10^{-9}$ in the dimensionless form written above, and at most $20000$ sweeps per solve
- candidate truncations $M = 4, 8, 12, \dots, 96$, every multiple of four up to and including $N$, and these only, so that the answer is a finite computation and not a search
- fields evaluated between the sample points on a uniform grid four times finer in each direction, so $4 \times 97 = 388$ points along each edge, which contains the sample grid and is long enough for the energy integral above to be exact at every candidate
- every reported figure carried in at least double precision

Report the ratio $M^{*} / N$ at which the energy-norm error is smallest over that list of candidates, to at least six decimal places. The answer is graded within $0.005$. Say also whether the value you obtain agrees with what has been reported for a checkerboard at this contrast, and note anything in the published treatment of the generalised operator below that does not survive a dimensional check.

Report alongside it, briefly: the two phase conductivities in siemens per metre and their contrast; the exact effective conductivity in siemens per metre, and the two-dimensional Hashin-Shtrikman interval that brackets it; the area fraction the grid actually samples, against the exact quarter; the energy-norm error at $M^{*}$ and at $M = N$, and the ratio of the second to the first; the macroscopic conductivity in siemens per metre at $M^{*}$ and at $M = N$, taken as the ratio of the mean current the iteration produces to the applied mean field, that is from the mean Fourier mode alone, and the relative error of each against the exact value; and the largest magnitude the first field component reaches on the reconstruction grid at $M = N$, divided by the largest it reaches on the sample grid. That ratio is a maximum over the stated set of points and not the supremum of the underlying polynomial; the two differ, and the reported figure is tied to the stated refinement.

Then repeat the sweep with the derivative rule changed, taking a forward difference quotient of the potential and a backward difference quotient of the current, each over a spacing of half the grid spacing rather than in the limit of vanishing spacing, and report the minimising ratio and the smallest value the same expression reaches, which for that family is an excess of apparent conductivity over the exact value rather than an energy-norm error: the expression is no longer certified non-negative by a variational argument once the derivative rule changes, because the field it produces is no longer a gradient in the ordinary sense. It is still the quantity to minimise, and it is what the two families are compared on. That family reduces to the operator above as the spacing goes to zero and to a finite-difference construction when the spacing equals the grid spacing, and the limit is the check that fixes it; an operator of the wrong physical dimension still converges, to a field that is wrong by a factor varying across the spectrum.

Everything named in the two paragraphs above is part of the answer and belongs inside the reasoning. State, in one or two sentences each: by what route the error above was evaluated without constructing the exact field, and why that route is legitimate; why the number of grid points must be at least the number of retained modes, and why nothing requires them to be equal; why the converged field does not depend on the reference conductivity although the number of sweeps does; why the truncation that is best for the field is not the one that is best for the macroscopic conductivity, whose relative error behaves differently across the same sweep; and what happens to the shape of the error curve if the cell average in that expression is instead formed on the sample grid, from the sampled conductivity array paired with the field values at the sample points.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-candidate tables.

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

01_resolve_phase_conductivities

Goal
----
The medium is a two-phase semiconductor film in which both phases are the same host crystal doped to different levels, so the only thing that distinguishes them is how many carriers each carries and how freely those carriers move. For an extrinsic n-type region in which every donor is ionised and minority carriers are negligible, the ohmic conductivity is

$$c = q * N * mu,$$

with q the elementary charge, N the donor density and mu the electron mobility at that donor density. The two factors do not move together. Raising the doping raises N, but it also raises the ionised-impurity scattering rate, so mu falls; over the range that separates a lightly doped region from a degenerately doped one the mobility can fall by a factor of five while the doping rises by a factor of five hundred. The conductivity contrast between the two regions is therefore not the doping ratio, and taking it to be the doping ratio puts the whole calculation on the wrong material. Both mobilities are supplied here, so no mobility model has to be fitted; what has to be done is to keep the two factors separate.

The contrast that results is the single number the rest of the problem depends on. Everything downstream is scale-free in the conductivity: multiplying both phases by a common factor multiplies the effective conductivity by that factor and leaves every relative error unchanged, so only the ratio of the two phase conductivities can affect a dimensionless answer.

Three classical estimates bracket the effective conductivity of any two-phase isotropic composite with the given area fraction f of the more conducting phase, and they are computed here so that the exact value obtained later has something to be tested against. The Wiener bounds are the arithmetic and harmonic means,

$$c_{W+} = f * c_2 + (1 - f) * c_1, c_{W-} = 1 / [f / c_2 + (1 - f) / c_1],$$

and they use no information beyond the volume fractions. The two-dimensional Hashin-Shtrikman bounds add the assumption that the composite is isotropic in the plane and are correspondingly tighter,

$$c_{HS-} = c_1 + f / [1 / (c_2 - c_1) + (1 - f) / (2 * c_1)],$$

$$c_{HS+} = c_2 + (1 - f) / [1 / (c_1 - c_2) + f / (2 * c_2)],$$

where the factor 2 in each denominator is the space dimension and would be 3 in three dimensions. At high contrast these four numbers are spread over more than an order of magnitude, which is the point: bounds of this kind confirm an arithmetic slip but they come nowhere near fixing the answer, and that is why an exact solution is needed at all.

```python
def resolve_phase_conductivities(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    area_fraction: float,
) -> dict:
    """Turn the doping and mobility of each phase into its conductivity, and bracket the composite.

    Parameters
    ----------
    donor_density_matrix : float
        Ionised donor density of the surrounding phase in reciprocal cubic metre, above zero.
    mobility_matrix : float
        Electron mobility of the surrounding phase in square metre per volt second, above zero.
    donor_density_inclusion : float
        Ionised donor density of the embedded phase in reciprocal cubic metre, above zero.
    mobility_inclusion : float
        Electron mobility of the embedded phase in square metre per volt second, above zero.
    area_fraction : float
        Area fraction of the embedded phase, strictly between zero and one.

    Returns
    -------
    dict
        Under the keys c_matrix, c_inclusion, contrast, wiener_lower, wiener_upper, hs_lower and hs_upper.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when any density or mobility fails to be above zero, when the area fraction falls outside the open interval from zero to one, or when the embedded phase fails to be the more conducting of the two.
    """
    return
```

### Step 2

02_spectral_lattice

Goal
----
The auxiliary problem that the iterative solver rests on is written in Fourier space, so before anything can be solved the lattice of frequencies has to be laid out, and the convention chosen here is not the one a routine FFT setup produces.

The cell is the square of side L, and it is sampled at N + 1 points along each edge with spacing

$$dx = L / (N + 1),$$

the point of index a sitting at a * dx for a = 0, ..., N. There are therefore N + 1 samples per direction, one period wide, and the discrete transform has odd length N + 1. That is deliberate. With N even the frequency indices

$$i = -N/2, ..., N/2$$

are N + 1 consecutive integers, which is exactly one complete residue system modulo N + 1, so every index names a distinct exponential and the set is symmetric about zero. An even-length transform cannot do both at once: it has a single unpaired mode at the folding frequency whose exponential is real on every sample point, and that mode has to be treated as a special case in any scheme built on it. Here there is no such mode and no special case. The wavevector attached to index i is

$$xi_i = 2 * pi * i / L,$$

and in two dimensions the pair (i, j) carries xi = (xi_i, xi_j).

Separately from the grid, the unknown potential fluctuation is represented by a partial Fourier series of order M + 1 per direction, retaining the modes with |i| <= M/2 and |j| <= M/2, with M an even integer no larger than N. This is the whole point of the construction: M and N are two independent numbers. N says how many points the trapezoidal rule uses to estimate Fourier coefficients, and M says how many of those coefficients the represented field is allowed to keep. The condition N >= M is required, because estimating a coefficient needs at least as many sample points as there are coefficients to estimate; the reverse inequality is not required by anything, and the habitual choice M = N is a convention rather than a consequence. The mode at i = j = 0 is excluded from the retained set, because the mean of the fluctuation is fixed at zero and the mean of the field is fixed by the applied loading instead.

The Green operator is assembled from a symbol vector beta which, for the standard spectral derivative rule, is the wavevector itself, beta = xi. In general it is built as

$$Gamma(xi) = conj(beta) tensor beta / [c0 * conj(beta) . beta],$$

with c0 the conductivity of the homogeneous reference medium. Written this way one property is visible immediately and is worth using as the check on the assembly: c0 * Gamma is an orthogonal projector. It is idempotent, because conj(beta) tensor beta applied to itself returns itself times the scalar conj(beta) . beta, and its trace is exactly one, so it projects each mode of the polarisation onto the single direction that a gradient field is allowed to occupy at that frequency. Any assembly that fails either property has the wrong operator, and the failure will not otherwise show until the iteration has converged to the wrong field.

```python
def spectral_lattice(
    n_grid: int,
    n_modes: int,
    period: float,
    reference: float,
) -> dict:
    """Lay out the frequency lattice, the retained-mode set and the Green operator.

    Parameters
    ----------
    n_grid : int
        The number N of grid intervals, even and above zero.
    n_modes : int
        The truncation order M, even, above zero and not above N.
    period : float
        The cell edge in metre, above zero.
    reference : float
        The reference conductivity in siemens per metre, above zero.

    Returns
    -------
    dict
        Under the keys frequency_index, wavevector, symbol, green, retained, n_points, n_retained, spacing, projector_defect and trace_defect. Write n for n_grid + 1. frequency_index is the one-dimensional integer array of the n signed frequency indices, in the order the discrete transform uses. wavevector is a real array of shape (2, n, n) and symbol a complex array of the same shape, the leading axis running over the two directions. green is a complex array of shape (2, 2, n, n), its two leading axes running over the operator's row and column direction. retained is a boolean array of shape (n, n). The remaining five entries are scalars: n_points is n itself, the number of sample points along one edge and not over the two-dimensional grid; n_retained is the number of true entries of retained, counted over the whole (n, n) mask; spacing is the sample spacing, the period divided by n; and projector_defect and trace_defect are the departures of the scaled operator from idempotence and from unit trace, each taken over the retained modes.

    Raises
    ------
    ValueError
        When N or M fails to be a positive even integer, when M exceeds N, or when the period or the reference conductivity fails to be finite and above zero.
    """
    return
```

### Step 3

03_checkerboard_conductivity

Goal
----
The microstructure is a square array of heavily doped squares set in a lightly doped host. Within the cell [0, L] x [0, L] the embedded phase occupies

$$x1 < L/2 and x2 < L/2,$$

so it is a square of side L/2 filling exactly a quarter of the cell area, and the grid point of index (0, 0) sits on its lower left corner. Repeated periodically this is a square lattice of squares of side L/2 on a lattice of pitch L, and the corners at which four quadrant boundaries meet are what make it a severe test: the exact field has an integrable singularity at each of them, and a representation by a finite trigonometric series has to approximate a discontinuity that is not aligned with any single mode.

The conductivity used by the solver is this field sampled at the N + 1 grid points per direction, at the positions a * L / (N + 1). Sampling it is not free of consequence. With N even there are N/2 + 1 sample coordinates strictly below L/2 out of N + 1, so the sampled area fraction is

$$[(N/2 + 1) / (N + 1)]^2,$$

which exceeds one quarter by a term of order 1/N. The grid therefore represents a square slightly larger than the real one, and no refinement of the Fourier truncation can remove that: it is an error in the geometry rather than in the representation of the field on it, and it is the reason the achievable error settles at a floor rather than falling to zero. Reporting it explicitly is how that floor is accounted for afterwards rather than mistaken for a failure of the solver.

It matters equally that the sampled field is used for one purpose only, namely applying the constitutive law pointwise inside the iteration. Whenever the resulting field is later measured against the truth, the truth belongs to the real microstructure, with its interface exactly at L/2 and its area fraction exactly one quarter, and not to the sampled copy. Conflating the two removes the whole effect being studied, because a field scored against the material the solver was handed is being scored against its own assumptions.

```python
def checkerboard_conductivity(
    n_grid: int,
    c_matrix: float,
    c_inclusion: float,
    period: float,
) -> dict:
    """Sample the two-phase conductivity field on the grid and report what the sampling costs.

    Parameters
    ----------
    n_grid : int
        The number N of grid intervals, even and above zero.
    c_matrix : float
        Conductivity of the surrounding phase in siemens per metre, above zero.
    c_inclusion : float
        Conductivity of the embedded phase in siemens per metre, above zero and above c_matrix.
    period : float
        The cell edge in metre, above zero.

    Returns
    -------
    dict
        Under the keys conductivity, coordinate, inclusion_points, sampled_area_fraction, exact_area_fraction and fraction_defect. Write n for n_grid + 1. conductivity is a real array of shape (n, n) indexed by the two sample indices, and coordinate is the one-dimensional array of the n sample coordinates. The remaining four entries are scalars: inclusion_points is the number of sample coordinates along one edge that fall below half the cell edge, counted in one direction and not over the two-dimensional grid, so that sampled_area_fraction is its square divided by n squared; exact_area_fraction is one quarter; and fraction_defect is sampled_area_fraction minus exact_area_fraction.

    Raises
    ------
    ValueError
        When N fails to be a positive even integer, when either conductivity or the period fails to be finite and above zero, or when the embedded phase fails to be the more conducting of the two.
    """
    return
```

### Step 4

04_moulinec_suquet_field

Goal
----
The heterogeneous conduction problem is solved by replacing it with an auxiliary problem posed on a homogeneous reference medium of conductivity c0 carrying a polarisation field,

$$J = c0 * E + tau, tau = [c(x) - c0] * E,$$

and iterating until the polarisation is the one the real material produces. Each sweep does two things in the two places each is cheap. In real space the constitutive law is applied point by point, which a heterogeneous material makes trivial and a convolution would not. In Fourier space the equilibrium of the auxiliary problem is imposed mode by mode through the Green operator, which a homogeneous reference medium makes trivial and a heterogeneous one would not. The sweep is

- form the current from the present field, J = c(x) * E, at every grid point;
- form the polarisation tau = J - c0 * E and transform it;
- replace every retained non-zero mode of the field by -Gamma(xi) applied to the transformed polarisation, set every mode outside the retained set to zero, and set the mean mode to the applied mean field;
- transform back.

Two features of that sweep are specific to this construction. The first is that the modes outside the retained set are set to zero rather than left alone: the represented field is a partial Fourier series of order M + 1 per direction, and the polarisation formed from it in real space is not, so its transform carries content above the truncation that the field is not permitted to hold. Discarding that content is the aliasing correction, and without it the update is not consistent with the space the field lives in. The second is that the mean mode is prescribed rather than solved: the operator annihilates it, and it carries the applied loading.

Convergence is measured by how far the current is from equilibrium. Equilibrium means the current is divergence free, and the divergence of a mode is the symbol beta contracted with that mode of the transformed current, so the natural residual is the root mean square divergence normalised by the mean current,

$$residual = [L / (2 * pi)] * sqrt(mean(|div J|^2)) / |mean(J)|,$$

the prefactor making it dimensionless so that the same tolerance means the same thing whatever the cell size. The sum runs over the retained modes and no others. That restriction is not a convenience: the represented field spans exactly those modes, so equilibrium against exactly those modes is what the discrete problem asks for, and the modes above the truncation carry a divergence that no admissible field can cancel. Including them would leave a residual that never falls below a floor set by the truncation itself. The residual is evaluated before any update and again after each one, so the count reported is the number of evaluations rather than the number of updates, and a starting field already in equilibrium reports one.

Three properties of the converged state are worth knowing, because each says something the iteration count does not. The converged field does not depend on c0, which only sets how fast the iteration contracts: at the fixed point every retained mode of the field is parallel to the direction the operator projects onto, the reference terms cancel, and what remains is the statement that the transformed current is orthogonal to that direction at every retained mode. The macroscopic conductivity follows from the mean modes alone,

$$c_num = mean(J) . Ebar / |Ebar|^2,$$

because the mean of a periodic fluctuation vanishes. And the energy of the converged field, evaluated with the same sampled conductivity the iteration used and averaged over the grid, equals c_num times |Ebar| squared exactly, which is the discrete form of the Hill-Mandel relation. It holds by Parseval: the grid average of J . E is the sum over frequencies of the transformed current against the transformed field, and the orthogonality just described annihilates every term of that sum except the mean one. This says nothing about the local energy density, whose own Fourier coefficients are a convolution and are not small; it is the average alone that collapses. That identity is returned as a defect and is the sharpest available check that the fixed point really is one. Its size tracks the stopping residual rather than the machine precision, so it certifies convergence and is not a constant.

```python
def moulinec_suquet_field(
    conductivity,
    symbol,
    green,
    retained,
    reference: float,
    mean_field,
    period: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Iterate the auxiliary problem to a divergence-free current on the retained modes.

    Parameters
    ----------
    conductivity : array
        Sampled conductivities of shape (n, n) in siemens per metre.
    symbol : array
        Symbol beta of shape (2, n, n) in reciprocal metre.
    green : array
        Green operator of shape (2, 2, n, n) in metre per siemens.
    retained : array
        Boolean mask of shape (n, n) marking the retained non-zero modes.
    reference : float
        Reference conductivity in siemens per metre, above zero.
    mean_field : sequence
        Two components of the applied mean electric field in volt per metre.
    period : float
        Cell edge in metre, above zero.
    tolerance : float
        Residual at which the iteration stops, above zero.
    max_iterations : int
        Largest number of sweeps permitted, above zero.

    Returns
    -------
    dict
        Under the keys electric_field, field_transform, iterations, residual, macroscopic_conductivity and hill_mandel_defect. electric_field is a real array of shape (2, n, n) and field_transform its complex unnormalised transform of the same shape, the leading axis running over the two directions as in symbol. The remaining four entries are scalars.

    Raises
    ------
    ValueError
        When the array shapes disagree, when the conductivity is not everywhere above zero, when the reference conductivity, the period or the tolerance fails to be finite and above zero, when the mean field is not two finite components of which at least one is non-zero, when no mode is retained, or when the residual has not fallen below the tolerance after max_iterations sweeps.
    """
    return
```

### Step 5

05_spectral_reconstruction

Goal
----
The field the solver produces is not a list of values at grid points. It is a partial Fourier series of order M + 1 per direction, and a partial Fourier series is a function defined at every point of the cell. The grid entered only as the quadrature rule that estimated the coefficients; once the coefficients are known the grid has done its work, and evaluating the series anywhere else is not interpolation but simply the same function read at other points.

That distinction has consequences that are easy to miss. A truncated series approximating a discontinuous field oscillates near the discontinuity, and the oscillation has a wavelength set by the highest retained mode, which is comparable with the grid spacing when M is comparable with N. Sampled at the grid points, oscillations of that wavelength are largely invisible: the samples can sit near the nodes of the ripple and return a field that looks clean. Evaluated between the grid points the same function shows the ripple at full amplitude. A field judged only at the points it was computed on can therefore appear far better than it is, and any measure of its quality has to be taken from the function rather than from the samples.

Reconstruction is done by evaluating

$$E(x) = sum over |i| <= M/2, |j| <= M/2 of F_ij * exp(i * xi_i * x1) * exp(i * xi_j * x2)$$

on a finer uniform grid. The Fourier series coefficients F are obtained from the discrete transform of the field by dividing by the number of grid points, since the transform approximates the coefficients through the trapezoidal rule. Carried out directly this costs one operation per fine point per retained mode. Done instead by placing the retained coefficients into a longer array whose remaining entries are zero and inverting a transform of that length, it costs the usual transform complexity in the length of the fine grid, and it returns exactly the same values: zero padding in the frequency domain is evaluation of the same trigonometric polynomial at more points, not smoothing and not resampling.

Taking the fine grid to be a whole multiple K of the computation grid, so that it carries K * (N + 1) points per direction, makes the computation grid a subset of it. That is worth doing for one reason beyond convenience: every K-th value of the reconstruction must then agree with the field on the computation grid to rounding error, which is the check that the padding was placed at the right positions. The frequency index of a mode is not its array position, and putting the retained coefficients at their array positions rather than at their indices modulo the fine length produces a plausible-looking field that is wrong. The node agreement is what catches that.

Reported alongside is the ratio of the largest magnitude the first field component reaches on the reconstruction grid to the largest it reaches on the computation grid. It is one when the coarse grid sees everything the refined one does and above one by the size of whatever the coarse grid was missing. Note what this ratio is not: the numerator is a maximum over a finite set of points, not the supremum of the underlying polynomial, and the two differ. Refining further moves the numerator, and not monotonically, because a finer grid samples a different set of points on the same ripple. The statistic is therefore tied to a stated refinement factor and is a diagnostic of what the computation grid hides rather than a property of the field alone.

```python
def spectral_reconstruction(
    field_transform,
    n_grid: int,
    n_modes: int,
    refinement: int,
) -> dict:
    """Evaluate the retained trigonometric polynomial on a refined grid and measure what the coarse grid hides.

    The reported peak ratio is taken over the reconstruction grid this call builds, so it
    depends on the refinement factor and is not the supremum of the polynomial.

    Parameters
    ----------
    field_transform : array
        Complex array of shape (2, N + 1, N + 1) holding the transform of the converged field.
    n_grid : int
        The number N of grid intervals, even and above zero.
    n_modes : int
        The truncation order M, even, above zero and not above N.
    refinement : int
        Whole refinement factor K, above zero.

    Returns
    -------
    dict
        Under the keys fine_field, coarse_field, n_fine, node_defect, peak_ratio, fine_peak and coarse_peak. fine_field is a real array of shape (2, n_fine, n_fine) and coarse_field a real array of shape (2, n_grid + 1, n_grid + 1), the leading axis running over the two directions as in field_transform. The remaining five entries are scalars.

    Raises
    ------
    ValueError
        When N or M fails to be a positive even integer, when M exceeds N, when the refinement factor fails to be an integer above zero, or when the transform does not have shape (2, N + 1, N + 1).
    """
    return
```

### Step 6

06_energy_norm_error

Goal
----
The quality of a computed field is measured against the exact field in the energy norm the problem itself supplies,

$$err = integral of (Eth - Enum) . c . (Eth - Enum) / integral of Eth . c . Eth,$$

with Eth the exact field of the real microstructure and Enum the computed one. Written that way the measure appears to need the exact field, which for this microstructure is an involved complex-variable construction. It does not. Expand the numerator,

$$integral of (Eth - Enum) . c . (Eth - Enum) = integral of Eth . c . Eth - 2 * integral of Eth . c . Enum + integral of Enum . c . Enum,$$

and treat the two ingredients separately. The exact current Jth = c * Eth is divergence free and periodic. The computed field is, by construction, a periodic gradient added to the prescribed mean: every retained non-zero mode of it lies along the direction the Green operator projects onto, and its mean mode is the loading. The Hill-Mandel lemma applies to that pairing and gives

$$average of Jth . Enum = average of Jth . average of Enum = cth * |Ebar|^2,$$

where cth is the exact effective conductivity, because the average of the computed field is the applied mean field exactly. The same lemma applied to the exact field against itself gives the first term as cth * |Ebar|^2 as well. Two of the three terms therefore collapse onto the same constant and

$$err = average of c * |Enum|^2 / (cth * |Ebar|^2) - 1.$$

Nothing about the exact field survives except the single number cth. Two things follow. The measure is computable from the numerical field alone once cth is known, and it is non-negative for every admissible field, because the exact solution is the one that minimises the energy over all periodic gradients with the prescribed mean; it vanishes only when the computed field is the exact one. A negative value is therefore not a small numerical accident but a sign that something in the chain is wrong.

The conductivity appearing in that average is the conductivity of the real material. Its interface sits exactly at half the cell edge and its area fraction is exactly one quarter, whatever the grid did to it. Using the sampled conductivity instead turns the average into the discrete energy of the fixed point, which equals the computed macroscopic conductivity identically, and the measure then collapses to a comparison of two effective conductivities that carries none of the information about the field. That substitution is the single most consequential error available here and it is silent: it produces a smooth, plausible curve with a different shape.

The average can be formed exactly rather than by quadrature. The computed field retains the modes with index of magnitude at most M/2 in each direction, so the square of its magnitude reaches index M in each direction, and its coefficients W are recovered without error by transforming it on any grid long enough to represent every index from minus M to M separately. An odd grid of 2M + 1 points already does that; an even grid needs 2M + 2, because an even-length transform reaches one index further in the negative direction than the positive. The conductivity is piecewise constant, c1 plus (c2 - c1) times the indicator of the quarter square, and the integral of a single exponential over that square factorises into

$$g(p) = integral from 0 to 1/2 of exp(2 * pi * i * p * u) du,$$

which is 1/2 at p = 0, zero at every other even p, and i / (pi * p) at odd p. The average is then c1 times W at the mean index plus (c2 - c1) times the sum of W times g(p) times g(q) over all indices, with no quadrature error at any stage and therefore no refinement study to perform.

The one number that must come from outside is cth. For this microstructure, a square of side half the cell edge repeated on a square lattice, it is known in closed form, and it is not any mean-field estimate: it is the exact effective conductivity of a boundary value problem solved by complex-variable methods, and it can be recognised by two properties. Exchanging the two phase conductivities multiplies it by the same construction applied to the exchanged pair to give exactly c1 times c2, and in the limit of a perfectly conducting square it approaches a finite multiple of c1 rather than diverging, because a square at quarter area fraction does not percolate.

```python
def energy_norm_error(
    fine_field,
    n_modes: int,
    c_matrix: float,
    c_inclusion: float,
    mean_field,
) -> dict:
    """Integrate the computed field against the true microstructure and turn the result into the energy-norm error.

    Parameters
    ----------
    fine_field : array
        Reconstructed field of shape (2, nf, nf) in volt per metre.
    n_modes : int
        Truncation order M, even and above zero.
    c_matrix : float
        Conductivity of the surrounding phase in siemens per metre, above zero.
    c_inclusion : float
        Conductivity of the embedded phase in siemens per metre, above zero.
    mean_field : sequence
        Two components of the applied mean electric field in volt per metre.

    Returns
    -------
    dict
        Under the keys cell_energy, mean_square_field, inclusion_integral, exact_macroscopic, apparent_conductivity and energy_norm_error.

    Raises
    ------
    ValueError
        When M fails to be a positive even integer, when the field is not a real array of shape (2, nf, nf) whose length cannot represent every index from minus M to M, when either conductivity fails to be finite and above zero, or when the mean field is not two finite components of which at least one is non-zero.
    """
    return
```

### Step 7

07_generalised_green_operator

Goal
----
The Green operator used so far follows from the spectral derivative rule, in which differentiating an exponential multiplies it by its own wavevector. That is not the only rule available. Replacing the derivative by a difference quotient taken over a finite spacing h gives a different operator, and because the fields here are continuous functions rather than tables of nodal values the spacing need not be the grid spacing: it is a free parameter, and letting it run recovers a one-parameter family that contains the spectral rule at one end and the familiar finite-difference construction at the other.

Take the potential forward and the current backward, which is the pairing that makes the resulting operator symmetric,

$$
forward: [phi(x + h) - phi(x)] / h, backward: [J(x) - J(x - h)] / h,
$$

each applied in its own direction with its own spacing. Acting on the exponential of index i these multiply it by

$$
[exp(i * xi_i * h) - 1] / h and [1 - exp(-i * xi_i * h)] / h
$$

respectively. Collecting the second derivative that the forward rule applied to the potential and then contracted with the backward rule produces gives a real, non-negative quantity per direction,

$$
alpha_i = 2 * sin(xi_i * h / 2) / h,
$$

while the first-derivative factor that contracts with the polarisation is the complex

$$
beta_i = [1 - exp(-i * xi_i * h)] / (i * h).
$$

The two are not independent: the modulus squared of beta in each direction is exactly alpha squared in that direction, which is worth checking numerically because it is the identity that lets the operator be written compactly. With it the generalised operator is

$$
Gamma_G(xi) = conj(beta) tensor beta / [c0 * conj(beta) . beta],
$$

the denominator being c0 times the sum of the alpha squared over the directions, a real positive number. The denominator carries that sum to the first power and not to the second. Any expression that squares it is not an operator of the right physical dimension, and the error is easy to make. What it does is not subtle: the squared denominator divides the operator by a quantity that is tiny in these units and varies by more than three orders of magnitude across the retained modes, which suppresses the update altogether, so the field collapses to the uniform applied mean, the macroscopic conductivity collapses onto the arithmetic mean of the sampled conductivity, and the equilibrium residual never falls below any useful tolerance.

Two limits fix the family and both should be checked rather than assumed. As the spacing goes to zero, beta tends to the wavevector, alpha tends to the wavevector as well, and the operator tends to the spectral one; a candidate expression that fails this limit is wrong however plausible it looks, and this is the cheapest available test of the assembly. When the spacing equals the grid spacing, the operator is the one a finite-difference discretisation on that grid produces, with the difference that here the field remains a trigonometric polynomial defined everywhere rather than a table of nodal values. Between the two the spacing interpolates continuously.

The structural property survives the generalisation: c0 times the operator is still an orthogonal projector, idempotent and of unit trace, now onto the direction of the conjugated symbol rather than onto the wavevector. What changes is which direction that is, and therefore which fields the iteration is allowed to produce. Whether moving that direction removes the oscillations that a truncated series shows near a material discontinuity is a separate question, and the answer is not settled by the fact that the operator is a projector.

```python
def generalised_green_operator(
    wavevector,
    spacing: float,
    reference: float,
    retained,
) -> dict:
    """Build the difference-quotient Green operator at an arbitrary spacing and verify its structure.

    Parameters
    ----------
    wavevector : array
        Wavevector components of shape (2, n, n) in reciprocal metre.
    spacing : float
        Difference spacing in metre, above zero.
    reference : float
        Reference conductivity in siemens per metre, above zero.
    retained : array
        Boolean mask of shape (n, n) marking the retained non-zero modes.

    Returns
    -------
    dict
        Under the keys symbol, alpha, green, symbol_defect, spectral_defect, projector_defect and trace_defect. symbol is a complex array and alpha a real array, both of the shape (2, n, n) of the supplied wavevector, and green is a complex array of shape (2, 2, n, n) laid out as the spectral Green operator is. The remaining four entries are scalars.

    Raises
    ------
    ValueError
        When the wavevector is not a real array of shape (2, n, n), when the mask does not have shape (n, n) or marks no mode, or when the spacing or the reference conductivity fails to be finite and above zero.
    """
    return
```

### Step 8

08_optimal_truncation_ratio

Goal
----
Everything assembled so far exists to answer one question: for a fixed computational grid, how many Fourier modes should the represented field actually keep. The habitual answer is all of them, M = N, and it is a habit rather than a result. The grid and the truncation do different jobs. More grid points improve the trapezoidal estimate of each Fourier coefficient and nothing else, so their benefit saturates. More modes let the represented field resolve the material interface more sharply, which helps, but they also admit the coefficients whose trapezoidal estimates are worst, because the quadrature error of a coefficient grows with its frequency index, and near a discontinuity those same high modes are the ones that ring. Two effects of opposite sign in the same parameter produce an interior optimum, and where it falls is a property of the microstructure and the contrast rather than of the algorithm.

This stage runs the whole chain once for each candidate truncation on a stated list, forms the energy-norm error of each converged field against the true microstructure, and reports the truncation that minimises it as a fraction of the grid size. The list is stated rather than searched so that the answer is a finite deterministic computation and not the outcome of an optimiser.

Two contrasts deserve to be reported alongside it, because each is a place where a plausible shortcut gives a different answer.

The first is the macroscopic conductivity. Over the sweep run here its relative error against the exact value falls monotonically as the truncation rises, so judged on the effective property alone the best truncation on this list is the largest one and the habit appears vindicated. The fall of the effective conductivity itself is a theorem rather than an observation, since the converged field minimises a fixed convex energy over spaces that grow with the truncation; that its error falls with it needs the sampled problem to overestimate the exact value as well, which holds here by measurement. It would be wrong to explain it by saying the mean mode cannot feel the high frequencies: the mean current is a convolution of the conductivity spectrum with the field spectrum, so changing the retained content does change it, and the package's own numbers show it changing. What can be said is that the quantity is a single spatial average, so the oscillations that dominate the local error enter it only through that average and are not weighted by their amplitude wherever they occur. Anyone who measures the discretisation by the effective property alone will therefore not see the optimum the local field has, and may conclude that the two parameters need not be separated at all.

The second is the generalised Green operator built on difference quotients. Changing the derivative rule changes which direction each mode of the field is projected onto, and it is a reasonable expectation that a rule with a built-in length scale would damp the ringing. Running the same sweep with it settles that by measurement rather than expectation. What that sweep reports is not the energy-norm error, and the distinction is not cosmetic. The identity that turns the energy-norm error into an energy needs the computed field to be a genuine periodic gradient plus the applied mean. The spectral rule guarantees that; a difference rule does not, because it projects each mode onto the conjugated difference symbol rather than onto the wavevector, and the resulting field carries a component of order one per cent outside the gradient subspace. The cross term in the expansion therefore no longer collapses onto the same constant, so the quantity reported for the difference rule is the excess of its apparent conductivity over the exact one and nothing more. It is a legitimate diagnostic, it is the quantity the two families are compared on, and it coincides with the energy-norm error only in the spectral case. It carries no variational certificate of non-negativity, and no conclusion about the true local-field error of the difference rule follows from it.

```python
def optimal_truncation_ratio(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    period: float,
    mean_field,
    n_grid: int,
    mode_step: int,
    refinement: int,
    spacing_ratio: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Sweep the candidate truncations and report the one that minimises the energy-norm error.

    Parameters
    ----------
    donor_density_matrix : float
        Ionised donor density of the surrounding phase in reciprocal cubic metre.
    mobility_matrix : float
        Electron mobility of the surrounding phase in square metre per volt second.
    donor_density_inclusion : float
        Ionised donor density of the embedded phase in reciprocal cubic metre.
    mobility_inclusion : float
        Electron mobility of the embedded phase in square metre per volt second.
    period : float
        Cell edge in metre, above zero.
    mean_field : sequence
        Two components of the applied mean electric field in volt per metre.
    n_grid : int
        The number N of grid intervals, even and above zero.
    mode_step : int
        Spacing of the candidate truncations, even, above zero and dividing N.
    refinement : int
        Whole refinement factor for the reconstruction, at least two.
    spacing_ratio : float
        Difference spacing of the generalised operator as a multiple of the grid spacing.
    tolerance : float
        Residual at which each iteration stops, above zero.
    max_iterations : int
        Largest number of sweeps permitted per candidate, above zero.

    Returns
    -------
    dict
        Under the keys optimal_ratio, optimal_modes, optimal_error, error_at_full, error_ratio, macroscopic_at_optimum, macroscopic_at_full, exact_macroscopic, macro_error_at_optimum, macro_error_at_full, c_matrix, c_inclusion, contrast, sampled_area_fraction, peak_ratio_at_optimum, peak_ratio_at_full, iterations_at_optimum, generalised_optimal_ratio, generalised_optimal_modes, generalised_optimal_excess, candidates and errors. candidates and errors are sequences of the same length, holding the candidate truncations in increasing order and the energy-norm error of each. Every other entry is a scalar.

    Raises
    ------
    ValueError
        When any physical argument fails to be finite and above zero, when N or the candidate spacing fails to be a positive even integer, when the spacing does not divide N, when the refinement factor is below two, or when any single solve fails to converge within max_iterations sweeps.
    """
    return
```
