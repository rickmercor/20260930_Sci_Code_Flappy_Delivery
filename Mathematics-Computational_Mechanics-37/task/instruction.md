# Mathematics-Computational_Mechanics-37

## Background

Embedding a body in a larger regular grid buys freedom from meshing at the cost of having to say what the grid holds outside the body. Immersed, fictitious-domain and finite-cell methods all fill that exterior with a phase intended to be mechanically absent. Intended is not the same as achieved, and whether the intention is met along the surface of the body is a question the construction does not settle on its own, which is why it is worth putting to a measurement.

Fourier-based micromechanical solvers meet the question in a sharp form, since every differential operator is applied as a multiplier on a transform and the cell must therefore close on itself. A specimen with real surfaces has to be seated inside a larger cell and the remainder filled, after which the same discrete specimen can be interrogated by two solvers built on the same grid, the same difference operator and the same material, one marching in time and one posing a harmonic eigenvalue problem. Whatever the two hold in common cancels from a comparison of what they return, which is what makes any residual difference between them attributable to the embedding rather than to the discretisation.

Reading a resonance out of a simulated record raises a second and quite separate question, one of resolution rather than of physics. The spectrum of a record of finite duration is sampled at the reciprocal of that duration, a spacing far coarser than the precision of the solver that produced the record, so a line has to be placed somewhere between bins before two readings of it can be compared at the resolution this task grades. Which curve is passed through which quantities, and what the resulting estimate is worth against simply naming the strongest bin, have to be established before the comparison means anything.

## Problem

An embedded-boundary solver buys its freedom from meshing by replacing the exterior of a body with a fictitious phase, and the bill for that convenience is that the fictitious phase is never quite inert: the body does not ring at the frequencies the same discretisation would give it if the exterior were truly absent. Your task is to measure that fictitious inertia, in hertz, for a free-standing polycrystalline titanium bar embedded in a periodic cell and solved by a Fourier-based elastodynamic scheme.

The measurement works by difference. The same discrete bar is interrogated twice on the same grid. Once in the frequency domain, where the fictitious phase is given no mass at all and the natural frequencies come out of a harmonic eigenvalue problem: that is the baseline, the bar as the discretisation would have it if the exterior weighed nothing. Once in the time domain, where the fictitious phase does carry mass, and the resonances have to be read off the spectrum of a receptor record after a short traction pulse. Every resonance in the second reading sits below its counterpart in the first. The mean of those signed differences, taken over every axial resonance of the specimen below a stated ceiling, is the number this task grades.

Let the bar run from $x_1 = 0$ to $x_1 = l$. Its displacement $u$ has three components but varies with $x_1$ and with time only; both transverse strains are held at zero; and no dissipation of any kind is present.

Use the following deterministic configuration:

- specimen length $l = 5$ mm along $x_1$, discretised into $129$ voxels, so that the voxel edge is $h = l/129$
- twelve grains occupying, in order from the struck face at $x_1 = 0$, 5, 11, 7, 15, 16, 18, 9, 16, 5, 6, 10 and 11 voxels
- every grain is a hexagonal titanium single crystal of density $\rho = 4506.3$ kg per cubic metre, with the sixfold axis along the third crystal axis and stiffness constants in gigapascal, in the Voigt ordering (11, 22, 33, 23, 13, 12), $c_{11} = c_{22} = 162.4$, $c_{33} = 180.7$, $c_{12} = 92.0$, $c_{13} = c_{23} = 69.0$, $c_{44} = c_{55} = 46.7$ and $c_{66} = 35.2$
- grain orientations as Bunge Euler angles $(\varphi_1, \Phi, \varphi_2)$ in degrees, grain by grain: (38.5, 37.9, 1.9), (311.8, 106.0, 249.3), (50.3, 112.1, 287.9), (158.3, 40.3, 350.5), (211.3, 150.3, 54.2), (114.9, 42.8, 90.7), (150.5, 49.0, 302.9), (319.3, 37.9, 33.1), (163.5, 151.7, 98.7), (194.1, 121.5, 162.6), (349.4, 56.6, 150.6) and (99.4, 97.3, 195.4), with the orientation matrix of a grain taken as the one whose columns are the crystal basis vectors expressed in the specimen frame
- each grain stands in for its crystal as an isotropic solid, taking for its Young modulus the stiffness the rotated crystal shows when pulled along $x_1$, and for its Poisson ratio the sideways contraction along $x_2$ that such a pull brings with it
- periodic cell of $141$ voxels holding the specimen at voxels $0$ to $128$ followed by $12$ padding voxels; voxel $0$ is adjacent to the struck face and voxel $128$ to the opposite face
- displacement is sampled at voxel centres and the material is uniform within a voxel, so the elastic link that joins two neighbouring centres runs through half of each; give every such link inside the specimen the series stiffness of those two halves
- the two links that join a specimen voxel to a padding voxel, one at each end of the bar because the cell is periodic, are prescribed to carry the modulus of the specimen voxel; links with padding on both sides carry the padding modulus
- padding for the march is weightless in stiffness, exactly zero, while its density matches the specimen; padding for the harmonic solver reverses this, taking zero density and isotropic moduli set at one part in ten million of the voxel mean of the matching specimen modulus
- spatial derivatives are evaluated in the Fourier domain, with the extension of a link taken as the forward first difference of the displacement over one voxel rather than as the continuous wavenumber, and the forces gathered back onto the voxels by the adjoint of that same difference
- marching in time by the implicit Newmark rule at the average-acceleration settings $\beta = 1/4$ and $\gamma = 1/2$, over $25000$ increments of $\Delta t = 2$ ns, starting from rest in displacement, velocity and acceleration alike, with the load of an increment sampled at the instant that increment ends and each linear system taken down to a relative residual of $10^{-10}$
- the face at $x_1 = 0$ is struck along $x_1$ by $T(t) = T_0 [\exp(-(t - t_0)^2 / (2 s^2)) - \exp(-(t - t_0 - s)^2 / (2 s^2))]$, where $T_0 = 1$ MPa, $s = 20$ ns and $t_0 = 80$ ns
- voxel $128$ acts as receptor; its displacement is written down at $t = 0$ and again after each increment, giving $25001$ samples
- those $25001$ samples of the three components are transformed in time, no window and no zero padding, and collapsed onto the non-negative bins to give the oscillation spectrum
- the set of modes measured is every axial natural frequency of the specimen below $4$ MHz, obtained from the harmonic form of the same discrete equations with the frequency-domain padding, the null mode of the family discarded
- for each of those frequencies the spectral line belonging to it is the bin of largest amplitude within $5$ per cent of it, and the position of that line is then refined below the bin spacing by fitting a parabola through the natural logarithms of the spectral amplitude at the winning bin and at the bin on either side of it, and taking the position of the maximum of that parabola
- the uniform bar used for comparison keeps the same density and takes the effective axial Young modulus of the grain chain, its Poisson ratio being the grain ratios weighted by length
- arithmetic throughout in IEEE float64, transforms under the default NumPy normalisation

Report the mean, over the axial resonances described above, of the refined line frequency minus the natural frequency, in kilohertz to five significant figures, with its sign. The answer is graded within $0.05$ kHz.

Your reasoning must also report eleven intermediate quantities: how many axial natural frequencies of the specimen fall below the ceiling, and how many natural frequencies of both families together fall below it, both as exact integers; the bin spacing of the spectrum in hertz to at least five significant figures; the refined position of the lowest line, in bins relative to its winning bin, to at least three significant figures, together with the value of $a - 2b + c$ for that line, where $a$, $b$ and $c$ are the natural logarithms of the spectral amplitude at the bin below, at the winning bin and at the bin above; the mean of the signed differences divided by the mean of the natural frequencies, to at least four significant figures, and the fictitious mass hanging on each free face that this implies, expressed as a multiple of the mass of one specimen voxel; the signed difference belonging to the lowest of the axial resonances on its own, in kilohertz to five significant figures; the same mean as the graded answer but taken over the winning bin centres themselves, with the refinement below the bin spacing left out and everything else unchanged, also in kilohertz to five significant figures; and the Young modulus in gigapascal and the Poisson ratio of the equivalent homogeneous bar, each to at least five significant figures.

Your reasoning must also justify the conventions the configuration prescribes but does not explain: why the link joining two voxel centres takes the series stiffness of the two halves it runs through, and what an arithmetic mean or the modulus of one side alone would do instead; why the two solvers cannot share one padding, and why the time-domain padding is given the density it is given rather than a negligible one; how a traction applied to a face reaches a periodic solver and with what normalisation; why a one-voxel difference replaces the continuous wavenumber and how the gathering of forces must be paired with it; what the refinement of a line position below the bin spacing is worth against the raw bin centre; and what the number you report is physically, including why it is a property of the embedding rather than of the bar.

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

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_reduce_oriented_grains

Goal
----
A bar cut from a coarse-grained metal is, along its axis, a short queue of single crystals. Each crystal meets an axial wave in one direction only, so it can be replaced by an isotropic surrogate provided the surrogate answers two questions the same way the rotated lattice does: how stiff the crystal is when pulled along the bar, and how much it draws in sideways while that happens. Inverting the rotated stiffness answers both. With $S$ the rotated compliance in Voigt form, the surrogate takes $E = 1/S_{11}$ and $nu = -S_{12}/S_{11}$.

Hexagonal symmetry leaves five free stiffness entries, quoted here as $c_{11}$, $c_{33}$, $c_{12}$, $c_{13}$ and $c_{44}$ with the sixfold axis on the third crystal axis; the rest follow, $c_{22} = c_{11}$, $c_{23} = c_{13}$, $c_{55} = c_{44}$ and $c_{66} = (c_{11} - c_{12})/2$, in the Voigt ordering (11, 22, 33, 23, 13, 12). Three angles $(phi_1, Phi, phi_2)$ in degrees seat a crystal in the specimen frame through $R = Z(phi_1) X(Phi) Z(phi_2)$, with $Z(a)$ and $X(a)$ the right-handed rotations by $a$ about the third and first axes; the columns of $R$ are the crystal axes read in the specimen frame, and every index of the stiffness is carried by $R$.

The wave problem does not consume $E$ and $nu$ directly. Transverse strain is held at zero throughout this bar, which stiffens the axial response to the constrained modulus $M = lambda + 2 G = E (1 - nu)/[(1 + nu)(1 - 2 nu)]$ and leaves the two transverse responses on the shear modulus $G = E/[2 (1 + nu)]$.

The same step also collapses the queue into the uniform bar that a homogeneous interpretation of the spectrum would assume. Grains laid end to end carry a common axial stress, so their compliances add in proportion to bar_length and the effective axial modulus is the reciprocal of the bar_length-weighted mean of the reciprocal grain moduli. The effective Poisson ratio follows a different rule, a plain bar_length-weighted mean. Weights are voxel fractions. That uniform bar, free at both ends, has standing waves at $f_n = n c/(2 l)$ with $c$ the speed of the family in question, and the ladder built on the constrained modulus is returned as the reference against which the real queue is read.

```python
def reduce_oriented_grains(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_modes: int,
) -> dict:
    """Replace each oriented crystal by its axial surrogate and collapse the queue into one uniform bar.

    Parameters
    ----------
    euler_angles : np.ndarray
        One row per grain, shape (n_grains, 3), holding (phi1, Phi, phi2) in degrees.
    crystal_constants : dict
        The five hexagonal stiffness entries in pascal, under the keys c11, c33, c12, c13 and c44.
    grain_voxels : np.ndarray
        Voxels held by each grain, an integer array of shape (n_grains,).
    specimen_density : float
        Density of the specimen in kilogram per cubic metre.
    bar_length : float
        Length of the specimen in metre.
    n_modes : int
        How many rungs of the analytic ladder to return.

    Returns
    -------
    dict
        Under the keys young, poisson, longitudinal_modulus, shear_modulus, young_hom, poisson_hom, longitudinal_modulus_hom, shear_modulus_hom, longitudinal_speed, transverse_speed and analytic_longitudinal.

    Raises
    ------
    ValueError
        When euler_angles is not shaped (n_grains, 3) with at least one grain, when an orientation angle fails to be finite, when one of the keys c11, c33, c12, c13 and c44 is absent, when a constant fails to be finite and above zero, when the five entries do not give a positive definite Voigt stiffness, when grain_voxels is not a one-dimensional integer array of matching bar_length whose entries all reach one, when a surrogate Poisson ratio falls outside the open interval (-1, 1/2), when specimen_density or bar_length fails to be finite and above zero, or when n_modes is not an integer of one or more.
    """
    return
```

### Step 2

02_assemble_face_stiffness_cell

Goal
----
A Fourier solver knows only periodic cells, so a free-standing bar has to be embedded in one. Lay the specimen down first, one voxel per grid point, each grain taking its prescribed run of voxels, then append a block of padding voxels whose job is to stand in for the emptiness outside the specimen. Closing the cell periodically puts that padding between the far end of the bar and its struck end. The cell is required to hold an odd number of voxels, which keeps the integer frequency set of the next stage symmetric about zero and free of a Nyquist entry.

Stiffness in this discretisation does not belong to a voxel. Displacement is sampled at voxel centres and the material is uniform within a voxel, so the elastic link that matters is the one joining two neighbouring centres, and that link runs through half of one voxel and half of the other. Two springs in series: each half contributes a compliance $h/(2 M)$ with its own modulus, the compliances add, and the link therefore carries

$$M_{link} = 2 M_{j} M_{j+1} / (M_{j} + M_{j+1}),$$

the harmonic mean of the two. Inside a grain the two are equal and the rule returns the grain value unchanged, so the rule bites only where two grains meet. Arithmetic averaging, or simply handing the link the modulus of the voxel on one side, both overstate the stiffness of a mismatched pair and both move every resonance of the bar.

The padding is not a material and the series rule does not apply to it. Two conventions are imposed instead. A link with a specimen voxel on one side and a padding voxel on the other carries the modulus of the specimen voxel, so that the padding hangs on a spring of the specimen's own stiffness; there are two such links, one at each end of the bar, because the cell is periodic. A link with padding on both sides carries the padding modulus.

The padding properties themselves are quoted as multiples of specimen averages: each padding modulus is a factor times the voxel mean of the matching specimen modulus, and the padding density is a factor times the specimen density, so that a factor of zero produces padding that is exactly stress free or exactly massless. Three rows come back because the three displacement components separate into one axial and two transverse scalar problems, row one carrying the constrained modulus and rows two and three the shear modulus.

```python
def assemble_face_stiffness_cell(
    grain_voxels: np.ndarray,
    longitudinal_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    specimen_density: float,
    n_pad: int,
    pad_stiffness_factor: float,
    pad_density_factor: float,
) -> dict:
    """Lay the specimen and its padding out, and give every link between neighbouring centres its modulus.

    Parameters
    ----------
    grain_voxels : np.ndarray
        Voxels held by each grain, an integer array of shape (n_grains,), none of them below one.
    longitudinal_modulus : np.ndarray
        Constrained longitudinal modulus of each grain in pascal, shape (n_grains,).
    shear_modulus : np.ndarray
        Shear modulus of each grain in pascal, shape (n_grains,).
    specimen_density : float
        Density of the specimen in kilogram per cubic metre.
    n_pad : int
        How many padding voxels follow the specimen, one or more.
    pad_stiffness_factor : float
        Padding modulus as a multiple of the specimen mean modulus, not negative.
    pad_density_factor : float
        Padding density as a multiple of the specimen density, not negative.

    Returns
    -------
    dict
        Under the keys n_specimen, n_total, density, voxel_modulus and face_modulus.

    Raises
    ------
    ValueError
        When grain_voxels is not a one-dimensional integer array whose entries all reach one, when either modulus array fails to match its length or holds an entry that is not finite and above zero, when specimen_density fails to be finite and above zero, when n_pad is not an integer of one or more, when a factor is not finite or drops below zero, or when the voxels total an even number.
    """
    return
```

### Step 3

03_solve_padded_eigenproblem

Goal
----
Before any pulse is fired, the padded cell can be asked directly what it rings at. Seek displacements that oscillate at a single angular frequency and the momentum balance turns into a generalised eigenvalue problem: the discrete stiffness operator on one side, the voxel voxel_density on the other, and the squared angular frequencies as the eigenvalues. Both operators are symmetric, so those eigenvalues come out real and non-negative. The answers are the baseline of this task, the frequencies against which the pulsed reading will later be compared.

The stiffness operator is built from the link moduli of the previous stage. Its action is applied in the transformed domain: multiply the transform of the displacement by the symbol of the one-voxel forward difference to get the extension of every link, weight each link by its own modulus to get the force it carries, and multiply by the conjugate symbol to gather those forces back onto the voxels. Pairing a difference with its own adjoint in this way is what makes the operator symmetric and positive semi-definite. An odd voxel count keeps the integer frequency set symmetric about zero so that no frequency lands on Nyquist.

Padding suited to a time march will not serve here. Stiffness that vanished on the padding would leave the pencil singular over every padding voxel, so the padding used for this stage is very compliant instead and carries no mass at all. Being massless is what makes it removable: padding voxels contribute no inertia, so they can be condensed out of the stiffness operator exactly, leaving a reduced pencil over the specimen voxels alone whose mass matrix is positive definite.

One mode of each family is then discarded. A constant displacement is annihilated by the forward difference, so it is an exact null vector of the stiffness operator over the whole cell, and the reduced operator inherits it exactly; whatever small frequency the solver returns for that mode is numerical dust. What remains, in ascending order, are the natural frequencies of the axial family and of the transverse family. One transverse family is enough, since both transverse components ride on the same modulus row.

```python
def solve_padded_eigenproblem(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    n_specimen: int,
    n_modes: int,
) -> dict:
    """Read the natural frequencies of the embedded specimen off the padded harmonic pencil.

    Parameters
    ----------
    face_modulus : np.ndarray
        Modulus of every link in pascal, one row per component, shape (3, n_total).
    voxel_density : np.ndarray
        Voxel voxel_density, shape (n_total,), above zero across the specimen and exactly zero across the padding.
    voxel_edge : float
        Edge of one voxel in metre.
    n_specimen : int
        How many voxels at the head of the cell belong to the specimen.
    n_modes : int
        How many frequencies to list for each family.

    Returns
    -------
    dict
        Under the keys longitudinal and transverse, float64 arrays of shape (n_modes,) in hertz.

    Raises
    ------
    ValueError
        When face_modulus is not shaped (3, n_total), when voxel_density is not shaped (n_total,), when n_total comes out even, when voxel_edge fails to sit above zero, when n_specimen is not an integer between one and n_total minus one, when the voxel_density fails to stay above zero across the specimen or to vanish across the padding, when a link beyond the specimen fails to stay above zero, or when n_modes is not an integer between one and n_specimen minus one.
    """
    return
```

### Step 4

04_invert_implicit_step

Goal
----
Every step of an implicit time march ends in the same linear system, and this stage both states that system and solves it. The operator is the voxel voxel_density on the diagonal plus a coefficient times the discrete stiffness operator of the padded cell, so a field $u$ is carried to $d u + c Q(u)$: $d$ is the voxel voxel_density, $c$ gathers the Newmark parameter with the square of the time step, and $Q(u)$ is the force the extended links exert back on the voxels. $Q$ is evaluated through discrete Fourier transforms, never assembled: extend every link with the symbol of the one-voxel forward difference, weight each link by its own modulus, and gather with the conjugate symbol, which pairs the difference with its own adjoint and leaves $Q$ symmetric and positive semi-definite.

Positive voxel_density everywhere makes the whole operator real, symmetric and positive definite, which is the setting the conjugate gradient method was written for, and each iteration reaches the operator only by applying it to the current search direction.

A preconditioner is needed because the padding leaves an enormous stiffness contrast. Take the same operator for a uniform reference medium carrying the volume-averaged link modulus and the volume-averaged voxel_density of the entire cell, padding included, and invert it exactly; that inverse is diagonal in the transformed domain, its multiplier at a given frequency being the reciprocal of the averaged voxel_density plus the coefficient times the averaged modulus times the squared magnitude of the difference symbol, applied to each component with the average of its own row. Since the preconditioned right-hand side already solves a uniform problem exactly, it makes the natural starting iterate. Iteration stops when the Euclidean norm of the residual across the three components drops below the residual_tolerance times the norm of the right-hand side.

Exhausting the iteration cap with the residual still above that threshold is a failure, not an answer. An unconverged displacement would be swallowed by the time integrator and would contaminate everything after it, so the routine raises a RuntimeError rather than handing back whichever iterate it happens to hold. The image of the accepted solution under the operator is returned alongside it, so that the residual can be checked without rebuilding the operator.

```python
def invert_implicit_step(
    rhs: np.ndarray,
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    newmark_coefficient: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Build the implicit elastodynamic operator and invert it by preconditioned conjugate gradients.

    Parameters
    ----------
    rhs : np.ndarray
        Right-hand side of the system, shape (3, n_total).
    face_modulus : np.ndarray
        Modulus of every link in pascal, one row per component, shape (3, n_total).
    voxel_density : np.ndarray
        Voxel voxel_density, shape (n_total,), above zero at every voxel.
    voxel_edge : float
        Edge of one voxel in metre.
    newmark_coefficient : float
        The Newmark parameter multiplied by the squared time step, not negative.
    residual_tolerance : float
        Relative residual at which the iteration is allowed to stop.
    iteration_ceiling : int
        Ceiling on the iteration count.

    Returns
    -------
    dict
        Under the keys solution, operator_image, iterations and residual.

    Raises
    ------
    ValueError
        When rhs or face_modulus is not shaped (3, n_total), when voxel_density is not shaped (n_total,) or fails to stay above zero, when n_total comes out even, when voxel_edge or residual_tolerance fails to sit above zero, when newmark_coefficient drops below zero, or when iteration_ceiling is not an integer of one or more.
    RuntimeError
        When the iteration ceiling is reached while the relative residual is still above the residual_tolerance.
    """
    return
```

### Step 5

05_march_pulsed_bar

Goal
----
With the baseline frequencies already in hand, the same cell is now struck and watched. A brief traction pulse is applied to one end face of the bar and the displacement of a single interior point is written down for a long time. A periodic solver admits no boundary condition, so the traction cannot be prescribed at a face; it enters as a singular body force, the traction vector multiplied by the discrete surface delta function of the loaded face. On the grid that delta is the reciprocal of the voxel size on the one specimen voxel layer adjacent to the face and vanishes everywhere else. The excitation reaches the equations nowhere else.

The pulse is two Gaussians of equal width, their centres one width apart, the later subtracted from the earlier. That construction gives it zero time integral, so the bar picks up no appreciable net velocity, and it spreads energy across a band wide enough to set every resonance of interest ringing.

The march is the implicit Newmark scheme. One step runs as follows: form the predictor from the current displacement, velocity and acceleration; solve for the new displacement with the routine of the previous stage, the right-hand side being the Newmark coefficient times the body force plus the voxel_density times the predictor; recover the new acceleration as the new displacement minus the predictor, divided by the Newmark coefficient; and update the velocity from the old and new accelerations weighted by the second Newmark parameter. Displacement, velocity and acceleration all start at zero, the body force of a step is sampled at the end time of that step, and the receptor displacement is written down after each step, the state at time zero included. Should any implicit solve miss the requested relative residual the march must stop, because that displacement would enter the next predictor and spoil the whole record; the routine raises a RuntimeError instead.

```python
def march_pulsed_bar(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    time_step: float,
    n_steps: int,
    newmark_beta: float,
    newmark_gamma: float,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    struck_voxel: int,
    receptor_voxel: int,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Strike the embedded bar and write down the receptor displacement step by step.

    Parameters
    ----------
    face_modulus : np.ndarray
        Modulus of every link in pascal, one row per component, shape (3, n_total).
    voxel_density : np.ndarray
        Voxel voxel_density, shape (n_total,), above zero at every voxel.
    voxel_edge : float
        Edge of one voxel in metre.
    time_step : float
        Length of one time step in second.
    n_steps : int
        How many time steps to take.
    newmark_beta : float
        The first Newmark parameter, somewhere in (0, 1].
    newmark_gamma : float
        The second Newmark parameter, somewhere in (0, 1].
    traction_amplitude : float
        Height of the traction pulse in pascal.
    gaussian_width : float
        Width of one Gaussian in second.
    gaussian_centre : float
        Where the earlier Gaussian is centred, in second.
    traction_axis : np.ndarray
        Aim of the traction, a non-zero vector of three entries, normalised inside the routine.
    struck_voxel : int
        The specimen voxel that touches the struck face.
    receptor_voxel : int
        The voxel whose displacement is written down.
    residual_tolerance : float
        Relative residual demanded of the linear solver.
    iteration_ceiling : int
        Ceiling on the linear solver iteration count.

    Returns
    -------
    dict
        Under the keys record, displacement, velocity, acceleration and mean_iterations.
        record : np.ndarray of shape (n_steps + 1, 3), float64. Row t holds the three
        displacement components of the receptor voxel after t steps; row 0 is the rest state.
        displacement, velocity, acceleration : np.ndarray of shape (3, n_total), float64, the
        state of every voxel after the last step, one row per component like face_modulus.
        mean_iterations : float, linear-solver iterations averaged over the steps.

    Raises
    ------
    ValueError
        When face_modulus is not shaped (3, n_total), when voxel_density is not shaped (n_total,) or fails to stay above zero, when n_total comes out even, when voxel_edge, time_step or gaussian_width fails to sit above zero, when n_steps is not an integer of one or more, when a Newmark parameter falls outside (0, 1], when traction_axis is not a non-zero vector of three entries, or when struck_voxel or receptor_voxel is not an integer addressing a voxel of the cell.
    RuntimeError
        When the implicit solve of a step misses the requested relative residual.
    """
    return
```

### Step 6

06_fit_subbin_resonance_lines

Goal
----
A receptor_record of finite length cannot say where a resonance is to better than the spacing of its frequency bins, and that spacing is coarse next to the accuracy the solver itself achieves. This stage closes that gap. It is the measuring instrument of the task: everything before it produces a receptor_record, and everything after it works with the numbers this stage reads off.

The three displacement components are transformed in time and collapsed into one curve by taking the root of the summed squared moduli, which puts every family on the same axis. Only the non-negative half of the frequency axis is kept, its bins spaced at the reciprocal of the receptor_record duration, with neither windowing nor zero padding applied.

Each resonance is then found twice over. First coarsely: given a target frequency, the line belonging to it is the bin of largest amplitude within a stated fractional window of that target, which is wide enough to hold a line that the discretisation has moved and narrow enough never to reach its neighbours. Second, and this is the point of the stage, finely: the bin index alone is a quantised estimate, so the position is sharpened from the shape of the peak. Take the natural logarithms of the amplitude at the winning bin and at its two immediate neighbours and pass a parabola through those three values. Writing $a$, $b$ and $c$ for those logarithms in order, the vertex of that parabola sits at

$$delta = (a - c) / [2 (a - 2 b + c)]$$

bins from the winning bin, and the sharpened frequency is $(k + delta)$ times the bin spacing with $k$ the winning bin index. The denominator is the discrete curvature of the log amplitude, which must be strictly negative for the parabola to have a maximum at all; a positive or vanishing curvature means the three amplitudes do not describe a peak and the sharpening is refused. A parabola through logarithms rather than through the amplitudes themselves is chosen because a resonance line in an unwindowed transform is close to Gaussian in shape near its crown, and a Gaussian is exactly a parabola once the logarithm is taken.

The refinement is defined only when the winning bin has a neighbour on each side and all three amplitudes are strictly positive.

```python
def fit_subbin_resonance_lines(
    receptor_record: np.ndarray,
    time_step: float,
    target_frequencies: np.ndarray,
    search_fraction: float,
) -> dict:
    """Collapse the receptor_record into one spectrum and sharpen the line belonging to each target frequency.

    Parameters
    ----------
    receptor_record : np.ndarray
        Receptor displacement samples, shape (n_samples, 3).
    time_step : float
        Spacing between samples in second.
    target_frequencies : np.ndarray
        Frequencies in hertz near which lines are sought, one dimension, every entry above zero.
    search_fraction : float
        Half-width of the search window as a fraction of the target, above zero and below one.

    Returns
    -------
    dict
        Under the keys frequencies, intensity, bin_spacing, line_index, line_offset, line_curvature, line_intensity and refined_frequency.

    Raises
    ------
    ValueError
        When receptor_record is not shaped (n_samples, 3) with two samples or more, when a recorded value fails to be finite, when time_step fails to sit above zero, when target_frequencies is not a one-dimensional array of at least one finite entry above zero, when search_fraction leaves the open interval (0, 1), when a search window admits no frequency bin, or when a winning bin cannot be sharpened, whether because it sits at an end of the spectrum, because one of the three amplitudes fails to sit above zero, or because the curvature of the log amplitude fails to be negative.
    """
    return
```

### Step 7

07_report_boundary_inertia_gap

Goal
----
This stage runs the whole measurement, from the raw configuration to the number the task grades. Stage 1 turns the orientation list into axial surrogates and collapses the queue into the uniform bar a homogeneous reading would assume. Stage 2 lays the specimen out inside the periodic cell twice over, once with the padding the harmonic solver needs, very compliant and massless, and once with the padding the march needs, stress free and carrying the specimen specimen_density, and gives every link its modulus. Stage 3 solves the harmonic pencil and fixes the baseline frequencies. Stage 4 supplies the implicit operator and its preconditioned inverse, which stage 5 uses to march the struck bar forward and write down the receptor displacement. Stage 6 collapses that record into a spectrum and sharpens the line belonging to each baseline frequency below the ceiling.

What is graded is the mean, over every axial resonance of the specimen below the ceiling, of the signed difference between the sharpened line and the baseline frequency. The two padding constructions differ in exactly one respect that the specimen can feel: the padding used by the march carries mass, and the link joining it to the specimen carries the specimen's own stiffness, so one voxel of fictitious mass hangs on each free face of the bar while the march runs. That mass is absent from the harmonic baseline. The mean difference is therefore a measurement of the fictitious inertia the embedding adds at the boundary, and averaging it over the whole set of resonances below the ceiling, rather than reading one mode, is what makes it a property of the specimen rather than of a mode.

The pulse strikes the first specimen voxel along the bar axis and the receptor sits on the last specimen voxel, so the two are the voxels adjacent to the two end faces. The Newmark parameters are one quarter and one half. Padding for the march is built with a stiffness factor of zero and a specimen_density factor of one; padding for the harmonic solver with a stiffness factor of one part in ten million and a specimen_density factor of zero.

Alongside the graded mean the stage returns what verifies it: how many axial resonances the ceiling admits, how many resonances of both families lie below it, the mean relative shift against twice the ratio of voxel size to specimen bar_length, the fictitious mass per face in units of one voxel of specimen mass, the sharpened offset and the log curvature of the lowest line, the lowest transverse natural frequency, the homogenised constants and the analytic ladder they imply.

```python
def report_boundary_inertia_gap(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_pad: int,
    time_step: float,
    n_steps: int,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    ceiling: float,
    search_fraction: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Measure the fictitious inertia that the periodic embedding hangs on the ends of the bar.

    Parameters
    ----------
    euler_angles : np.ndarray
        Orientation angles in degrees, shape (n_grains, 3).
    crystal_constants : dict
        The five hexagonal entries in pascal, under the keys c11, c33, c12, c13 and c44.
    grain_voxels : np.ndarray
        Voxels held by each grain, an integer array of shape (n_grains,).
    specimen_density : float
        Density of the specimen in kilogram per cubic metre.
    bar_length : float
        Length of the specimen in metre.
    n_pad : int
        How many padding voxels follow the specimen.
    time_step : float
        Length of one time step in second.
    n_steps : int
        How many time steps to take.
    traction_amplitude : float
        Height of the traction pulse in pascal.
    gaussian_width : float
        Width of one Gaussian in second.
    gaussian_centre : float
        Where the earlier Gaussian is centred, in second.
    traction_axis : np.ndarray
        Aim of the traction, a non-zero vector of three entries.
    ceiling : float
        Frequency in hertz below which the axial resonances are collected.
    search_fraction : float
        Half-width of the line search window as a fraction of the target frequency.
    residual_tolerance : float
        Relative residual demanded of the linear solver.
    iteration_ceiling : int
        Ceiling on the linear solver iteration count.

    Returns
    -------
    dict
        Under the keys mean_gap, mode_count, resonance_count, eigen_longitudinal, refined_line, gap, mean_relative_shift, voxel_mass_fraction, face_mass_ratio, first_offset, first_curvature, bin_spacing, first_transverse, analytic_first_longitudinal, young_hom, poisson_hom, longitudinal_speed and mean_cg_iterations.

    Raises
    ------
    ValueError
        When ceiling fails to sit above zero, when no axial resonance falls below it, when the voxels total an even number, or when an argument breaks a condition of one of the earlier stages, whose rejections travel through unchanged.
    RuntimeError
        When an implicit solve of the march misses the requested relative residual.
    """
    return
```
