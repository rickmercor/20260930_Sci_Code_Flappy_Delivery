# Physics-Condensed_Matter_Physics-11

## Background

A medium whose properties travel as a wave has a preferred direction, and wave propagation through it is not reciprocal. When the travelling modulation is slower than the waves the medium supports, the interaction is the familiar one: the modulation acts as a moving grating, adjacent branches of the dispersion relation repel where they cross, and the gaps that open are gaps in frequency, inside which a real wavenumber is impossible and disturbances decay away from the region that launched them. When the modulation outruns the medium the character of the interaction changes, because the crossings that the modulation connects are no longer between branches of the same sign of energy, and what opens is a separation between adjacent wavenumber bands instead.

Most of what is known about that supersonic case has been learned from media that are unbounded in space and switched in time, where the natural question is posed the other way round, with a real wavenumber prescribed and an eigenfrequency sought, and where the answer is a complex frequency and a disturbance that grows everywhere at once. A segment of finite length embedded in a uniform rod poses the complementary question. Its interfaces sit in space rather than in time, and a steady incident tone fixes the frequency rather than the wavenumber, so the question it answers is how a disturbance varies along the rod rather than how it varies in time. What such a calculation settles, and what it leaves open, is one of the things this task asks you to state.

That difference is not merely presentational. It determines which quantity is the eigenvalue and which the parameter, how many modes the truncated problem supplies, how the source method labels and selects the modes used in its interface system, and how many orders the two interfaces can actually match once the modulus has carried its own harmonic content into the stress condition. None of these is settled by the physical description of the segment, and each has to be decided before a number can be produced.

## Problem

A stiffness modulation that travels faster than the medium can carry sound does not merely scatter a wave, it pumps it, and because the modulation has a direction the pumping is not the same for a wave running with it as for a wave running against it. Your task is to measure that asymmetry for a finite modulated segment cut into an otherwise uniform elastic rod, as a single dimensionless number: the ratio of the total power gain the segment delivers to a wave incident along the modulation direction, to the total power gain it delivers to a wave of the same frequency incident against it.

The rod carries longitudinal waves and is governed by $\partial_x [ E(x,t) \partial_x u ] - \rho_0 \partial_t^2 u = 0$ with constant density. Outside the segment the Young modulus is the constant $E_0$. Inside it the modulus travels as $E(x,t) = E_0 [ 1 + \alpha_m \cos(\omega_m t - \kappa_m x) ]$. A unit-amplitude wave of angular frequency $\omega_0$ arrives from one side; because the medium varies in time the scattered field is not monochromatic but carries orders at $\omega_0 + n \omega_m$, each of which leaves the segment as a propagating wave in the uniform rod, and the power the segment returns is therefore spread across those orders and need not sum to the power it received.

Use the following deterministic configuration.

- density $\rho_0 = 1$ kg per cubic metre and exterior modulus $E_0 = 1$ Pa, so the unmodulated wave speed is $c_0 = 1$ m per second
- modulation wavenumber $\kappa_m = 10$ rad per metre and modulation angular frequency $\omega_m = 20$ rad per second, so the modulation travels at twice $c_0$
- normalised modulation depth $\alpha_m = 0.3$
- the modulated segment occupies $x_0 \le x \le x_1$ with $x_0 = 0$ and a length $\Delta x = 2.5 \lambda_m$, where $\lambda_m = 2\pi / \kappa_m$ is the spatial period of the modulation, so that the modulation does not present the same phase at the two interfaces
- incident dimensionless frequency $\Omega_0 = \omega_0 / (c_0 \kappa_m) = 1.5$, taken to be the same for both directions of incidence
- the plane-wave expansion of the field is truncated at harmonic orders $|n| \le 8$, and the Fourier expansion of the modulus is exact at order $P = 1$ because the modulation law is a single cosine
- positive incidence means the incident wave travels in the same direction as the modulation and enters the segment at $x_0$; negative incidence means it travels in the opposite direction and enters at $x_1$
- the displacement and the axial stress $E(x,t) \partial_x u$ are both continuous at each of the two interfaces, at every retained order, with the modulus inside the segment carrying its own space-time dependence into the stress condition
- a scattering order of order $n$ leaves into the uniform rod with the signed wavenumber that keeps it travelling away from the segment, which for a down-converted order of negative frequency is not the wavenumber of the same magnitude with the opposite sign
- the power a scattering order carries in the uniform rod is to be referred to the power the incident wave carries, and the total gain of a direction of incidence is the sum of those referred powers over every transmitted and every reflected order retained
- arithmetic throughout in IEEE float64 with the default NumPy normalisations

Report the ratio of the total gain under positive incidence to the total gain under negative incidence, to five significant figures. The answer is graded within $0.002$.

Your reasoning must also report eleven intermediate quantities: how many wavenumber eigenvalues the modulated segment supports at a single prescribed frequency, and how many of them enter the interface matching, both as exact integers; the largest magnitude of the imaginary part of any of those eigenvalues, which settles whether any mode of the segment decays or grows along the rod; the dimensionless centre of the first forward wavenumber bandgap of the fundamental branch, to at least four significant figures, together with the same centre for the first backward bandgap; the zeroth-order transmission coefficient and the minus-first-order reflection coefficient under positive incidence in this configuration, each to at least five significant figures; the total gain under positive incidence and the total gain under negative incidence, each on its own to at least five significant figures; and the split of the positive-incidence gain into the part carried by the transmitted orders and the part carried by the reflected orders, each to at least four significant figures.

Your reasoning must also state what evidence you have that the reported number is converged in the harmonic truncation, giving the value of the graded ratio at more than one truncation order rather than adopting the prescribed one on the authority of this configuration alone.

Your reasoning must also validate the implementation against the published scattering coefficients for these same modulation parameters at a segment length of three modulation wavelengths, which is the configuration the literature on this system reports. Give your own computed zeroth-order transmission coefficient and minus-first-order reflection coefficient for that length, each alongside the published value it is being checked against.

Your reasoning must also justify the conventions the configuration prescribes but does not explain: why a spatially bounded problem forces the dispersion to be solved for the wavenumber at a prescribed real frequency rather than the other way round, what a real wavenumber spectrum does and does not establish about the segment, and what a separate calculation would be needed to settle; how the modes supported inside the segment are to be sorted into the source method's forward and backward group-velocity branch families, why an eigenvalue index will not do it, and why the sign of the time-averaged energy flux, although it measures physical energy-flow direction, does not reproduce those branch labels at exactly the frequencies this task grades; why the number of scattering orders that can be matched at an interface is smaller than the number of harmonics retained in the expansion, and what fixes the difference; and what it means physically that the two gains are unequal, including which feature of the modulation their inequality is attributable to.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short by leaving out whatever this task did not ask for, and report in full everything it did: every quantity named above, the working that determines each, and the justifications requested.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_modulation_coupling_matrix

Goal
----
The modulated segment carries an elastic modulus that travels as a wave, E(x,t) = E0 [1 + alpha_m cos(omega_m t - kappa_m x)]. Because that law is periodic in space and in time together, it is represented exactly by a Fourier series in the single phase omega_m t - kappa_m x, and because the law is one cosine the series closes at order one: the coefficient at order zero is E0, the coefficients at orders plus and minus one are E0 alpha_m / 2, and every higher coefficient vanishes. The truncation order P of the modulus expansion is therefore not a numerical choice but a property of the modulation law, and the value P = 1 is exact rather than approximate.

What the later stages need is not the coefficient list but the operator it induces. A field inside the segment is expanded in harmonics indexed by n, and multiplying that field by the modulus mixes neighbouring harmonics: the product at harmonic q draws on the field at harmonic q - p weighted by the coefficient at order p. Collecting those weights into a matrix whose entry in row q and column n is the coefficient at order q - n turns the multiplication into a matrix product. The matrix is banded with half-bandwidth P, symmetric because the cosine makes the coefficients at plus and minus p equal, and it reduces to E0 times the identity when the modulation depth is zero, which is the check that the convention has the right orientation.

The band structure is the reason the later matching cannot use every retained harmonic. A product that spreads harmonic content by P in each direction cannot be balanced at the outermost P orders of a window truncated at n_order, and that shortfall propagates into the number of scattering orders the interfaces can carry. This step fixes the bandwidth that causes it.

```python
def modulation_coupling_matrix(alpha_m: float, E0: float, n_order: int) -> dict:
    """Build the Fourier coefficients of the travelling modulus and the harmonic coupling operator they induce.

    Parameters
    ----------
    alpha_m : float
        Normalised modulation depth, finite and of magnitude below one.
    E0 : float
        Elastic modulus of the unmodulated rod in pascal, above zero.
    n_order : int
        Truncation order of the harmonic expansion, one or more.

    Returns
    -------
    dict
        Under the keys coefficients and coupling. The coefficients entry is a float64
        array of shape (3,) holding the modulus Fourier coefficients at orders minus
        one, zero and plus one. The coupling entry is a float64 array of shape
        (2 n_order + 1, 2 n_order + 1) whose entry in row q and column n is the
        modulus coefficient at order q - n.

    Raises
    ------
    ValueError
        If alpha_m is not finite or is not of magnitude below one, if E0 is not
        finite or not above zero, or if n_order is below one.
    """
    return
```

### Step 2

02_floquet_quadratic_operator

Goal
----
Inside the modulated segment the field is written in the generalised Floquet form, a carrier at the driving frequency multiplied by a series in the modulation phase. Harmonic n of that series then sits at frequency omega + n omega_m and at wavenumber kappa + n kappa_m, where kappa is the single unknown shared by the whole series. Substituting that form into the equation of motion and collecting the terms belonging to each harmonic leaves one algebraic equation per retained harmonic, and the set of them is the eigenvalue problem this stage assembles.

The structure follows from where the wavenumber enters. The axial stress is the modulus times the gradient, and taking the gradient of the field multiplies harmonic n by its own wavenumber; multiplying by the modulus then mixes neighbouring harmonics through the coupling operator of the previous step; and taking the gradient once more multiplies the result at harmonic q by the wavenumber of harmonic q. The wavenumber therefore appears twice, once on each side of the coupling operator, and the operator on the left of the harmonic balance is the product of a diagonal of shifted wavenumbers, the coupling matrix, and a second diagonal of shifted wavenumbers. Writing the shifted wavenumber of harmonic n as kappa plus n kappa_m and expanding that product in powers of kappa gives a matrix polynomial of degree two, whose coefficient at the square of kappa is the coupling matrix itself, whose coefficient at the first power is the sum of the coupling matrix with the order diagonal on either side, and whose constant term carries the order diagonal on both sides less the inertia.

Two features of the result matter later. The polynomial is quadratic in the wavenumber because the wavenumber enters the harmonic balance twice, once on each side of the coupling operator. Posing the same medium the other way round, with the wavenumber prescribed and the frequency sought, does not give a simpler problem: the inertia term carries the shifted frequencies, and writing that diagonal as omega times the identity plus omega_m times the order diagonal leaves a polynomial that is quadratic in omega and whose coefficient of the first power of omega is minus twice rho0 omega_m times the order diagonal, which does not vanish. The two postures are different problems rather than one problem in two guises, and which is appropriate is decided by the physics of the bounded segment, whose spatial interfaces prescribe frequency. And every coefficient matrix is real and symmetric, the coupling matrix by construction and the order diagonals because they commute with transposition in the combination that appears here, which is what later permits the group velocity to be obtained from the same eigenvector on both sides without a separate left eigenproblem.

The sign of kappa_m is not restricted. Reversing it reverses the direction in which the modulus travels, which is exactly how the second direction of incidence is posed.

```python
def floquet_quadratic_operator(
    omega: float,
    coupling: np.ndarray,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    """Assemble the quadratic matrix polynomial in the wavenumber for the modulated segment at a prescribed frequency.

    Parameters
    ----------
    omega : float
        Driving angular frequency in radians per second, finite and above zero.
    coupling : np.ndarray
        Harmonic coupling operator of the modulus, shape (2 n_order + 1, 2 n_order + 1).
    rho0 : float
        Mass density in kilogram per cubic metre, above zero.
    kappa_m : float
        Modulation wavenumber in radians per metre, finite and not zero.
    omega_m : float
        Modulation angular frequency in radians per second, finite and not zero.

    Returns
    -------
    dict
        Under the keys a2, a1 and a0, each a float64 array of shape
        (2 n_order + 1, 2 n_order + 1), the coefficient matrices of the quadratic
        matrix polynomial in the wavenumber from the square term down to the
        constant term.

    Raises
    ------
    ValueError
        If coupling is not a square array of odd side length or is not finite, or if
        omega, rho0, kappa_m or omega_m is not finite or violates its stated sign.
    """
    return
```

### Step 3

03_floquet_wavenumber_spectrum

Goal
----
A quadratic matrix polynomial has no direct eigensolver, so it is reduced to a linear one. The standard device introduces the product of the wavenumber with the eigenvector as a second unknown, which doubles the dimension and leaves a generalised linear eigenvalue problem in block form, the companion pencil. A window truncated at n_order carries 2 n_order + 1 harmonics, so the pencil has side 4 n_order + 2 and returns that many wavenumber eigenvalues with their harmonic mode shapes. That count is the first thing worth checking against the physics: the unmodulated rod supports two waves, one running each way, at each of the 2 n_order + 1 shifted frequencies, which is the same number.

Because the segment is posed for the wavenumber at a prescribed real frequency, the supersonic regime returns eigenvalues that are real to working precision. Every mode the segment supports then propagates along the rod without growing or decaying in space, so the amplification this task measures is not the spatial growth of an evanescent or spatially amplifying mode. The largest magnitude of the imaginary part across the spectrum is returned so that this can be confirmed rather than assumed, and a large value is the signature of a configuration that has left the supersonic regime. Note what the reality of this spectrum does not establish. It is a statement about how a disturbance varies in space at a prescribed real frequency, and it decides nothing about whether the finite segment is stable in time; that is a separate question about the bounded system, and answering it needs a Floquet analysis of the finite segment or a time-domain simulation rather than this spectrum.

The eigenvectors are fixed only up to a scale by any solver, so each is normalised to unit Euclidean length before it is returned. The eigenvalues themselves come back in whatever order the underlying routine produced, which is not reproducible across libraries or across threading configurations, so they are sorted here by real part and then by imaginary part purely to make the returned arrays deterministic. That ordering carries no physical meaning and must not be used to decide which mode is which; the next step decides that from a property of the mode itself.

```python
def floquet_wavenumber_spectrum(a2: np.ndarray, a1: np.ndarray, a0: np.ndarray) -> dict:
    """Linearise the quadratic wavenumber problem and solve it for every mode the segment supports.

    Parameters
    ----------
    a2 : np.ndarray
        Coefficient matrix of the square of the wavenumber.
    a1 : np.ndarray
        Coefficient matrix of the first power of the wavenumber.
    a0 : np.ndarray
        Constant coefficient matrix.

    Returns
    -------
    dict
        Under the keys wavenumbers, mode_shapes and max_abs_imag. The wavenumbers
        entry is a complex128 array of shape (4 n_order + 2,) sorted by real part
        then imaginary part. The mode_shapes entry is a complex128 array of shape
        (2 n_order + 1, 4 n_order + 2) whose columns are unit-norm mode shapes in
        the same order. The max_abs_imag entry is a native float.

    Raises
    ------
    ValueError
        If the three matrices are not square, are not all of the same shape, do not
        have odd side length, are not finite, or if the pencil returns a number of
        finite eigenvalues other than twice the side length.
    """
    return
```

### Step 4

04_directional_mode_families

Goal
----
The source method assigns the segment's modes to forward and backward group-velocity families to define its basic-mode indexing and retained modal subset. These are branch labels, not necessarily literal directions of physical energy flux in a driven medium. The assignment cannot be made by eigenvalue index, because the order returned by an eigensolver is a numerical convention rather than a physical label. It cannot be made by the sign of the wavenumber either, since the shifted wavenumber of a harmonic may have either sign on either family. It must be made using a branch property computed from the mode.

The source method uses group velocity. Differentiating the harmonic balance along a branch and projecting onto the mode gives it without finite differencing. Because the coefficient matrices assembled earlier are symmetric, the left eigenvector is the transpose of the right one, and the derivative of frequency with respect to wavenumber reduces to the ratio of the mode projected through the shifted-wavenumber diagonal and the coupling matrix to the mode projected through the shifted-frequency diagonal and scaled by density. Positive group velocity labels the forward family and negative group velocity labels the backward family. In the graded supersonic configuration the split is even, with 2 n_order + 1 modes in each family, which is the required source-method family count.

Time-averaged power flux remains the physical measure of energy-flow direction, but its sign does not reproduce these group-velocity family labels at the graded anticrossing. Two nearly degenerate modes can both carry positive flux while their group velocities have opposite signs. Flux signs therefore give an 18/16 split for the complete N=8 spectrum, whereas group-velocity labels give 17/17. This mismatch does not make the flux sign physically invalid and does not make unequal column groupings intrinsically impossible to match. It means that flux sign cannot substitute for the source method's equal-family group-velocity convention in this function.

Within each family the modes are labelled by basic-mode order. At vanishing modulation depth their physical wavenumbers are kappa_s^+ = (omega+s omega_m)/c0 - s kappa_m and kappa_s^- = -(omega+s omega_m)/c0 - s kappa_m. For omega_m > 0 and |omega_m/kappa_m| > c0, the forward physical wavenumber increases strictly with s and the backward physical wavenumber decreases strictly with s for either sign of kappa_m. Ordering the forward family by increasing real physical wavenumber and the backward family by decreasing real physical wavenumber therefore assigns s from minus n_order upwards in both. That correspondence survives the anticrossings, where a rule based on the dominant harmonic does not because the modes hybridise.

```python
def directional_mode_families(
    wavenumbers: np.ndarray,
    mode_shapes: np.ndarray,
    coupling: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    """Sort the Floquet modes into forward and backward families by group velocity and label them by basic-mode order.

    Parameters
    ----------
    wavenumbers : np.ndarray
        The wavenumber eigenvalues, shape (4 n_order + 2,).
    mode_shapes : np.ndarray
        Their harmonic mode shapes, shape (2 n_order + 1, 4 n_order + 2).
    coupling : np.ndarray
        Harmonic coupling operator of the modulus.
    omega : float
        Driving angular frequency in radians per second.
    rho0 : float
        Mass density in kilogram per cubic metre.
    kappa_m : float
        Modulation wavenumber in radians per metre.
    omega_m : float
        Modulation angular frequency in radians per second.

    Returns
    -------
    dict
        Under the keys forward_wavenumbers, backward_wavenumbers, forward_modes,
        backward_modes and group_velocities. The wavenumber entries are complex128
        arrays of shape (2 n_order + 1,) indexed by basic-mode order. The mode
        entries are complex128 arrays of shape (2 n_order + 1, 2 n_order + 1). The
        group_velocities entry is a complex128 array of shape (4 n_order + 2,).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, if any input is not finite, if omega or rho0
        is not above zero, if kappa_m or omega_m is zero, or if the group velocity
        does not split the spectrum into two families of equal size.
    """
    return
```

### Step 5

05_interface_coupling_system

Goal
----
The segment is bounded by two interfaces in space, and across each of them the displacement and the axial stress must be continuous at every instant. Because the exteriors are uniform and the interior is not, the two sides of each condition are written in different bases: outside, a sum of scattering orders each travelling at the wave speed of the unmodulated rod; inside, a sum over the basic modes of the previous step, each of which is itself a series in the harmonics. Projecting the conditions onto the time basis, which is orthogonal because the harmonic frequencies are distinct, turns each condition into one algebraic equation per retained order.

The count of retained orders is not free. The displacement condition involves the field alone and would balance at every order the expansion carries. The stress condition does not, because the stress inside the segment is the modulus times the gradient, and the modulus carries its own harmonic content: forming that product spreads each harmonic by up to P orders in each direction, so balancing the stress at order j calls on the interior field at orders j - P to j + P. Those lie inside the truncated window only when the magnitude of j does not exceed n_order - P. The mode-coupling order is therefore J = n_order - P, the scattering orders run from minus J to plus J, and only the basic modes of the same range are retained, which leaves 4 J + 2 interior amplitudes against 4 J + 2 exterior ones and a square system of 8 J + 4 equations. Matching at every order the expansion carries instead would call on harmonics the expansion does not hold and silently set them to zero.

Two conventions in the exterior have to be got right. A scattering order of order n sits at frequency omega + n omega_m, which for a sufficiently down-converted order is negative, and the wavenumber that keeps such an order travelling away from the segment is the signed ratio of that frequency to the wave speed rather than the positive root of its square. And the position of the second interface enters through more than the carrier: harmonic n of an interior mode advances with the shifted wavenumber, so the phase it accumulates across the segment is that of the mode plus n times the modulation wavenumber times the length. Those two contributions coincide only when the segment length is a whole number of modulation wavelengths, so a segment of any other length distinguishes an implementation that carries the shifted phase from one that does not.

The unknowns are ordered so that the solution of the next step can be sliced without ambiguity: the forward interior amplitudes indexed by basic-mode order, then the backward interior amplitudes in the same indexing, then the reflected amplitudes indexed by scattering order, then the transmitted amplitudes.

Phasor and amplitude convention: use the complex field convention exp[i(omega_n t - k_n x)], with omega_n = omega + n omega_m, and the common global coordinate x=0 at the left interface and x=segment_length at the right interface. In the left exterior, the reflected unknown R_n multiplies exp[i(omega_n t - k_n^- x)]; in the right exterior, the transmitted unknown T_n multiplies exp[i(omega_n t - k_n^+ x)], where k_n^- = -omega_n/c_0, k_n^+ = omega_n/c_0 and c_0 is the exterior wave speed. Thus the transmitted displacement at the right interface contains T_n exp(-i k_n^+ segment_length). Each interior amplitude multiplies the supplied mode vector in the same convention: its harmonic n carries exp[i(omega_n t - (kappa_s + n kappa_m)x)], where kappa_s is that supplied basic-mode wavenumber. The incident coefficient is one at x=0. Equivalent rescaling or reordering of the equation rows is permitted while the stated unknown ordering and amplitude definitions are retained.

```python
def interface_coupling_system(
    forward_wavenumbers: np.ndarray,
    forward_modes: np.ndarray,
    backward_wavenumbers: np.ndarray,
    backward_modes: np.ndarray,
    coefficients: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    segment_length: float,
) -> dict:
    """Assemble the square linear system that the two interface conditions impose on the interior and exterior amplitudes.

    Parameters
    ----------
    forward_wavenumbers : np.ndarray
        Forward basic-mode wavenumbers indexed by order.
    forward_modes : np.ndarray
        Their harmonic mode shapes.
    backward_wavenumbers : np.ndarray
        Backward basic-mode wavenumbers indexed by order.
    backward_modes : np.ndarray
        Their harmonic mode shapes.
    coefficients : np.ndarray
        One-dimensional modulus Fourier coefficients indexed by orders -P through +P,
        in that order, with odd length 2P+1 for any integer P >= 0. The single-cosine
        task instance has P=1 and therefore three entries. The P=0 one-entry limit is
        also supported by this function, including the N=0, J=0 one-mode limit.
    omega : float
        Driving angular frequency in radians per second.
    rho0 : float
        Mass density in kilogram per cubic metre.
    kappa_m : float
        Modulation wavenumber in radians per metre.
    omega_m : float
        Modulation angular frequency in radians per second.
    segment_length : float
        Length of the modulated segment in metres.

    Returns
    -------
    dict
        Under the keys matrix, rhs and j_order. The matrix entry is a complex128
        array of shape (8 J + 4, 8 J + 4) and the rhs entry a complex128 array of
        shape (8 J + 4,) for a unit-amplitude incident wave, with the unknowns
        ordered as the forward interior amplitudes, the backward interior
        amplitudes, the reflected amplitudes and the transmitted amplitudes. The
        j_order entry is the native int mode-coupling order J.

    Raises
    ------
    ValueError
        If the coefficients are not one-dimensional with odd length, if the mode-family
        shapes are inconsistent or not of odd side length, if any input is not finite,
        if omega, rho0 or segment_length is not above zero, if kappa_m or omega_m is
        zero, or if the truncation leaves no matchable order.
    """
    return
```

### Step 6

06_harmonic_scattering_coefficients

Goal
----
With the system assembled the scattering amplitudes follow from one dense solve. The unknown vector holds, in order, the forward interior amplitudes, the backward interior amplitudes, the reflected amplitudes and the transmitted amplitudes, each block running over its index from minus J to plus J, so the two exterior blocks are recovered by slicing the solution rather than by any further algebra. The scattering coefficients quoted for a unit incident amplitude are the magnitudes of those complex amplitudes.

The conditioning of the system deserves a look rather than a trust. The interior modes accumulate phase across the segment, and where the segment sits inside a wavenumber bandgap the two nearly degenerate modes that dominate the response carry almost the same phase, which brings their columns close to parallel. The condition number is returned so that a configuration in which the solve has lost its meaning can be recognised; in the regime this task prescribes it stays modest, and a value orders of magnitude larger is the signal that the segment has been pushed into a range where the steady scattering problem no longer has a well conditioned answer.

A second and more physical caution belongs here. The linear system returns a number for any parameters it is handed, including parameters for which the bounded segment has no steady state at all, because a sufficiently deep or sufficiently long modulation drives the system parametrically unstable and the response then grows without bound in time. Nothing in this solve detects that: the algebra is the algebra of a steady response that has been assumed to exist. Whether the prescribed configuration admits such a steady state is not settled by anything computed here; it is assumed, and the coefficients returned are conditional on that assumption.

```python
def harmonic_scattering_coefficients(matrix: np.ndarray, rhs: np.ndarray, j_order: int) -> dict:
    """Solve the interface system and read off the transmission and reflection magnitudes at every retained order.

    Parameters
    ----------
    matrix : np.ndarray
        The interface system matrix, shape (8 J + 4, 8 J + 4).
    rhs : np.ndarray
        Its right-hand side for unit incident amplitude, shape (8 J + 4,).
    j_order : int
        The mode-coupling order J, zero or more.

    Returns
    -------
    dict
        Under the keys transmission, reflection and condition_number. The
        transmission and reflection entries are float64 arrays of shape (2 J + 1,)
        indexed by scattering order from minus J to plus J. The condition_number
        entry is a native float.

    Raises
    ------
    ValueError
        If matrix is not square, if its size is not four times two J plus one, if
        rhs does not match it, if either is not finite, if j_order is negative, or
        if the system is singular.
    """
    return
```

### Step 7

07_harmonic_power_budget

Goal
----
Scattering coefficients are amplitudes, and amplitudes are not what a modulated segment conserves or fails to conserve. Energy is. Converting the two families of amplitudes into a power budget is what turns the computation into a statement about the segment drawing energy from the modulation rather than merely redistributing it.

In the uniform rod outside the segment, a scattering order of order n is a plane wave at frequency omega + n omega_m travelling at the unmodulated wave speed, and the time-averaged power it carries past a section is proportional to the product of its wavenumber, its frequency and the square of its amplitude. Since the wavenumber of such an order is its own frequency divided by the wave speed, the power goes as the square of the frequency times the square of the amplitude, with the same constant of proportionality for every order and for the incident wave. Referring each order to the incident wave therefore weights the square of its amplitude by the square of the ratio of its frequency to the incident frequency. A down-converted order of negative frequency contributes with the square of that ratio like any other, its sign having been absorbed into the direction it travels when the exterior wavenumber was assigned.

The sum of those referred powers over both families is the total gain of the segment for that direction of incidence. A lossless and time-invariant scatterer would return exactly one; a passive but lossy one would return less, so unity is the reference for a medium that neither absorbs nor is driven rather than a property of passivity alone. A value above one is the signature of parametric amplification, the modulation having done work on the wave, and it is available here only because the modulus depends explicitly on time, so that the balance of mechanical energy carries a source term proportional to the rate of change of the modulus. The split between the transmitted and the reflected part is reported alongside the total because the two are not amplified equally and the asymmetry between them is part of what the segment does.

```python
def harmonic_power_budget(
    transmission: np.ndarray,
    reflection: np.ndarray,
    omega: float,
    omega_m: float,
) -> dict:
    """Convert the scattering magnitudes into the power the segment returns, referred to the incident power.

    Parameters
    ----------
    transmission : np.ndarray
        Transmission magnitudes indexed by scattering order, odd length.
    reflection : np.ndarray
        Reflection magnitudes in the same indexing.
    omega : float
        Incident angular frequency in radians per second.
    omega_m : float
        Modulation angular frequency in radians per second.

    Returns
    -------
    dict
        Under the keys transmitted_gain, reflected_gain and total_gain, each a
        native float giving the referred power of that family relative to the
        incident power.

    Raises
    ------
    ValueError
        If the two arrays are not one-dimensional, of equal odd length, finite and
        non-negative, or if omega is not above zero or omega_m is zero.
    """
    return
```

### Step 8

08_nonreciprocal_gain_contrast

Goal
----
This step runs the whole chain twice and returns the asymmetry between the two runs. Step one builds the Fourier coefficients of the travelling modulus and the harmonic coupling operator they induce. Step two turns those into the quadratic matrix polynomial in the wavenumber at the prescribed frequency. Step three linearises that polynomial and solves it for every mode the segment supports, confirming along the way that the supersonic regime leaves the spectrum real. Step four sorts those modes into forward and backward families by the sign of their group velocity and labels them by basic-mode order. Step five imposes displacement and stress continuity at the two interfaces, retaining the scattering orders the truncation can actually balance. Step six solves that system for the transmission and reflection magnitudes. Step seven converts them into the power the segment returns, referred to the power it received.

The asymmetry comes from the direction of the modulation, not from the geometry, which is symmetric. Reversing the sign of the modulation wavenumber reverses the direction in which the stiffness pattern travels and therefore exchanges the case of a wave running with the modulation for the case of a wave running against it, while leaving the segment, its length and the incident frequency untouched. Running the chain once with each sign and taking the ratio of the two total gains isolates that dependence. Its value is one for a reciprocal scatterer and departs from one only because the modulation has a direction, which is the whole content of the phenomenon.

Only the modulation wavenumber is reversed; the modulation frequency is left alone. The harmonic frequencies omega_n = omega_0 + n omega_m are therefore the same in both runs and the scattering orders are not relabelled. What changes is which orders the segment feeds, because the phase matching that couples a forward branch to a backward one depends on the direction the modulus travels. Posing the graded quantity as a total over all retained orders rather than as a single order keeps it independent of which individual order happens to dominate in each direction.

```python
def nonreciprocal_gain_contrast(
    alpha_m: float,
    E0: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    omega: float,
    segment_length: float,
    n_order: int,
) -> dict:
    """Run the scattering chain for both directions of incidence and return the ratio of the total gains.

    Parameters
    ----------
    alpha_m : float
        Normalised modulation depth, of magnitude below one.
    E0 : float
        Elastic modulus of the unmodulated rod in pascal.
    rho0 : float
        Mass density in kilogram per cubic metre.
    kappa_m : float
        Modulation wavenumber in radians per metre for positive incidence.
    omega_m : float
        Modulation angular frequency in radians per second.
    omega : float
        Incident angular frequency in radians per second.
    segment_length : float
        Length of the modulated segment in metres.
    n_order : int
        Truncation order of the harmonic expansion, at least two.

    Returns
    -------
    dict
        Under the keys positive_gain, negative_gain, contrast,
        positive_transmission_zero and positive_reflection_minus_one, each a native
        float. The contrast entry is the ratio of positive_gain to negative_gain.

    Raises
    ------
    ValueError
        If any argument violates its stated range, if n_order is below two, or if
        the negative-incidence run returns a gain of zero.
    """
    return
```
