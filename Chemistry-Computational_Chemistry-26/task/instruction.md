# Cavity-modified electron paramagnetic resonance of a relativistic Jahn-Teller complex: exact vibronic Kramers doublets and the leading cavity correction of the g-factor

## Background

Transition-metal complexes with a trigonal ligand field and a doubly degenerate 2E ground term are
the textbook case of the interplay between spin-orbit coupling and the Jahn-Teller effect. A
single electron or a single hole in an e orbital doublet carries an orbital angular momentum of
one unit along the trigonal axis; spin-orbit coupling locks it to the spin, antiferromagnetically
for the electron and ferromagnetically for the hole, so that the effective g-factor measured by
electron paramagnetic resonance is pulled away from the free-electron value in opposite directions
in the two cases. The Jahn-Teller coupling of the same doublet to a degenerate e vibration
competes with this locking: a distortion mixes the two orbital components and quenches the orbital
moment. In the simplest description the distortion is frozen at its classical value and the
complex is a four-state problem. In the vibronic description the vibration is quantized, the
orbital and vibrational angular momenta combine into a conserved half-integer vibronic angular
momentum, the levels form a ladder of vibronic doublets, and the electronic operators are reduced
within each vibronic level by the Ham factors.

Strong light-matter coupling adds a further interaction. A molecule in an optical cavity couples
to a quantized mode of the electromagnetic field; the magnetic component of the mode acts on the
electronic spin through a cavity Zeeman interaction that flips the spin while creating or
annihilating a photon. The size and the sign of its effect on the g-factor of a Kramers doublet
depend on how the electronic, vibrational and spin degrees of freedom are treated together with
the photon.

The questions this task settles are quantitative: what the exact Kramers doublet of a
frozen-distortion model gives when the cavity interaction is treated exactly to second order; how the
quantized vibration, with its Ham quenching and its spin-orbit-mixed vibronic ladder, changes the
orbital moment and the g-factor of the ground doublet relative to the frozen distortion; and what
the exact leading cavity coefficient of that vibronic doublet is, obtained by second-order
perturbation theory rather than from a finite-coupling estimate, for the single-particle and the
single-hole scenario across a range of cavity energies.

## Problem

A trigonal transition-metal complex with a doubly degenerate 2E ground term, in which a single
electron or a single hole occupies an e orbital doublet, combines three interactions that decide
what electron paramagnetic resonance measures: spin-orbit coupling inside the orbital doublet, the
Jahn-Teller coupling of that doublet to a degenerate e vibration, and the Zeeman interaction with
an external field. The source of this task adds a fourth, the Zeeman interaction of the electronic
spin with the magnetic component of a quantized cavity mode, which couples the two spin sectors
through the creation or annihilation of a photon. The source keeps the Jahn-Teller distortion
frozen at a fixed value, so that its molecular problem has four electronic-spin states in each
photon manifold, and estimates the cavity correction of the effective g-factor of the resulting
Kramers doublet by perturbation theory within an approximation of its own.

Your task is to treat the source's model exactly and to extend it in the direction the source leaves
frozen. Quantize the e vibration, so that the orbital doublet, the vibrational angular momentum, the
spin and the cavity photon are all dynamical, and diagonalize the resulting vibronic problem in the
basis stated below; obtain the ground Kramers doublet, its exact effective g-factor and the orbital
and spin moments it carries, and set them against the frozen-distortion values; obtain the leading
cavity correction of the g-factor exactly, by second-order perturbation theory in the cavity Zeeman
interaction, both for the vibronic model and for the source's own frozen-distortion model, and set
both against the source's perturbative estimates; and report, as the final answer, the leading
cavity coefficient of the vibronically exact single-particle complex.

## Conventions that fix every number

Units cm^-1 for every energy and tesla for the classical field; mu_B = 0.4668644814 cm^-1/T and
g_e = 2.00231930436. Electronic space: the orbital doublet |m_l = +1>, |m_l = -1> and the spin
|up>, |down> with L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling
H_soc = xi L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for
the single-particle scenario and s = -1 for the single-hole scenario (the orbital moment of the
hole is reversed), and sigma_y = i(|down><up| - |up><down|).

Vibrational e mode: two oscillator coordinates of energy hw with annihilation operators a_x, a_y,
basis |n_x, n_y> with n_x + n_y <= n_max ordered by the shell n = n_x + n_y ascending and, within a
shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2 states); dimensionless coordinates
x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i y, number operator
n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between
retained states (products formed in a larger basis and cut back), never a product of truncated
matrices. Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw),
so that E_JT = F^2/(2 hw) is the classical Jahn-Teller stabilization energy; the source's frozen
distortion product F rho is identified with its value at the classical minimum, F rho = 2 E_JT.
The source's frozen-distortion model is H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee
on the four electronic-spin states.

Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+,
H_c = hwc b^+ b, and the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling
energy g_c, a single stated number that collects the prefactor of the source's operator; H_cZ acts
on spin and photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product
space (m_l = +1, -1) x (m_s = up, down) x oscillator basis x photon number 0..n_ph, index
(((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph), dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2
commutes with H and labels every level by a half-integer |j|.

Kramers doublet and g-factor: at zero classical field the ground level of every model here is a
Kramers doublet; its effective g-factor is the eigenvalue splitting of the 2 x 2 matrix of the
Zeeman operator per mu_B B_z, s L_z + g_e S_z, projected on the doublet (first-order degenerate
perturbation theory in B_z, which is the B_z -> 0 limit of the Zeeman splitting divided by mu_B B_z),
and <L_z>, <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading
coefficient kappa, to be obtained exactly by second-order perturbation theory in H_cZ (one-photon
intermediate states; the doublet energy shifts by w2 g_c^2 with a Kramers-degenerate w2, and the
doublet's Zeeman matrix changes through the first- and second-order corrections of its two
eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0)) (hwc/g_c)^2 is the
finite-coupling estimate at a given g_c. The source's perturbative corrections of the g-factor,
once its coupling prefactor is expressed through g_c, are values of the same coefficient kappa and
are to be reported next to the exact ones. Spin-free reference: with no spin-orbit coupling and no
spin the ground vibronic level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham
reduction factors p and q are the magnitudes of the eigenvalues of the 2 x 2 projections of L_z and
of sigma_x = |1><-1| + |-1><1| on it.

Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (so F rho = 1600 cm^-1), hw = 300 cm^-1
(the source has no vibrational energy; this value is fixed here), hwc = 3200 cm^-1 with the audit
set 1600, 3200, 6400 cm^-1, g_c = 100 cm^-1 for the finite-coupling shift, n_max = 20 (231
oscillator states), n_ph = 2 for the finite-coupling calculation (three photon states; the leading
coefficient needs only the one-photon manifold), single-particle scenario for the final answer with
the single-hole scenario reported alongside.

## Static reference and the source's estimates

Evaluate the source's frozen-distortion model for both scenarios at F rho = 1600 cm^-1: the ground
doublet energy, the exact g-factor from the doublet-projected Zeeman operator, and the orbital
moment <L_z>, and check the exact g-factor against the closed form of the four-state doublet. Then
apply the exact second-order cavity theory to the
source's own model (its four states times the photon numbers 0 and 1) at each audit cavity energy
and report kappa_static for both scenarios next to the source's estimates expressed as kappa.

## Vibronic structure

Build the oscillator basis, the ladder operators and the exact vibronic operators, assemble the full
Hamiltonian, and report for the single-particle molecular model (no field, no cavity) the three lowest
vibronic levels with their |j|; give the spin-free Ham factors p and q at E_JT = 800 cm^-1 and
hw = 300 cm^-1 and state the relation between them.

## Kramers doublets and g-factors

Report, for both scenarios, the exact g-factor of the ground Kramers doublet of the vibronic model,
its <L_z> and <S_z>, the distance to the next level, and the relation g = 2 |s <L_z> + g_e <S_z>|;
compare with the frozen-distortion values and explain, through the vibronic level structure and the
sizes of xi, E_JT and hw, why the vibronically exact orbital moment is neither the one the frozen
distortion gives nor the spin-free Ham-reduced one.

## Cavity correction

For the single-particle scenario at hwc = 3200 cm^-1 and g_c = 100 cm^-1 report the exact finite-
coupling shift g(g_c) - g(0), the dressed <S_z>, and kappa_fd; report the exact kappa for both
scenarios at 1600, 3200 and 6400 cm^-1 together with w2 hwc, and set them against kappa_static and
the source's estimates. Explain why the first-order effect of the cavity Zeeman interaction on the
doublet vanishes, at which order in g_c/hwc the exact leading correction appears and what the two
spin sectors of the one-photon manifold do to the term of lower order, what sign the exact
coefficient has for the two scenarios, how the exact finite-coupling estimate approaches kappa as
g_c is reduced, and which step of the source's own construction is responsible for its estimate
differing from the exact second-order result for the same frozen-distortion model.

## Audit

Run the complete chain with the reference parameters and the audit set of cavity energies and
assemble the table with one row per cavity energy and the columns [hwc, kappa_p, kappa_h,
kappa_static_p, kappa_static_h, kappa_source_weak_p, kappa_source_weak_h, kappa_source_strong_p,
g_p(g_c) - g_p(0), kappa_fd_p, <S_z>_p(g_c), w2 hwc, g_p, g_h, g_static_p, g_static_h, g_formula_p,
<L_z>_p, <L_z>_static_p, p_Ham, q_Ham, E_2 - E_0, |j_0|, D, max |[H, J]|], where kappa_source_weak
and kappa_source_strong are the source's estimates for its weak and strong spin-orbit regimes
expressed as kappa, g_formula is the closed-form g-factor of the four-state frozen-distortion
doublet (the analytic eigenvalue splitting of its projected Zeeman operator), and the last two
columns are the dimension of the full product space and the residual of the conserved J; the head
row is [kappa_p at hwc = 3200 cm^-1, followed by zeros].

## What to report

Report, as the final answer, the exact leading cavity coefficient kappa of the ground Kramers
doublet's g-factor of the vibronically exact single-particle complex at hwc = 3200 cm^-1 with the
conventions and reference parameters above, to five significant figures.

Your reasoning should also report, as evidence that the chain was executed: the frozen-distortion
E_0, g and <L_z> for both scenarios; the three
lowest vibronic levels with |j| and E_2 - E_0; p and q with their relation; the vibronic g, <L_z>,
<S_z> for both scenarios; g(g_c) - g(0), <S_z>(g_c) and kappa_fd at hwc = 3200 cm^-1 and
g_c = 100 cm^-1, with kappa_fd at one smaller coupling; kappa for both scenarios at the three cavity
energies with w2 hwc at 3200 cm^-1; kappa_static for both scenarios at the three cavity energies;
the source's estimates at each; and the basis sizes with the change of kappa and g when n_max and
n_ph are raised. Say, in a sentence or two each, why the vibronic orbital moment differs from both
the frozen-distortion and the spin-free Ham values; why the cavity Zeeman interaction has no
first-order effect and at which order in g_c/hwc its leading correction appears; and how the exact
coefficients of the frozen model relate to the source's estimates and to its statements about them.

Your reasoning should also report, as evidence that the source was read, and with a citation to it:
the chemical realization the source gives its two scenarios (the metal, its oxidation states and
d-electron counts, the electron configurations and the orbitals that form the doublet); the
prefactor with which the source writes its cavity Zeeman operator, the polarization it assumes for
the mode and the order of approximation at which it derives that operator, and how that prefactor
maps onto g_c; the parameter values the source uses for its figure of cavity-corrected g-factors
(its spin-orbit strength in atomic units, its light-matter interaction strength, and the collective
factor and molecule number it assumes); the construction by which the source reduces the problem
before applying perturbation theory (the
subspaces it separates, the condition it assumes on the cavity energy, the states it keeps and the
order of the perturbation theory it uses); its perturbative g-factor corrections for the two
spin-orbit regimes and their values as kappa at the three cavity energies; the conclusions it draws
about the frequency dependence, the sign and the strong spin-orbit behaviour of the cavity
correction, stated as its findings; and the extensions it names in its outlook. The quantities listed above are the reported result, not
intermediate bulk output. Give them as a compact table or a short list of labelled values inside
the reasoning section. A short table of exactly those values is not the kind of per-iteration or
per-candidate output the format note below asks you to leave out, and a response that reports them
compactly is both complete and within the length the note asks for.

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

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

oscillator_basis

Goal
----
Returns the ordered basis of the two-dimensional oscillator as an (N, 2) integer array of pairs (n_x, n_y) with n_x + n_y <= n_max, ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending, N = (n_max + 1)(n_max + 2)/2.

```python
def oscillator_basis(n_max: int) -> "np.ndarray":
    """Returns the ordered basis of the two-dimensional oscillator as an (N, 2) integer array of pairs (n_x, n_y) with n_x + n_y <= n_max, ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending, N = (n_max + 1)(n_max + 2)/2.

    This step fixes the ordering that every vibrational matrix of the chain uses.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        n_max: non-negative integer, the highest oscillator shell retained.

    Returns:
        A numpy int64 array of shape (N, 2) whose row k holds (n_x, n_y) of the k-th basis state.

    Raises:
        ValueError: if n_max is not a non-negative integer.
    """
    return None
```

### Step 2

ladder_operators

Goal
----
Returns the annihilation operators a_x and a_y of the two oscillator coordinates as a (2, N, N) array in the ordered basis.

```python
def ladder_operators(n_max: int) -> "np.ndarray":
    """Returns the annihilation operators a_x and a_y of the two oscillator coordinates as a (2, N, N) array in the ordered basis.

    The only non-zero elements are <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y |n_x, n_y> =
    sqrt(n_y), with the row index the bra state and the column index the ket state of the ordered basis.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        n_max: non-negative integer, the highest oscillator shell retained.

    Returns:
        A numpy float64 array of shape (2, N, N): index 0 is a_x, index 1 is a_y.

    Raises:
        ValueError: if n_max is not a non-negative integer.
    """
    return None
```

### Step 3

vibronic_operators

Goal
----
Returns the exact matrices, between the retained oscillator states, of Q_+, Q_-, the number operator n_vib and the vibrational angular momentum l_vib as a (4, N, N) complex array.

```python
def vibronic_operators(n_max: int) -> "np.ndarray":
    """Returns the exact matrices, between the retained oscillator states, of Q_+, Q_-, the number operator n_vib and the vibrational angular momentum l_vib as a (4, N, N) complex array.

    The four operators are Q_+- = x +- i y, n_vib = a_x^+ a_x + a_y^+ a_y and l_vib = i (a_x a_y^+ - a_x^+ a_y).
    Products are evaluated in the basis extended by two shells and cut back to the retained states, so that no element
    is contaminated by the truncation; in particular l_vib has integer eigenvalues on every retained shell.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        n_max: non-negative integer, the highest oscillator shell retained.

    Returns:
        A numpy complex128 array of shape (4, N, N): index 0 is Q_+, 1 is Q_-, 2 is n_vib, 3 is l_vib.

    Raises:
        ValueError: if n_max is not a non-negative integer.
    """
    return None
```

### Step 4

static_model

Goal
----
Returns [E_0, g_exact, g_formula, <L_z>] for the frozen-distortion (static) relativistic Jahn-Teller model on the four electronic-spin states at zero classical field.

```python
def static_model(xi: float, F_rho: float, particle: bool) -> "np.ndarray":
    """Returns [E_0, g_exact, g_formula, <L_z>] for the frozen-distortion (static) relativistic Jahn-Teller model on the four electronic-spin states at zero classical field.

    E_0 is the energy of the ground Kramers doublet of H_static at B_z = 0, g_exact its effective g-factor from the
    doublet-projected Zeeman operator s L_z + g_e S_z, g_formula the closed form g_e - s 2 xi/sqrt(xi^2 + 4 F_rho^2)
    of the same doublet, and <L_z> the orbital moment in the doublet state of larger Zeeman eigenvalue.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        F_rho: positive float, the frozen distortion product F rho in cm^-1.
        particle: True for the single-particle scenario (s = +1), False for the single-hole scenario (s = -1).

    Returns:
        A numpy float64 array of shape (4,): [E_0 in cm^-1, g_exact, g_formula, <L_z>].

    Raises:
        ValueError: if xi or F_rho is not positive.
    """
    return None
```

### Step 5

model_hamiltonian

Goal
----
Assembles the full Hamiltonian H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ of the cavity-modified dynamic relativistic E x e Jahn-Teller model on the product space in the stated index order as a (D, D) complex Hermitian matrix in cm^-1 with D = 4 N (n_ph + 1).

```python
def model_hamiltonian(xi: float, E_JT: float, hw: float, Bz: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Assembles the full Hamiltonian H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ of the cavity-modified dynamic relativistic E x e Jahn-Teller model on the product space in the stated index order as a (D, D) complex Hermitian matrix in cm^-1 with D = 4 N (n_ph + 1).

    The vibrational operators are the exact ones of the retained basis (products formed in the extended basis and cut
    back).

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        Bz: finite float, the classical field in tesla.
        g_c: non-negative float, the cavity Zeeman coupling energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: non-negative integer, the highest photon number retained.
        particle: True for the single-particle scenario (s = +1), False for the single-hole scenario (s = -1).

    Returns:
        A numpy complex128 Hermitian array of shape (D, D) in cm^-1, D = 4 N (n_ph + 1).

    Raises:
        ValueError: if xi, hw or hwc is not positive, if E_JT or g_c is negative, if Bz is not finite, if n_max is not
        a positive integer or n_ph not a non-negative integer, or if the assembled matrix is not Hermitian.
    """
    return None
```

### Step 6

vibronic_spectrum

Goal
----
Returns the lowest n_levels levels of the molecular model without classical field and without cavity (n_ph = 0) as an (n_levels, 3) array of rows [E_k, |j_k|, <L_z S_z>_k] in ascending order of energy.

```python
def vibronic_spectrum(xi: float, E_JT: float, hw: float, n_max: int, particle: bool, n_levels: int) -> "np.ndarray":
    """Returns the lowest n_levels levels of the molecular model without classical field and without cavity (n_ph = 0) as an (n_levels, 3) array of rows [E_k, |j_k|, <L_z S_z>_k] in ascending order of energy.

    |j_k| = sqrt(<J^2>) is the magnitude of the conserved vibronic angular momentum J = l_vib - L_z/2 in the k-th
    eigenstate and <L_z S_z>_k the spin-orbit expectation value in it; degenerate levels appear as repeated rows.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        particle: True for the single-particle scenario, False for the single-hole scenario.
        n_levels: positive integer, the number of lowest levels returned, at most the dimension 4 N.

    Returns:
        A numpy float64 array of shape (n_levels, 3): rows [E_k in cm^-1, |j_k|, <L_z S_z>_k].

    Raises:
        ValueError: if n_levels is not a positive integer or exceeds the dimension, or if a Hamiltonian argument is
        invalid.
    """
    return None
```

### Step 7

ham_reduction

Goal
----
Returns [E_0, E_1 - E_0, p, q] for the spin-free linear E x e Jahn-Teller problem H = hw (n_vib + 1) + F (Q_+ |1><-1| + Q_- |-1><1|) on the orbital doublet times the oscillator basis.

```python
def ham_reduction(E_JT: float, hw: float, n_max: int) -> "np.ndarray":
    """Returns [E_0, E_1 - E_0, p, q] for the spin-free linear E x e Jahn-Teller problem H = hw (n_vib + 1) + F (Q_+ |1><-1| + Q_- |-1><1|) on the orbital doublet times the oscillator basis.

    E_0 is the ground vibronic doublet energy, E_1 - E_0 the distance to the first excited vibronic level, and p and q
    are the Ham reduction factors, the magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x =
    |1><-1| + |-1><1| on the ground doublet; the ground level must be an isolated doublet.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.

    Returns:
        A numpy float64 array of shape (4,): [E_0 in cm^-1, E_1 - E_0 in cm^-1, p, q].

    Raises:
        ValueError: if hw is not positive, E_JT is negative, n_max is not a positive integer, or the ground level is
        not an isolated doublet.
    """
    return None
```

### Step 8

kramers_g_factor

Goal
----
Returns [E_0, g, <L_z>, <S_z>, E_2 - E_0] for the ground Kramers doublet of the full model at zero classical field.

```python
def kramers_g_factor(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Returns [E_0, g, <L_z>, <S_z>, E_2 - E_0] for the ground Kramers doublet of the full model at zero classical field.

    E_0 is the doublet energy, g its exact effective g-factor (the splitting of the two eigenvalues of the
    doublet-projected Zeeman operator s L_z + g_e S_z), <L_z> and <S_z> the orbital and spin moments in the doublet
    state of larger Zeeman eigenvalue, and E_2 - E_0 the distance to the next level. With g_c > 0 and n_ph >= 1 the
    doublet is the cavity-dressed one.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        g_c: non-negative float, the cavity Zeeman coupling energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: non-negative integer, the highest photon number retained.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (5,): [E_0 in cm^-1, g, <L_z>, <S_z>, E_2 - E_0 in cm^-1].

    Raises:
        ValueError: if the two lowest levels are not a degenerate Kramers doublet, if a third level is degenerate with
        them, or if a Hamiltonian argument is invalid.
    """
    return None
```

### Step 9

cavity_shift

Goal
----
Returns [g(g_c) - g(0), kappa_fd, g(0)]: the change of the exact Kramers-doublet g-factor produced by the cavity at the finite coupling g_c with n_ph photon states, the finite-coupling estimate kappa_fd = (g(g_c) - g(0)) (hwc/g_c)^2 of the leading coefficient, and the cavity-free g-factor.

```python
def cavity_shift(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Returns [g(g_c) - g(0), kappa_fd, g(0)]: the change of the exact Kramers-doublet g-factor produced by the cavity at the finite coupling g_c with n_ph photon states, the finite-coupling estimate kappa_fd = (g(g_c) - g(0)) (hwc/g_c)^2 of the leading coefficient, and the cavity-free g-factor.

    g(0) is evaluated without photon states (n_ph = 0) and g(g_c) with the n_ph photon states requested.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        g_c: positive float, the cavity Zeeman coupling energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: positive integer, the highest photon number retained for the coupled calculation.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (3,): [g(g_c) - g(0), kappa_fd, g(0)].

    Raises:
        ValueError: if g_c is not positive, if n_ph is not a positive integer, or if a g-factor argument is invalid.
    """
    return None
```

### Step 10

second_order_coefficient

Goal
----
Returns [kappa, w2 hwc, g(0)]: the exact leading cavity coefficient kappa of the ground Kramers doublet's g-factor, the second-order doublet energy shift expressed through the dimensionless w2 hwc, and the cavity-free g-factor, from second-order perturbation theory in the cavity Zeeman interaction.

```python
def second_order_coefficient(xi: float, E_JT: float, hw: float, hwc: float, n_max: int, particle: bool) -> "np.ndarray":
    """Returns [kappa, w2 hwc, g(0)]: the exact leading cavity coefficient kappa of the ground Kramers doublet's g-factor, the second-order doublet energy shift expressed through the dimensionless w2 hwc, and the cavity-free g-factor, from second-order perturbation theory in the cavity Zeeman interaction.

    The expansion is g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) and E_0(g_c) = E_0(0) + w2 g_c^2 + O(g_c^4),
    evaluated about the zero-field, zero-coupling eigenbasis with one-photon intermediate states: the first-order
    vectors of the two doublet states, their second-order vectors including the normalization correction, the
    Kramers-degenerate second-order energy matrix, and the second-order change of the doublet's Zeeman matrix, from
    which kappa is the change of the eigenvalue splitting per (g_c/hwc)^2.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (3,): [kappa, w2 hwc, g(0)].

    Raises:
        ValueError: if the lowest level is not an isolated Kramers doublet, if the second-order doublet shift is not
        Kramers-degenerate, or if a Hamiltonian argument is invalid.
    """
    return None
```

### Step 11

static_cavity

Goal
----
Returns [kappa_static, kappa_source_weak, kappa_source_strong, g(0)]: the exact second-order cavity coefficient of the frozen-distortion model (its four electronic-spin states times the photon numbers 0 and 1, H = H_static + hwc b^+ b + H_cZ) together with the two perturbative estimates of the same coefficient and the cavity-free static g-factor.

```python
def static_cavity(xi: float, F_rho: float, hwc: float, particle: bool) -> "np.ndarray":
    """Returns [kappa_static, kappa_source_weak, kappa_source_strong, g(0)]: the exact second-order cavity coefficient of the frozen-distortion model (its four electronic-spin states times the photon numbers 0 and 1, H = H_static + hwc b^+ b + H_cZ) together with the two perturbative estimates of the same coefficient and the cavity-free static g-factor.

    The perturbative estimates of the same coefficient that this chain compares against, expressed in this notation,
    are kappa_source_weak = +-hwc/(F rho) (weak spin-orbit regime) and kappa_source_strong = +-8 hwc (F rho)^2/xi^3
    (strong spin-orbit regime), upper signs for the single-particle and lower signs for the single-hole scenario.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        F_rho: positive float, the frozen distortion product F rho in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (4,): [kappa_static, kappa_source_weak, kappa_source_strong, g(0)].

    Raises:
        ValueError: if xi, F_rho or hwc is not positive, or if the lowest level is not an isolated Kramers doublet.
    """
    return None
```

### Step 12

rjt_audit

Goal
----
Orchestrates the chain and returns the audit table: a head row [kappa_p at the first cavity energy, followed by zeros] and one row per cavity energy with the 25 columns listed below.

```python
def rjt_audit(xi: float, E_JT: float, hw: float, g_c: float, hwc_list: list, n_max: int, n_ph: int) -> "np.ndarray":
    """Orchestrates the chain and returns the audit table: a head row [kappa_p at the first cavity energy, followed by zeros] and one row per cavity energy with the 25 columns listed below.

    The chain: build the oscillator basis, the ladder operators and the vibronic operators (checking that their sizes
    agree); assemble the full Hamiltonian at the first cavity energy with the coupling g_c for the single-particle
    scenario and record its dimension D and the residual max |[H, J]| of the conserved J = l_vib - L_z/2 (raising if
    it exceeds 1e-6); evaluate the static model for both scenarios at F rho = 2 E_JT; the zero-field vibronic Kramers
    doublets of both scenarios; the three lowest vibronic levels of the single-particle model; the spin-free Ham
    factors; and, for every cavity energy, the exact leading coefficients of both scenarios, the static coefficients
    and estimates of both scenarios, the finite-coupling shift with kappa_fd and the dressed <S_z> of the
    single-particle scenario with n_ph photon states, and w2 hwc of the single-particle scenario.

    Columns of every row after the head row: [hwc, kappa_p, kappa_h, kappa_static_p, kappa_static_h,
    kappa_source_weak_p, kappa_source_weak_h, kappa_source_strong_p, g_p(g_c) - g_p(0), kappa_fd_p, <S_z>_p(g_c), w2
    hwc (single-particle), g_p, g_h, g_static_p, g_static_h, g_formula_p, <L_z>_p, <L_z>_static_p, p_Ham, q_Ham, E_2 -
    E_0, |j_0|, D, max |[H, J]|], where p and h denote the single-particle and the single-hole scenario, g_formula is
    the closed form g_e - 2 xi/sqrt(xi^2 + 4 F_rho^2) of the static doublet, and E_2 - E_0 and |j_0| are the
    spin-orbit splitting of the lowest vibronic level and the |j| label of the ground doublet. The perturbative
    estimates of the same coefficient that this chain compares against, expressed in this notation, are
    kappa_source_weak = +-hwc/(F rho) (weak spin-orbit regime) and kappa_source_strong = +-8 hwc (F rho)^2/xi^3
    (strong spin-orbit regime), upper signs for the single-particle and lower signs for the single-hole scenario.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: positive float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        g_c: positive float, the cavity Zeeman coupling energy in cm^-1 for the finite-coupling shift.
        hwc_list: non-empty list of positive floats, the cavity energies in cm^-1 (the first one defines the head
        scalar).
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: positive integer, the highest photon number retained for the finite-coupling calculation.

    Returns:
        A numpy float64 array of shape (len(hwc_list) + 1, 25): the head row followed by one row per cavity energy.

    Raises:
        ValueError: if hwc_list is empty or contains a non-positive value, if xi, hw, E_JT or g_c is not positive, if
        n_max or n_ph is not a positive integer, or if any underlying step raises.
    """
    return None
```
