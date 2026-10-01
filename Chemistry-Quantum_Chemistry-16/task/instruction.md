# Chemistry-Quantum_Chemistry-16

## Background

# Scientific background

## Seniority and the pair model of strong correlation

The seniority of a determinant or configuration is the number of singly
occupied spatial orbitals in it. When you sort a configuration interaction
expansion by seniority, you separate two different types of electron
correlation. In the seniority-zero sector, each occupied spatial orbital
contains a full electron pair. This sector contains most of the *strong*
(static) correlation. Strong correlation occurs when bonds are stretched and
one determinant is not sufficient. The sectors with higher seniority add the
*weak* (dynamic) correlation.

Research in this field shows that you can always select a special set of
orbitals. These orbitals make the contribution of the seniority-conserving
part of the Coulomb Hamiltonian as large as possible. Therefore, a reference
made of electron pairs is a good starting point. But the cost of a complete
seniority-zero configuration interaction calculation still increases
exponentially.

## Richardson-Gaudin states and the perfect-pairing limit

Richardson-Gaudin (RG) states are the exact eigenvectors of a reduced
Bardeen-Cooper-Schrieffer pairing Hamiltonian. This Hamiltonian has arbitrary
single-particle energies and a constant pairing strength. An RG state is a
product of pair creators. Each pair creator is spread over the orbitals, with
weights. Numbers called rapidities set these weights. RG states are an
alternative to seniority-based configuration interaction, with a polynomial
cost. But the rapidities are the solution of a set of coupled non-linear
equations, the Richardson equations. Because of this, the wavefunction is
difficult to interpret, and its derivatives are expensive to calculate.

Numerical studies of the optimal single-particle energies show a
core-valence-virtual pattern. Core values are isolated and strongly negative.
Valence values occur in near-degenerate pairs. Virtual values are isolated and
strongly positive. Each near-degenerate valence pair, together with its two
spatial orbitals, is a *valence-bond subsystem* (VBS). A VBS has one bonding
orbital, one antibonding orbital and one electron pair.

In one limit, pairs move only inside each VBS. In this limit, the reduced
pairing Hamiltonian separates into independent problems, one for each VBS, and
its eigenvectors become products. This limit is the perfect-pairing (PP)
limit. It is closely related to the generalized valence bond model and to the
antisymmetric product of strongly orthogonal geminals. The PP energy is a
functional of only the one-particle reduced density matrix.

Each VBS has one scalar parameter, the VBS gap. The gap sets the balance
between two effects. Aufbau filling puts the pair into the bonding orbital.
Coulomb repulsion moves the pair toward the antibonding orbital. A large gap
means weak correlation in that pair. A small gap means strong correlation. A
gap equal to one is the boundary between the two regimes.

## Optimization of a perfect-pairing reference

A PP reference has two types of variational parameters. Both types must be
stationary, as in a multiconfigurational self-consistent-field calculation.
The *orbital* degrees of freedom are the rotations between the spatial
orbitals. The *electronic* degrees of freedom are the VBS gaps.

A generalized Fock matrix gives the orbital gradient. For a state with a
seniority of exactly zero, this matrix has three parts. These parts are the
one-electron integrals, the direct block of the two-body reduced density
matrix and the pair block of the two-body reduced density matrix. The
electronic gradient is much simpler. The condition that this gradient is zero
fixes each gap.

Two families of two-electron integrals are important in this algebra. The
first family is the direct and exchange integrals, as in Hartree-Fock theory.
The second family is the pair-transfer integrals. These integrals move an
electron pair from one spatial orbital to a different spatial orbital. For
real orbitals, the exchange integrals and the pair-transfer integrals have the
same numerical values. But the formulas use them in different places.

## Epstein-Nesbet perturbation theory on a pair reference

Perfect pairing includes the strong correlation inside each pair. But it does
not include the weaker correlation *between* pairs. Second-order perturbation
theory over the low-lying excitations of the reference is a low-cost method to
add part of this correlation.

In the Epstein-Nesbet partition, the zeroth-order Hamiltonian is diagonal in
the selected basis of reference and excited states. Each diagonal entry is the
true expectation value of the Coulomb Hamiltonian in that state. Each excited
state then adds one term. This term is made from the square of the coupling of
the state to the reference, divided by the excitation energy of the state. The
total correction is negative.

Two properties of this method are important. First, the correction changes if
you use a different orthonormal basis for the excited space. Therefore, the
selection of that basis is part of the definition of the method. It is not a
free convention. Second, ordinary second-quantized single and double
excitations of a determinant do not give a good description of excitations
from a pair reference. An excitation operator that acts on a correlated pair
state removes some configurations, and the Pauli principle blocks other
configurations. Therefore, you must construct and normalize these excited
states as separate objects.

## The model system

Minimal-basis hydrogen chains are the standard test systems for these methods.
For these chains, the exact answer (full configuration interaction) is
available. You can change the amount of strong correlation if you stretch the
chain. A linear H4 chain in a minimal basis has exactly four spatial orbitals
and four electrons. This is the smallest system that has two interacting
valence-bond subsystems and no core or virtual orbitals.

## Problem

A valence space has four spatial orbitals and four electrons. Treat it with
the perfect-pairing limit of Richardson-Gaudin theory, in its recent
formulation for strongly correlated electrons. The valence has two
valence-bond subsystems (VBS). VBS A contains spatial orbitals 0 and 1. VBS B
contains spatial orbitals 2 and 3. In each VBS, the first orbital is the
bonding orbital and the second orbital is the antibonding orbital. There are
no core orbitals and no virtual orbitals. Therefore, all excitations in this
problem are in the valence.

The one-electron and two-electron integrals below use a real orbital basis.
This basis is already optimized for the perfect-pairing reference. Therefore,
the orbital degrees of freedom are stationary for these integrals. You can use
all results of orbital stationarity that the source formulation derives. You
must find only the electronic degrees of freedom. The two-electron integrals
use chemists' notation,
$(pq|rs)=\int dx_1 dx_2\, \phi_p(x_1)\phi_q(x_1)|x_1-x_2|^{-1}\phi_r(x_2)\phi_s(x_2)$.
The two-electron table gives only the elements that are unique under the
eightfold symmetry of this notation. All values are in hartree.

One-electron integrals $h_{pq}$:

```
h(0,0) = -1.1114074682
h(0,1) = -0.0988634861
h(0,2) = +0.0394997191
h(0,3) = +0.0276626291
h(1,1) = -1.0016712982
h(1,2) = +0.0276223721
h(1,3) = +0.0323783079
h(2,2) = -1.1566175779
h(2,3) = -0.0914760389
h(3,3) = -1.0021521670
```

Two-electron integrals $(pq|rs)$:

```
(00|00) = +0.5086943889
(00|01) = +0.0007761246
(00|02) = -0.0039529506
(00|03) = +0.0240128970
(00|11) = +0.5193065462
(00|12) = -0.0190798470
(00|13) = +0.0107169190
(00|22) = +0.1659122228
(00|23) = +0.0448520167
(00|33) = +0.1705764890
(01|01) = +0.2588967397
(01|02) = -0.0146179188
(01|03) = +0.0027594542
(01|11) = +0.0016634764
(01|12) = +0.0005507129
(01|13) = +0.0173057410
(01|22) = +0.0497530189
(01|23) = +0.0227179403
(01|33) = +0.0528800876
(02|02) = +0.0021644804
(02|03) = -0.0003519105
(02|11) = -0.0043252094
(02|12) = +0.0011509838
(02|13) = -0.0014063378
(02|22) = +0.0027098461
(02|23) = -0.0154139659
(02|33) = +0.0023019345
(03|03) = +0.0034703934
(03|11) = +0.0243312593
(03|12) = -0.0020915451
(03|13) = +0.0018027476
(03|22) = -0.0193959713
(03|23) = +0.0055565438
(03|33) = -0.0201738379
(11|11) = +0.5356642676
(11|12) = -0.0198276503
(11|13) = +0.0111388328
(11|22) = +0.1694444468
(11|23) = +0.0469793698
(11|33) = +0.1742687371
(12|12) = +0.0031582938
(12|13) = -0.0002348973
(12|22) = +0.0231282609
(12|23) = -0.0044870212
(12|33) = +0.0232195201
(13|13) = +0.0030052328
(13|22) = +0.0009517892
(13|23) = +0.0170754010
(13|33) = +0.0011381886
(22|22) = +0.5231507138
(22|23) = +0.0021499098
(22|33) = +0.5338118313
(23|23) = +0.2476679337
(23|33) = +0.0039310041
(33|33) = +0.5535510976
```

Calculate the perfect-pairing reference for this valence. Find the VBS gaps
that make its energy stationary. Then correct this reference with the complete
second-order Epstein-Nesbet treatment of its low-lying valence excitations, as
the source formulation defines it. Use the excited states of the source
formulation. A single electron transfer connects one orbital of VBS A and one
orbital of VBS B. For each such orbital pair, use the two combinations of
transfer states that the source defines. Do not use the two individual
transfer states. Include all valence excitations that the method permits, and
no other excitations.

Give the second-order valence correction to the perfect-pairing energy. Give
it in millihartree, as one number rounded to three decimal places. The
correction is negative because it decreases the energy. Give only the
correction, not the corrected total energy.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal number inside
<final_answer>...</final_answer>, even if the value is approximate or you are
unsure.

Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
  Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
  lines.
- Keep short (a few hundred words). Show only the few scalars that determine
  the final number.
- Do not paste the input matrices, full coefficient vectors, per-iteration
  paths, or per-fold candidate tables.

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

Perfect-pairing occupation numbers and orbital energies.

Goal
----
A valence has two valence-bond subsystems, and their gaps

are given. Return the pair amplitudes, the four orbital occupation numbers and

the four perfect-pairing orbital energies.

```python
def pp_occupations_and_orbital_energies(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
    w_a: float,
    w_b: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Return the pair amplitudes, occupation numbers and orbital energies.

    There are four spatial orbitals, with indices 0..3. Orbitals 0 and 1 are
    the bonding and antibonding orbitals of VBS A. Orbitals 2 and 3 are the
    bonding and antibonding orbitals of VBS B. There are no core orbitals and
    no virtual orbitals. ``h[p][q]`` is the one-electron integral.
    ``v[p][q][r][s]`` is the two-electron integral in chemists' notation
    ``(pq|rs)``. All orbitals are real. ``w_a`` and ``w_b`` are the VBS gaps
    of A and B. The occupation numbers are in the range ``0 < n < 2``, and
    their sum in each VBS is 2.

    Definitions. Write ``alpha`` for a VBS (A or B) and ``w_alpha`` for its
    gap. Its orbitals are ``alpha_0`` (bonding) and ``alpha_1``
    (antibonding).

    - Integral combinations: ``J_pq = (pp|qq)``, ``K_pq = (pq|qp)``,
      ``L_pq = (pq|pq)`` and ``G_pq = 2 J_pq - K_pq``.
    - Pair amplitude: ``eta_alpha = sqrt(w_alpha**2 + 1)``.
    - Occupation numbers: ``n(alpha_0) = 1 + w_alpha / eta_alpha`` and
      ``n(alpha_1) = 1 - w_alpha / eta_alpha``.
    - Perfect-pairing orbital energy of orbital ``p`` in VBS ``alpha``:
      ``eps_p = h_pp + L_pp / 2 + (1/2) * sum_q G_pq * n_q``. The sum is
      over the two orbitals ``q`` of the other VBS only. It does not
      include ``p`` or the partner orbital of ``p``.

    Expected return: a tuple of ten floats
    ``(eta_a, eta_b, n0, n1, n2, n3, eps0, eps1, eps2, eps3)``. ``eta_a`` and
    ``eta_b`` are the pair amplitudes of A and B. ``n0..n3`` are the
    occupation numbers of orbitals 0..3. ``eps0..eps3`` are the
    perfect-pairing orbital energies of orbitals 0..3. Raise ``ValueError``
    if ``h`` or ``v`` does not have the four-orbital shape.
    """
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

### Step 2

Perfect-pairing reference energy and gap gradient.

Goal
----
For given valence-bond-subsystem gaps, return the

perfect-pairing reference energy and the gradient of this energy with respect

to each gap.

```python
def pp_reference_energy(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
    w_a: float,
    w_b: float,
) -> tuple[float, float, float]:
    """Return the perfect-pairing energy and its gradient for each VBS gap.

    The orbital labels, the integral conventions and the meaning of ``w_a``
    and ``w_b`` are the same as in the preceding step. The energy is only the
    electronic energy of the perfect-pairing reference. Do not include the
    nuclear repulsion. Calculate the gradients at fixed integrals, with
    respect to ``w_a`` and to ``w_b``.

    Definitions. Use ``J``, ``K``, ``L``, ``G``, ``eta``, ``n`` and ``eps``
    as the preceding step defines them. The perfect-pairing electronic
    energy is

    ``E = sum_p eps_p * n_p
    - sum_alpha L(alpha_0, alpha_1) * sqrt(n(alpha_0) * n(alpha_1))
    - (1/2) * sum'_{p<q} G_pq * n_p * n_q``.

    The first sum is over all four orbitals. The second sum is over the two
    VBS. The primed sum is over the orbital pairs ``p < q`` that are in
    different VBS; it excludes the pairs (0, 1) and (2, 3). ``grad_a`` and
    ``grad_b`` are the total derivatives ``dE/dw_a`` and ``dE/dw_b``. They
    include the dependence of ``eta``, ``n`` and ``eps`` on the gaps.

    Expected return: a tuple ``(energy, grad_a, grad_b)`` of three floats.
    Raise ``ValueError`` if ``h`` or ``v`` does not have the four-orbital
    shape.
    """
    return (0.0, 0.0, 0.0)
```

### Step 3

Stationary valence-bond-subsystem gaps.

Goal
----
Find the VBS gaps that make the perfect-pairing reference

energy stationary. Return these gaps and the reference energy at these gaps.

```python
def stationary_vbs_gaps(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, float, float]:
    """Return the stationary VBS gaps and the perfect-pairing energy there.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The integrals use an orbital basis that already makes
    the perfect-pairing energy stationary with respect to orbital rotations.
    Therefore, you must find only the gaps. The stationary gaps are the
    gaps at which both gradients of the preceding step are zero. Return the
    solution with both
    gaps positive. In this solution, the bonding orbital of each VBS has the
    larger occupation. Converge the gaps to at least ten decimal places.

    Expected return: a tuple ``(w_a, w_b, energy)`` of three floats, with
    ``w_a > 0`` and ``w_b > 0``. ``energy`` is the perfect-pairing
    electronic energy at these gaps. Raise ``ValueError`` if ``h`` or ``v``
    does not have the four-orbital shape.
    """
    return (0.0, 0.0, 0.0)
```

### Step 4

Generalized Fock matrix of the perfect-pairing reference.

Goal
----
Return the generalized Fock matrix of the perfect-pairing

reference at its stationary gaps.

```python
def generalized_fock_elements(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, ...]:
    """Return the generalized Fock matrix at the stationary gaps.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The gaps are the stationary gaps for the given
    integrals. They are not free parameters. Use the index order
    ``f[p][q]``, with ``p`` the row. The diagonal direct
    density-matrix element is zero. The diagonal pair element is equal to the
    occupation number of that orbital.

    Definitions. Use ``eta`` and ``n`` from the first step, at the
    stationary gaps.

    - Direct elements: ``D_rp = n_r * n_p`` when ``r`` and ``p`` are in
      different VBS. ``D_rp = 0`` when ``r`` and ``p`` are in the same VBS,
      and ``D_pp = 0``.
    - Pair elements: ``P_pp = n_p``. Inside a VBS,
      ``P(alpha_0, alpha_1) = P(alpha_1, alpha_0) = -1 / eta_alpha``. All
      other ``P_rp`` are zero.
    - Generalized Fock matrix:
      ``f[p][q] = h_pq * n_p
      + (1/2) * sum_r (2 (rr|pq) - (rq|pr)) * D_rp
      + sum_r (rp|rq) * P_rp``,
      with the sums over all four orbitals ``r``. The occupation number and
      the density-matrix elements carry the row index ``p``, not ``q``.
      Therefore, ``f`` is not symmetric in general.

    Expected return: a tuple of 16 floats. These are all elements of the
    matrix in row order: ``f[0][0]``, ``f[0][1]``, ``f[0][2]``, ``f[0][3]``,
    ``f[1][0]``, and so on to ``f[3][3]``. Raise ``ValueError`` if ``h`` or
    ``v`` does not have the four-orbital shape.
    """
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

### Step 5

Pairing and double-counting intermediates.

Goal
----
At the stationary gaps, return the two-body intermediates

that the excitation energies of the valence use many times.

```python
def pairing_intermediates(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, ...]:
    """Return the open-shell-singlet and double-counting intermediates.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The gaps are the stationary gaps for the given
    integrals. Give the open-shell-singlet intermediate for each orbital pair
    in the return list. Give the double-counting intermediates for the
    subsystem pair (A, B).

    Expected return: a tuple of twelve floats
    ``(t_01, t_23, t_02, t_03, t_12, t_13, g_dd, g_plain, g_a_at_2,
    g_a_at_3, g_b_at_0, g_b_at_1)``:

    - ``t_pq`` is the open-shell-singlet intermediate for orbitals ``p`` and
      ``q``.
    - ``g_dd`` is the double-counting intermediate for (A, B) with a gap
      weight for both subsystems.
    - ``g_plain`` is the double-counting intermediate for (A, B) with no gap
      weight.
    - ``g_a_at_2`` and ``g_a_at_3`` are the intermediates of VBS A with one
      gap weight, at orbitals 2 and 3.
    - ``g_b_at_0`` and ``g_b_at_1`` are the intermediates of VBS B with one
      gap weight, at orbitals 0 and 1.

    Definitions. Use ``J``, ``K``, ``L``, ``G``, ``eta`` and the gaps ``w``
    from the first step, at the stationary gaps. Write ``x_A = w_A / eta_A``
    and ``x_B = w_B / eta_B``.

    - Same VBS (``t_01`` and ``t_23``):
      ``t_pq = J_pq + K_pq - L_pp / 2 - L_qq / 2``.
    - Different VBS (``t_02``, ``t_03``, ``t_12``, ``t_13``):
      ``t_pq = J_pq + K_pq - L_pp / 2 - L_qq / 2 - G_pq / 2``.
    - ``g_dd = (1/2) * x_A * x_B * (G_02 - G_03 - G_12 + G_13)``.
    - ``g_plain = (1/2) * (G_02 + G_03 + G_12 + G_13)``. Each term has one
      orbital of A and one orbital of B. (One printed version of this
      definition has ``G_11`` as the last term. That is a misprint. Use
      ``G_13``.)
    - ``g_a_at_p = (1/2) * x_A * (G_0p - G_1p)`` for ``p`` = 2, 3.
    - ``g_b_at_p = (1/2) * x_B * (G_2p - G_3p)`` for ``p`` = 0, 1.

    Raise ``ValueError`` if ``h`` or ``v`` does not have the four-orbital
    shape.
    """
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

### Step 6

Single electron-transfer channel.

Goal
----
The two orbital labels are given. For these labels, find the

electron-transfer combination that the source method uses. Return its squared

norm, its excitation energy and the square of its reference coupling.

```python
def electron_transfer_channel(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
    mu: int,
    nu: int,
) -> tuple[float, float, float]:
    """Return the norm, excitation energy and squared coupling of one channel.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The gaps are the stationary gaps for the given integrals.
    ``mu`` selects the orbital of VBS A: 0 for its bonding orbital, 1 for its
    antibonding orbital. ``nu`` selects the orbital of VBS B in the same way.
    For this ``(mu, nu)``, there are two orthogonal combinations of the two
    electron-transfer states. Give the values for the combination
    ``|plus>`` that the definitions below give. Measure the
    excitation energy from the perfect-pairing reference energy. Give the
    excitation energy and the squared coupling for the combination after you
    normalize it to unity.

    Definitions. ``|vac>`` is the empty state. ``a+_{p,s}`` creates an
    electron with spin ``s`` in orbital ``p``. Use ``eta``, ``n`` and the
    gaps ``w`` from the first step, at the stationary gaps.

    - Pair creator: ``P+_p = a+_{p,up} a+_{p,down}``.
    - Normalized open-shell-singlet creator for ``p != q``:
      ``S+_pq = (a+_{p,up} a+_{q,down} + a+_{q,up} a+_{p,down}) / sqrt(2)``.
      These operators have an even number of fermion operators, so their
      order in a product does not change the state.
    - Perfect-pairing reference: ``|ref>`` is proportional to
      ``(P+_0 + (w_A - eta_A) P+_1) (P+_2 + (w_B - eta_B) P+_3) |vac>``,
      normalized to unity. Its energy ``<ref|H|ref>`` is the energy of the
      second step.
    - Write ``a = mu``, ``a' = 1 - mu`` (orbitals of A) and ``b = 2 + nu``,
      ``b' = 3 - nu`` (orbitals of B). The two electron-transfer states are
      ``|X1> = S+_{a b} P+_{b'} |vac>`` (one electron from A to B) and
      ``|X2> = S+_{a b} P+_{a'} |vac>`` (one electron from B to A). They
      are normalized and orthogonal.
    - With ``c1 = sqrt(n_a * n_b')`` and ``c2 = sqrt(n_a' * n_b)``, the two
      combinations are
      ``|minus> = (c1 |X1> - c2 |X2>) / sqrt(2)`` and
      ``|plus> = (c2 |X1> + c1 |X2>) / sqrt(2)``. Return the values for
      ``|plus>``.
    - Squared norm: ``<plus|plus> = <minus|minus> = (c1**2 + c2**2) / 2``.
      This is equal to ``1 + (-1)**(mu + nu + 1) * w_A * w_B / (eta_A * eta_B)``.
    - With ``|u> = |plus> / sqrt(<plus|plus>)``, the excitation energy is
      ``<u|H|u> - <ref|H|ref>`` and the squared coupling is
      ``<ref|H|u>**2``. ``H`` is the electronic Hamiltonian of the given
      integrals, without nuclear repulsion.

    Expected return: a tuple ``(norm, excitation_energy, coupling_squared)``
    of three floats. ``norm`` is the squared norm of the combination before
    normalization. ``excitation_energy`` is positive. ``coupling_squared`` is
    the square of the coupling, so it is not negative. Raise ``ValueError``
    if ``mu`` or ``nu`` is not 0 or 1, or if ``h`` or ``v`` does not have
    the four-orbital shape.
    """
    return (0.0, 0.0, 0.0)
```

### Step 7

Remaining valence channels grouped by seniority.

Goal
----
Return the second-order Epstein-Nesbet energy contributions

of all valence excitations other than the single electron transfers. Group

them by the seniority of the excited state.

```python
def remaining_valence_channels(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, float, float]:
    """Return the second-order contributions without transfers, by seniority.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The gaps are the stationary gaps for the given
    integrals. Use all valence excitations of the perfect-pairing reference,
    except the single electron transfers between the two subsystems. A
    different step calculates those transfers. Use the orthonormal set of
    excited states that the definitions below give, because the second-order
    Epstein-Nesbet correction changes with that selection. Add the
    second-order contributions of these excitations into three groups: excited
    states of seniority zero, two and four.

    Definitions. Use ``|vac>``, ``P+_p``, ``S+_pq``, ``H`` and ``|ref>`` as
    the preceding step defines them, and ``eta`` and the gaps ``w`` from the
    first step, at the stationary gaps. For VBS ``alpha`` with orbitals
    ``alpha_0`` and ``alpha_1``, define two pair factors:

    - ground factor ``R_alpha = P+(alpha_0) + (w_alpha - eta_alpha) P+(alpha_1)``;
    - swapped factor ``Q_alpha = P+(alpha_0) + (w_alpha + eta_alpha) P+(alpha_1)``.

    Then ``|ref>`` is proportional to ``R_A R_B |vac>``. The excited states
    of this step are the states below, each normalized to unity:

    - seniority zero: the swaps ``Q_A R_B |vac>`` and ``R_A Q_B |vac>``; the
      double swap ``Q_A Q_B |vac>``; the two whole-pair transfers
      ``P+_2 P+_3 |vac>`` and ``P+_0 P+_1 |vac>``;
    - seniority two: the splits ``S+_01 R_B |vac>`` and ``R_A S+_23 |vac>``;
      the swap-plus-splits ``Q_A S+_23 |vac>`` and ``S+_01 Q_B |vac>``;
    - seniority four: the double split ``S+_01 S+_23 |vac>`` and the
      complementary double split
      ``(S+_02 S+_13 - S+_03 S+_12) |vac> / sqrt(3)``.

    The second-order Epstein-Nesbet contribution of one normalized excited
    state ``|u>`` is
    ``-<ref|H|u>**2 / (<u|H|u> - <ref|H|ref>)``.

    Expected return: a tuple ``(sen0, sen2, sen4)`` of three floats in
    millihartree. Each value is the sum of the second-order contributions of
    the excitations with that seniority. Each value is less than or equal to
    zero. Raise ``ValueError`` if ``h`` or ``v`` does not have the
    four-orbital shape.
    """
    return (0.0, 0.0, 0.0)
```

### Step 8

Second-order valence correction to perfect pairing

Goal
----
Add the single electron-transfer

channels and the remaining valence channels to get the complete second-order

Epstein-Nesbet correction to the perfect-pairing reference. Give the result in

millihartree.

```python
def en2_valence_correction(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> float:
    """Return the complete second-order valence correction in millihartree.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The gaps are the stationary gaps for the given
    integrals. Include all valence excitations of the perfect-pairing
    reference. Get the single electron transfers between the two subsystems
    from the channel step. Get all other excitations from the grouped step.
    Do not count an excitation two times, and do not omit an excitation. The
    grouped step gives its values in millihartree. The transfer channels do
    not. The contribution of one transfer channel is
    ``-coupling_squared / excitation_energy`` in hartree. Give the total in
    millihartree, not in hartree.

    Expected return: one float, the correction in millihartree, rounded to
    three decimal places. The value is negative because the correction
    decreases the energy. Raise ``ValueError`` if ``h`` or ``v`` does not
    have the four-orbital shape.
    """
    return 0.0
```
