# Exact spin-adapted triplet-pair projections for the dark state of octatetraene in the Pariser-Parr-Pople model

## Background

Linear polyenes are the textbook case of a correlated pi-electron system. Their lowest excited
singlet, the 2^1Ag state, is optically dark and lies below the bright 1^1Bu state once electron
correlation is included; its ordering was the first striking failure of the independent-electron
picture. In the covalent (valence-bond) language of the Pariser-Parr-Pople model the dark state is
dominated by configurations with one electron per carbon atom and is read as two triplet excitons,
each a neutral soliton-antisoliton pair, bound together and mixed with charge-transfer configurations
in which an electron has moved from one ethylene unit to a neighbour. The weight of the triplet-pair
component is the quantity that decides whether the dark state can serve as the intermediate of
intramolecular singlet fission, the process in which one singlet exciton splits into two triplets.

Two triplets of spin one couple to a total spin of zero, one or two; only the singlet-coupled
combination can appear in a singlet state and only the quintet-coupled one in the lowest quintet.
When a chain is cut into two segments, each segment carries its own triplet eigenstates, the members
of a family that behaves like a particle in a box, and a triplet pair on the whole chain is a tensor
product of a segment triplet on the left and a segment triplet on the right. Products belonging to
different cuts of the chain are not orthogonal to each other, so a population defined as the weight of
a state inside the space they span has to be taken with respect to their overlap matrix; the
population then measures the projection onto the span and does not depend on how the span is
orthonormalised. Whether the products are formed from the M = 0 components alone or from the
spin-coupled combinations is a separate question, because the overlap between two M = 0 products on
different cuts contains both the singlet and the quintet channel.

For a short chain every eigenstate of the pi-electron Hamiltonian and of each segment is accessible
by exact diagonalization in a determinant basis, with the total spin, the spatial parity and the
alternancy (particle-hole) symmetry resolving every level, including degenerate ones. Fermionic
statistics enter the tensor products through the ordering of the creation operators, and the three
spin components of a segment triplet are tied together by the ladder operators. The task compares,
on this exact footing, the population that the source of this task defines through M = 0 products
with the population that the spin-adapted products define, in the source's own product basis and in
its systematic completions.

## Problem

The lowest excited singlet of linear polyenes, the dark 2^1Ag state, has been read for fifty years
as a bound pair of triplet excitons mixed with charge-transfer configurations, and the weight of the
triplet-pair component decides whether the state can act as an intermediate in intramolecular
singlet fission. The source of this task quantifies that weight for the pi-electron Pariser-Parr-Pople
(PPP) model of polyene chains. It partitions the
chain of N_d ethylene dimers into a left segment of m dimers and a right segment of N_d - m dimers,
takes on each segment the members of the lowest covalent triplet family, forms the tensor products of
the M = 0 spin components T_j(m) x T_1(N_d - m), orthonormalises these products by Loewdin's symmetric
procedure, and defines the triplet-pair population of the dark state as three times the sum of its
squared projections on the orthonormalised products, the factor three being the one that converts the
projection on a single T_0 x T_0 product into the projection on the singlet-coupled pair
(T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3.

Your task is to treat the same model exactly for octatetraene (N = 8 carbon atoms, N_d = 4 dimers),
where every state of the full chain and of every segment is available by exact diagonalization, and to
replace the source's shortcut by the exact spin-adapted projection. Build the PPP Hamiltonian in a
stated geometry and determinant basis, label the eigenstates by spin, spatial parity and alternancy
(particle-hole) symmetry, identify the dark state, the bright state, the covalent 1^1Bu+ state, the
lowest triplet and the lowest quintet; construct the covalent triplet families of the isolated
segments and their three spin components; form the site-ordered tensor products of segment triplets in
the T_0 x T_0 form and in the singlet-coupled and quintet-coupled forms; compute, for the source's own
product basis and for two nested extensions of it, both the source's prescription (span of the
T_0 x T_0 products, times 3 or 3/2) and the exact population (span of the spin-coupled products);
establish that the two agree for a single product and disagree as soon as products on different
partitions overlap, and explain why; show what the prescription does in the complete
covalent basis at strong coupling; and report, as the final answer, the exact spin-adapted population
of the dark state in the complete covalent triplet-pair basis at U = 8 eV.

## Conventions that fix every number

Model: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers
with dimer n formed by the sites 2n and 2n+1, one pi electron per site, and the PPP Hamiltonian
H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2)
+ sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for
even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential
V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom =
e^2/(4 pi eps_0) and eps the relative permittivity. The source leaves the carbon skeleton unstated;
here it is fixed: site 0 at the origin, bond i (from site i to site i+1) of length r_double for even i
and r_single for odd i, making the angle +phi (even i) or -phi (odd i) with the chain axis,
phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting
planar distances.

Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over
occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the
C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a
determinant is i_up D_down + i_down; a determinant is the string of all up creation operators
(ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so
that nearest-neighbour hopping matrix elements carry no fermionic sign.

Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin
S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site
reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation
complement J maps a determinant to the determinant with every occupation inverted (empty <->
occupied for each spin); both permutations, with the constant sign of the operator reordering,
commute with H. The labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its
P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster
of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors
are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by
(S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with
j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose
M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet
with (p, j) = (+1, +1), the dark state; 1^1Bu- = the lowest singlet with (-1, -1), the optically
bright state; 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1);
1^5Ag+ = the lowest quintet with (+1, +1).

Segments and triplet families: a partition of the N_d dimers into a left segment of m dimers (sites
0 to 2m-1) and a right segment of N_d - m dimers (sites 2m to N-1); each segment is treated as an
isolated open PPP chain with the same parameters and the distances of its own piece of the geometry;
its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels
resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent
triplets; the spin components of a triplet are related by the ladder operators,
T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2.

Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the
creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...)
followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain
determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled
pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0)
pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6. Pair bases: 'paper' =
{T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}, the source's basis; 'family' =
{T_j(m) x T_k(N_d - m)} over all family members on both segments; 'covalent' = every pair of covalent
triplets of the two segments, all of them, taken from each segment's full diagonalization (n_states
applies to the full chain only; the set is linearly dependent, and the span is defined by the singular
values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0
prescription is the squared norm of the projection of Psi onto the span of the T0T0 products
multiplied by 3 for a singlet and by 3/2 for a quintet (the source's rule); the exact spin-adapted
population is the squared norm of the projection of Psi onto the span of the singlet-coupled products
(singlet states) or of the quintet-coupled products (quintet state). The projection onto a span does
not depend on how the span is orthonormalised, so Loewdin's procedure and any other one give the same
population.

Reference parameters: N = 8, U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12,
r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100 lowest states of
the half-filled S_z = 0 sector searched for the named states.

## Electronic structure

Build the Ohno matrix and the Hamiltonian, diagonalize the half-filled S_z = 0 sector and label the
lowest states; report the excitation energies of 2^1Ag+, 1^1Bu-, 1^1Bu+, 1^3Bu and
1^5Ag+ together with the triplet-pair binding energy E(1^5Ag+) - E(2^1Ag+), and the number of covalent
triplets below the dark state; say which of the dark and bright states lies lower at U = 8 eV.

## Triplet families of the segments

For the one-, two- and three-dimer segments report the family excitation energies, each above its
own segment's ground state, and their parities relative to that same ground state;
say what the parities of the family members are and why.

## Products and metrics

For the product m = 1, j = 1, k = 1 report |<T0T0|2^1Ag+>|^2 and
|<1TT|2^1Ag+>|^2, |<T0T0|1^5Ag+>|^2 and |<2TT|1^5Ag+>|^2, |<1TT|1^1Ag+>|^2 and the diabatic
triplet-pair energy <1TT|H|1TT> - E_0; state the two single-product identities. Report the largest
and smallest Gram eigenvalues of the singlet-coupled 'paper' and 'family' sets (the T0T0 extremes may
be added), and explain, through the decomposition of a T0T0 product into its
S = 0 and S = 2 parts, why the two metrics differ once partitions overlap.

## Populations

Report, at U = 8 eV, the T0T0 prescription and the exact spin-adapted population of 2^1Ag+ in the
three bases and of 1^1Bu+ and 1^5Ag+ in the source's basis and in the complete covalent basis; repeat
the 2^1Ag+ values in those two bases at U = 4 and 14 eV and the 1^1Bu+ covalent-basis values at
U = 14 eV. Explain why the exact populations of the three bases are nested, in which direction the
prescription errs for 2^1Ag+ and for the quintet, and what happens to the prescription in the
covalent basis at U = 14 eV. Check the construction against the source's four-site limit: the
'paper' population of 2^1Ag+ for N = 4 at U -> 0 and at U = 8 eV, and say why the prescription and the
exact value coincide there.

## Audit

Run the complete chain with the reference parameters for U = 8, 4 and 14 eV and assemble the table
with one row per Coulomb parameter and the columns [exact and prescription populations for the bases
(covalent, family, paper) of 2^1Ag+, of 1^1Bu+ and of 1^5Ag+ (18 columns); the six energies; V_01;
V_{0,N-1}; the mean diagonal element of the Hamiltonian; the number of covalent triplets below the dark
state; the first and last family energies of the three-dimer segment and <S^z_0> of its T_1; the four
entries [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0] of the product
m = 1, j = 1, k = 1; and for each basis its size, largest and smallest singlet-coupled Gram eigenvalue
and rank]; the head element, the exact covalent-basis population of 2^1Ag+ at U = 8 eV, is the final
answer.

## What to report

Report, as the final answer, the exact spin-adapted population of the 2^1Ag+ state of octatetraene in
the complete covalent triplet-pair basis at U = 8 eV with the conventions and reference parameters
above, to five significant figures.
Report every other population, excitation energy, family energy and product-basis quantity listed above to at least four significant figures (three for the single-product entries, the Gram extremes and the four-site checks).

Your reasoning should also report, as evidence that the chain was executed: the five
excitation energies and the binding energy; the number of covalent triplets below the dark state and
which of the dark and bright states lies lower; the family energies and parities of the three segments;
the six single-product values for the product named above; the two single-product identities; the Gram extremes of the 'paper' and
'family' singlet-coupled sets; the prescription and exact
populations named in the Populations section (2^1Ag+ in the three bases and 1^1Bu+ and 1^5Ag+ in the
source's and the covalent bases at U = 8 eV; 2^1Ag+ in the source's and the covalent bases at U = 4
and 14 eV; 1^1Bu+ in the covalent basis at U = 14 eV), and the four-site check values. Say, in a sentence or two each, why the prescription
and the exact projection coincide for one product and separate for a basis; why the parities alternate
along a family; why the prescription and the exact value coincide for N = 4; why the exact
populations grow with the basis; in which direction the prescription errs for 2^1Ag+ and for the
quintet and what its value in the covalent basis at U = 14 eV shows; and what the exact numbers say
about the source's statement that its population is a lower bound and about its extrapolated long-chain
population.

Report, with a citation to the source, its model, parameters and numerical method; the solvable
two-particle model on which it motivates its definition, and what it establishes about that model's
product basis; its definition of the triplet-pair population and its statement about the bound; its
main results for the dark state, the four-site chain and the quintet; and the context it builds on.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
Output Format Requirements:
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

coulomb_matrix

Goal
----
Return the (N, N) symmetric matrix of the Ohno interaction V_ij between the sites of the polyene geometry, with the on-site Coulomb parameter U on the diagonal. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError if N is not an even integer >= 2, if U < 0, eps <= 0, a bond length is not positive or angle_deg is outside (0, 180].

```python
def coulomb_matrix(N: int, U: float, eps: float, r_double: float, r_single: float, angle_deg: float) -> np.ndarray:
    """Return the (N, N) symmetric matrix of the Ohno interaction V_ij between the sites of the
    polyene geometry, with the on-site Coulomb parameter U on the diagonal.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).

    Returns
    -------
    An (N, N) float64 array with V_ii = U.

    Raises
    ------
    ValueError
        If N is not an even integer >= 2, if U < 0, eps <= 0, a bond length is not
        positive or angle_deg is outside (0, 180].
    """
    return None
```

### Step 2

ppp_hamiltonian

Goal
----
Return the dense (D, D) matrix of the PPP Hamiltonian in the ordered determinant basis of the (n_up, n_down) sector, D = C(N, n_up) C(N, n_down), in eV. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for an invalid geometry or model parameter, or if n_up or n_down is not an integer between 0 and N.

```python
def ppp_hamiltonian(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_up: int, n_down: int) -> np.ndarray:
    """Return the dense (D, D) matrix of the PPP Hamiltonian in the ordered determinant basis
    of the (n_up, n_down) sector, D = C(N, n_up) C(N, n_down), in eV. Model and conventions:
    an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2
    ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site,
    described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma}
    c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i
    - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i
    (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij
    = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV
    angstrom = e^2/(4 pi eps_0) and eps the relative permittivity.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_up : int
        Number of up-spin electrons (0 <= n_up <= N).
    n_down : int
        Number of down-spin electrons (0 <= n_down <= N).

    Returns
    -------
    A (D, D) float64 symmetric array in eV.

    Raises
    ------
    ValueError
        For an invalid geometry or model parameter, or if n_up or n_down is not an
        integer between 0 and N.
    """
    return None
```

### Step 3

symmetry_labels

Goal
----
Return the (n_states, 4) array whose row k, for the n_states lowest eigenstates of the half-filled S_z = 0 sector in ascending energy, is [E_k - E_0, S(S+1), p_k p_0, j_k j_0]. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid model parameters or if n_states is not an integer between 1 and the sector dimension.

```python
def symmetry_labels(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    """Return the (n_states, 4) array whose row k, for the n_states lowest eigenstates of the
    half-filled S_z = 0 sector in ascending energy, is [E_k - E_0, S(S+1), p_k p_0, j_k
    j_0].

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.

    Returns
    -------
    An (n_states, 4) float64 array [E_k - E_0 (eV), S(S+1), p_k p_0, j_k j_0].

    Raises
    ------
    ValueError
        For invalid model parameters or if n_states is not an integer between 1 and the
        sector dimension.
    """
    return None
```

### Step 4

dark_state_energies

Goal
----
Return the (6,) array [E(2^1Ag+) - E_0, E(1^1Bu-) - E_0, E(1^1Bu+) - E_0, E(1^3Bu) - E_0, E(1^5Ag+) - E_0, E(1^5Ag+) - E(2^1Ag+)] of excitation energies (eV) identified among the n_states lowest eigenstates of the half-filled S_z = 0 sector, the last entry being the triplet-pair binding energy. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if n_states < 2, if the ground state is not a covalent totally symmetric singlet, or if one of the six states is not among the n_states lowest eigenstates.

```python
def dark_state_energies(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    """Return the (6,) array [E(2^1Ag+) - E_0, E(1^1Bu-) - E_0, E(1^1Bu+) - E_0, E(1^3Bu) -
    E_0, E(1^5Ag+) - E_0, E(1^5Ag+) - E(2^1Ag+)] of excitation energies (eV) identified
    among the n_states lowest eigenstates of the half-filled S_z = 0 sector, the last entry
    being the triplet-pair binding energy.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.

    Returns
    -------
    A (6,) float64 array of energies in eV.

    Raises
    ------
    ValueError
        For invalid parameters, if n_states < 2, if the ground state is not a covalent
        totally symmetric singlet, or if one of the six states is not among the n_states
        lowest eigenstates.
    """
    return None
```

### Step 5

triplet_family

Goal
----
Return the (m, 2m + 2) array describing the triplet family T_j(m), j = 1, ..., m, of an isolated open subchain of m dimers (2m sites): row j-1 is [E_j - E_0^sub, p_j p_0^sub, <S^z_0>, ..., <S^z_{2m-1}>], the excitation energy above the subchain ground state, the site-reversal label relative to the subchain ground state and the spin-density profile of the M = +1 component T_{+1} = S+ T_0 / sqrt2. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if m is not a positive integer, or if the subchain has fewer than m covalent triplets.

```python
def triplet_family(m: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float) -> np.ndarray:
    """Return the (m, 2m + 2) array describing the triplet family T_j(m), j = 1, ..., m, of an
    isolated open subchain of m dimers (2m sites): row j-1 is [E_j - E_0^sub, p_j p_0^sub,
    <S^z_0>, ..., <S^z_{2m-1}>], the excitation energy above the subchain ground state, the
    site-reversal label relative to the subchain ground state and the spin-density profile
    of the M = +1 component T_{+1} = S+ T_0 / sqrt2.

    Parameters
    ----------
    m : int
        Number of dimers of the (left) segment (1 <= m; for products 1 <= m <= N/2 - 1).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).

    Returns
    -------
    An (m, 2m + 2) float64 array.

    Raises
    ------
    ValueError
        For invalid parameters, if m is not a positive integer, or if the subchain has
        fewer than m covalent triplets.
    """
    return None
```

### Step 6

triplet_pair_product

Goal
----
For the single triplet-pair product built from the family member T_{j+1}(m) of the left subchain of m dimers and T_{k+1}(N_d - m) of the right subchain (0-based j, k), return the (7,) array [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<T0T0|1^5Ag+>|^2, |<2TT|1^5Ag+>|^2, |<1TT|1^1Bu+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0], the target states being identified among the n_states lowest eigenstates of the full chain. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if m is not an integer between 1 and N_d - 1, or if j or k is outside its family (0 <= j < m, 0 <= k < N_d - m).

```python
def triplet_pair_product(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int, m: int, j: int, k: int) -> np.ndarray:
    """For the single triplet-pair product built from the family member T_{j+1}(m) of the left
    subchain of m dimers and T_{k+1}(N_d - m) of the right subchain (0-based j, k), return
    the (7,) array [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<T0T0|1^5Ag+>|^2,
    |<2TT|1^5Ag+>|^2, |<1TT|1^1Bu+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0], the target
    states being identified among the n_states lowest eigenstates of the full chain.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.
    m : int
        Number of dimers of the (left) segment (1 <= m; for products 1 <= m <= N/2 - 1).
    j : int
        Zero-based family index on the left segment (0 <= j < m).
    k : int
        Zero-based family index on the right segment (0 <= k < N/2 - m).

    Returns
    -------
    A (7,) float64 array.

    Raises
    ------
    ValueError
        For invalid parameters, if m is not an integer between 1 and N_d - 1, or if j or
        k is outside its family (0 <= j < m, 0 <= k < N_d - m).
    """
    return None
```

### Step 7

pair_basis_gram

Goal
----
Return the (2, n_b) array of the eigenvalues, in descending order, of the Gram (overlap) matrices of the n_b product states of the requested pair basis ('paper', 'family' or 'covalent'): row 0 for the T0T0 products, row 1 for the singlet-coupled products 1TT. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if N < 4 or if basis_kind is not one of the three names.

```python
def pair_basis_gram(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, basis_kind: str) -> np.ndarray:
    """Return the (2, n_b) array of the eigenvalues, in descending order, of the Gram (overlap)
    matrices of the n_b product states of the requested pair basis ('paper', 'family' or
    'covalent'): row 0 for the T0T0 products, row 1 for the singlet-coupled products 1TT.
    Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0,
    ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi
    electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma}
    t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) +
    sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 +
    delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i;
    Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in
    angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    basis_kind : str
        Pair basis: 'paper', 'family' or 'covalent'.

    Returns
    -------
    A (2, n_b) float64 array of descending Gram eigenvalues.

    Raises
    ------
    ValueError
        For invalid parameters, if N < 4 or if basis_kind is not one of the three names.
    """
    return None
```

### Step 8

triplet_pair_populations

Goal
----
Return the (3, 2) array of triplet-pair populations in the requested pair basis: rows for the states 2^1Ag+, 1^1Bu+ and 1^5Ag+, column 0 the T0T0 prescription (span of the T0T0 products, times 3 for the singlets and 3/2 for the quintet) and column 1 the exact spin-adapted population (span of the singlet-coupled products for the singlets, of the quintet-coupled products for the quintet). Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if N < 4, if basis_kind is not one of the three names, or if a target state is not among the n_states lowest eigenstates.

```python
def triplet_pair_populations(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int, basis_kind: str) -> np.ndarray:
    """Return the (3, 2) array of triplet-pair populations in the requested pair basis: rows
    for the states 2^1Ag+, 1^1Bu+ and 1^5Ag+, column 0 the T0T0 prescription (span of the
    T0T0 products, times 3 for the singlets and 3/2 for the quintet) and column 1 the exact
    spin-adapted population (span of the singlet-coupled products for the singlets, of the
    quintet-coupled products for the quintet).

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.
    basis_kind : str
        Pair basis: 'paper', 'family' or 'covalent'.

    Returns
    -------
    A (3, 2) float64 array of populations.

    Raises
    ------
    ValueError
        For invalid parameters, if N < 4, if basis_kind is not one of the three names,
        or if a target state is not among the n_states lowest eigenstates.
    """
    return None
```

### Step 9

triplet_pair_audit

Goal
----
Return the (len(U_list), 47) audit table with one row per Coulomb parameter of U_list: columns 0 to 17 the populations [exact spin-adapted, T0T0 prescription] for the bases (covalent, family, paper) in that order, for 2^1Ag+ (columns 0 to 5), 1^1Bu+ (6 to 11) and 1^5Ag+ (12 to 17); columns 18 to 23 the six excitation energies of dark_state_energies; 24 V_01; 25 V_{0,N-1}; 26 the mean diagonal element of the half-filled Hamiltonian; 27 the number of covalent triplets below 2^1Ag+; 28 and 29 the first and last family energies of the (N_d - 1)-dimer subchain; 30 <S^z_0> of its T_1; 31 to 34 the entries [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0] of the product m = 1, j = 0, k = 0; 35 to 46 for the bases (covalent, family, paper) the four numbers [n_b, largest and smallest singlet-coupled Gram eigenvalue, rank, the rank being the number of singlet-coupled Gram eigenvalues above 1e-8 of the largest]. The head element [0, 0], the exact spin-adapted covalent-basis population of 2^1Ag+ at U_list[0], is the final answer. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if N < 4 or if U_list is empty.

```python
def triplet_pair_audit(N: int, U_list: list, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    """Return the (len(U_list), 47) audit table with one row per Coulomb parameter of U_list:
    columns 0 to 17 the populations [exact spin-adapted, T0T0 prescription] for the bases
    (covalent, family, paper) in that order, for 2^1Ag+ (columns 0 to 5), 1^1Bu+ (6 to 11)
    and 1^5Ag+ (12 to 17); columns 18 to 23 the six excitation energies of
    dark_state_energies; 24 V_01; 25 V_{0,N-1}; 26 the mean diagonal element of the half-
    filled Hamiltonian; 27 the number of covalent triplets below 2^1Ag+; 28 and 29 the first
    and last family energies of the (N_d - 1)-dimer subchain; 30 <S^z_0> of its T_1; 31 to
    34 the entries [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> -
    E_0] of the product m = 1, j = 0, k = 0; 35 to 46 for the bases (covalent, family,
    paper) the four numbers [n_b, largest and smallest singlet-coupled Gram eigenvalue,
    rank, the rank being the number of singlet-coupled Gram eigenvalues above 1e-8 of
    the largest].

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U_list : list
        Coulomb parameters in eV, one audit row each (non-empty).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.

    Returns
    -------
    A (len(U_list), 47) float64 array; entry [0, 0] is the final answer.

    Raises
    ------
    ValueError
        For invalid parameters, if N < 4 or if U_list is empty.
    """
    return None
```
