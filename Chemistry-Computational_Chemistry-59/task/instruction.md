# Chemistry-Computational_Chemistry-59

## Background

Heterogeneous catalysis is understood, at the atomic scale, as a network of elementary steps in which adsorbed species bind, migrate, break bonds and hand atoms to one another across a surface. What decides whether such a network is useful is rarely the stability of its intermediates alone; it is the height of the barriers between them, because those barriers enter the rate exponentially. Computing a barrier means locating a first-order saddle point of the potential energy surface that separates two adsorbed minima, and the standard way to do that is a double-ended search: fix a reactant geometry and a product geometry, draw a chain of intermediate configurations between them, and relax that chain onto the lowest path connecting the two while forbidding it to collapse into either basin.

The difficulty is that the two end points are not given by nature. A molecule on a surface has many inequivalent adsorption sites and many orientations at each of them, two co-adsorbed fragments multiply those possibilities, and the surface itself repeats the whole arrangement at every symmetry-equivalent position of its lattice. Which reactant geometry pairs with which product geometry, and which atom of the reactant is identified with which atom of the product, are choices rather than data, and different choices give different paths and different barriers. Worse, many of the choices describe a different chemical event altogether: a chain drawn between a poorly matched pair spends its energy carrying a fragment across the surface, or pulling it apart and putting it back together, rather than on the bond rearrangement that was intended, and the barrier it returns can exceed the real one by more than an electronvolt without anything in the calculation signalling that something went wrong.

Automating those choices therefore means more than automating the saddle search. It means generating candidate end states without prior assumptions about binding motifs, establishing a correspondence between the atoms of the two states, recognising when several correspondences describe genuinely different mechanisms and when they are merely relabelings of identical atoms, exploiting the symmetry of the clean surface to place a product on the same patch of surface as its reactant, interpolating across periodic boundaries without severing molecules that ought to remain whole, and ranking the resulting candidates by a criterion cheap enough to apply to thousands of them. Only the handful that survive that cascade justify the cost of a path optimisation. The individual ingredients - graph isomorphism for the atom correspondence, crystallographic symmetry analysis for the placement, interpolation schemes that work in the space of interatomic distances rather than in Cartesian coordinates, and chains of images held together by springs and relaxed onto the minimum energy path - are all standard; the science lies in the order in which they are applied and in what each stage is allowed to discard.

Two practical facts shape any such workflow. The first is that energies and forces must be cheap: machine-learned interatomic potentials fitted to first-principles data, or comparably cheap analytic surrogates, are what make a cascade of this size affordable at all, and the logic of the cascade is indifferent to which of them supplies the numbers. The second is that a purely pairwise description of bonding is not enough when an atom is being handed from one centre to another, because a pair potential lets an atom bond to every neighbour at once and so places no barrier between a reactant and a product that differ only in which centre a hydrogen is attached to; a description in which the energy depends on how many bonds an atom already has is what separates the two states and creates the saddle between them.

## Problem

Calculating the activation barrier of a surface reaction is still, in practice, a task delegated to human intuition: somebody decides which adsorbed geometry is the reactant, which is the product, which atom of the reactant becomes which atom of the product, and how the two are to be placed relative to one another before a double-ended path search is started. Every one of those decisions changes the barrier that comes out, and none of them is easy to automate, because the space of initial and final states is combinatorial and most of it describes the wrong elementary process. A workflow that removes the human from that loop has to generate candidate end states, map their atoms onto one another, reduce the resulting mappings to a non-redundant set, place the product on the same patch of surface as the reactant using the symmetry of the clean slab, interpolate between them without tearing molecules apart across the periodic boundary, rank the resulting candidate paths by something much cheaper than a path optimisation, and only then optimise the few survivors. **This task asks you to run one such cascade end to end, exactly, on a fully specified deterministic instance, and to report the barrier it returns.** Nothing is random and no structure is supplied by hand: the ensembles are enumerated, the calculator is analytic, and the whole computation is a single reproducible number.

*The calculator.* Energies and forces come from an analytic surrogate that plays the part a machine-learned interatomic potential plays in practice. With species indices $s_i$ drawn from the order $\mathrm{M}$ (slab metal), $\mathrm{C}$, $\mathrm{O}$, $\mathrm{H}$,

$$E=\sum_{\substack{i<j\\ \text{not both slab}}} S(r_{ij})\,D_{s_is_j}\Big(e^{-2\alpha_{s_is_j}(r_{ij}-r^{0}_{s_is_j})}-2e^{-\alpha_{s_is_j}(r_{ij}-r^{0}_{s_is_j})}\Big)\;+\;\sum_{i\in\text{ads}}\kappa_{s_i}\big[\max(0,\,c_i-v_{s_i})\big]^{2},$$

$$c_i=\sum_{\substack{j\in\text{ads},\;j\neq i\\ b_{s_is_j}>0}} f\!\left(r_{ij};\,1.3\,b_{s_is_j},\;2.0\,b_{s_is_j}\right),\qquad S(r)=f(r;\,4.0\,\text{\AA},\,5.0\,\text{\AA}),$$

$$f(r;r_1,r_2)=\tfrac12\Big[1+\cos\!\big(\pi\,\mathrm{clip}\big(\tfrac{r-r_1}{r_2-r_1},0,1\big)\big)\Big].$$

The coordination counter runs over adsorbate neighbours only, and a species pair with $b=0$ never contributes to it. All distances are minimum-image distances in the two periodic in-plane directions; the surface normal is not periodic. Slab atoms are frozen, so slab-slab pairs are omitted from the sum entirely and never carry a force. The parameter matrices, symmetric and in the species order above, are

$$D=\begin{pmatrix}0&0.25&0.30&0.12\\0.25&3.50&0.30&3.50\\0.30&0.30&2.50&4.00\\0.12&3.50&4.00&0.30\end{pmatrix}\ \text{eV},\quad \alpha_{ij}=1.0\ \text{\AA}^{-1}\ \text{for every pair},$$

$$r^{0}=\begin{pmatrix}2.80&2.90&2.80&2.60\\2.90&1.55&1.45&1.15\\2.80&1.45&1.50&1.05\\2.60&1.15&1.05&0.95\end{pmatrix}\ \text{\AA},\qquad b=\begin{pmatrix}0&0&0&0\\0&1.55&0&1.15\\0&0&1.50&1.05\\0&1.15&1.05&0\end{pmatrix}\ \text{\AA},$$

$$v=(0,\,4,\,2,\,1),\qquad \kappa=(0,\,2,\,2,\,2)\ \text{eV}.$$

*The system and the reaction.* The slab is a square-lattice (100) model of lattice constant $a=2.80$ \AA{} with $4\times4$ surface cells and three layers in ABAB stacking: layer $\ell=0,1,2$ holds the atoms at $\big((i+\tfrac{\ell\bmod 2}{2})a,\;(j+\tfrac{\ell\bmod 2}{2})a,\;-1.40\,\ell\ \text{\AA}\big)$ for $i,j=0,\dots,3$, so the top layer lies at $z=0$ and the periods are $L_x=L_y=11.2$ \AA. Every slab atom is species $\mathrm{M}$ and is held fixed throughout. The adsorbates are six atoms in the fixed order $\mathrm{C},\mathrm{H}_1,\mathrm{H}_2,\mathrm{H}_3,\mathrm{O},\mathrm{H}_4$, that is species $(\mathrm{C},\mathrm{H},\mathrm{H},\mathrm{H},\mathrm{O},\mathrm{H})$ with masses $(12.011,1.008,1.008,1.008,15.999,1.008)$, and the reaction is the transfer of $\mathrm{H}_3$ from the carbon to the oxygen,

$$\mathrm{CH_3}^{*}+\mathrm{OH}^{*}\;\rightleftharpoons\;\mathrm{CH_2}^{*}+\mathrm{H_2O}^{*} .$$

The donor is the carbon, the transferring atom is $\mathrm{H}_3$ and the acceptor is the oxygen. The atom mapping leaves the bonds $\mathrm{C}\!-\!\mathrm{H}_1$, $\mathrm{C}\!-\!\mathrm{H}_2$ and $\mathrm{O}\!-\!\mathrm{H}_4$ intact, so the moieties that must interpolate rigidly are $\{\mathrm{C},\mathrm{H}_1,\mathrm{H}_2\}$ and $\{\mathrm{O},\mathrm{H}_4\}$, while the active atom $\mathrm{H}_3$ forms a component of its own. The two adsorbates are $\{\mathrm{C},\mathrm{H}_1,\mathrm{H}_2,\mathrm{H}_3\}$ and $\{\mathrm{O},\mathrm{H}_4\}$ in the initial state and $\{\mathrm{C},\mathrm{H}_1,\mathrm{H}_2\}$ and $\{\mathrm{H}_3,\mathrm{O},\mathrm{H}_4\}$ in the final state. The point group of the clean surface is $C_{4v}$: the four rotations about $z$ by multiples of $90^{\circ}$, followed by those same four composed on the right with the mirror $\mathrm{diag}(1,-1,1)$, eight operations in that order.

*The ensembles.* Both ensembles are enumerated rather than sampled. The four in-cell site types are, in units of $a$, top $(0,0)$, bridge-$x$ $(\tfrac12,0)$, bridge-$y$ $(0,\tfrac12)$ and hollow $(\tfrac12,\tfrac12)$. A candidate places the carbon fragment on site $p$ of the cell $(0,0)$ at height $z=2.05$ \AA{} and the oxygen fragment on site $q$ of the cell $(1,0)$ at height $z=1.95$ \AA, with $p$ varying slowest and $q$ fastest, giving sixteen candidates per state. The rigid fragment geometries, relative to the site, are

$$\mathrm{CH_3}:\;\mathrm{C}\,(0,0,0),\;\mathrm{H}_1\,(1.03,0,0.36),\;\mathrm{H}_2\,(-0.515,0.892,0.36),\;\mathrm{H}_3\,(-0.515,-0.892,0.36),$$
$$\mathrm{OH}:\;\mathrm{O}\,(0,0,0),\;\mathrm{H}_4\,(0,0,0.97),$$
$$\mathrm{CH_2}:\;\mathrm{C}\,(0,0,0),\;\mathrm{H}_1\,(1.03,0,0.36),\;\mathrm{H}_2\,(-0.515,0.892,0.36),$$
$$\mathrm{H_2O}:\;\mathrm{H}_3\,(-0.757,0,0.586),\;\mathrm{O}\,(0,0,0),\;\mathrm{H}_4\,(0.757,0,0.586),$$

all in \AA. Each fragment is then rotated rigidly about the vertical axis through its own site by $25^{\circ}$, from the $x$ axis towards the $y$ axis, before being placed; this takes every fragment off the mirror planes of the square lattice, so that no enumerated candidate is invariant under a point-group operation of the surface and no relaxation has to break an exact degeneracy. Every one of the thirty-two candidates is then relaxed with the calculator above.

*The workflow to run.* The cascade to evaluate is a **published, fully automated transition-state workflow for surface reactions** that needs only a set of formal reactions, a slab model and a calculator, and that combines global end-state generation, atom mapping with index-permutation reduction, a two-objective geometry selection, a symmetry-based final-to-initial alignment, a periodic-boundary-aware interpolation and two cheap path-ranking scores, and finally a nudged elastic band with variable spring constants. **Find that source and take its constructions from it exactly.** From it you need, and must state in your reasoning: the two thresholds that bound the geometry selection a priori and the front construction with its tolerance window; the definition of the translation vector that aligns the final state to the initial state, together with the rule that picks the anchoring surface atom and the registration procedure, with its starting radius, its growth rule and its threshold, by which the source determines the operation composed with that translation, and what each side of the threshold implies; the ranking of the candidate periodic images of an atom that may not follow the minimum image convention, with the two terms of its score and the numerical weight that balances them; the ranking score by which interpolations are ranked, which is not the accumulated Euclidean displacement; the transfer-specific metric that supplements it; the two-stage filtering that follows, including the energy window applied to the relaxed modified final states and the change in interpolation density between the stages; and the scaling law for the spring constants, with its reference energy, the energy attributed to each spring and the numerical bounds of the range. Do not substitute a plausible reconstruction: the graded number depends on the exact cut-offs, on the exact score, on the exact metric and on the exact spring law. For the pair-potential interpolation on which the raw linear band is refined, state its objective, its distance weighting and how its target distances are built.

*Conventions fixed by this task.* Twelve choices the source leaves generic or omits are fixed here.

(i) **Relaxation.** Every geometry relaxation and every band optimisation uses the fast inertial relaxation engine with $\Delta t_0=0.1$, $\Delta t_{\max}=0.3$, $N_{\min}=5$, $f_{\mathrm{inc}}=1.1$, $f_{\mathrm{dec}}=0.5$, $a_{\mathrm{start}}=0.1$, $f_a=0.99$ and a cap of $0.1$ on the norm of the whole displacement of one system in one step, the velocity being rescaled by the same factor whenever that cap binds. One step of one system, with $P=\mathbf F\cdot\mathbf v$ over that system, is: if $P>0$, mix $\mathbf v\leftarrow(1-\alpha)\mathbf v+\alpha\lVert\mathbf v\rVert\,\mathbf F/\lVert\mathbf F\rVert$, then, if more than $N_{\min}$ consecutive earlier steps had $P>0$, set $\Delta t\leftarrow\min(f_{\mathrm{inc}}\Delta t,\Delta t_{\max})$ and $\alpha\leftarrow f_a\alpha$, and count this step as positive; otherwise set $\mathbf v\leftarrow0$, $\Delta t\leftarrow f_{\mathrm{dec}}\Delta t$, $\alpha\leftarrow a_{\mathrm{start}}$ and reset the count to zero. Then $\mathbf v\leftarrow\mathbf v+\Delta t\,\mathbf F$ and the displacement is $\Delta t\,\mathbf v$, subject to the cap. Velocities start at zero, convergence is tested on the largest force vector before each step, and each independent system, meaning each candidate geometry or each image of a band, carries its own time step, mixing parameter and velocity. Geometry relaxations run to $10^{-5}$ eV/\AA{} or $10^{4}$ steps, and the two band phases to their own thresholds or $3\times10^{4}$ steps each; on this instance every one of the four relaxations reaches its threshold, so none of the three budgets is binding and the answer does not depend on them. Systems relaxed together share one stopping test, applied to the largest force vector over all of them: the thirty-two enumerated placements are relaxed as one set, the shortlisted modified final states as one set, and the images of a band as one set.

(ii) **Structure filters.** A relaxed candidate is discarded unless every adsorbate atom has $z>0$ and $z<6$ \AA{} and every hydrogen's nearest heavy atom, in the minimum-image sense, is the one the state requires: $\mathrm{H}_1,\mathrm{H}_2,\mathrm{H}_3$ on the carbon and $\mathrm{H}_4$ on the oxygen in the initial state, $\mathrm{H}_1,\mathrm{H}_2$ on the carbon and $\mathrm{H}_3,\mathrm{H}_4$ on the oxygen in the final state.

(iii) **Selection objectives and redundancy.** The two objectives of the geometry selection are the relaxed energy and the minimum-image carbon-oxygen distance. The similarity clustering of the source is replaced by an exact rule: walking the survivors in enumeration order, a candidate is dropped when an already-kept candidate agrees with it to within $10^{-3}$ eV in energy and $10^{-2}$ \AA{} in every entry of the sorted vector of its fifteen minimum-image adsorbate pair distances. Of what remains, the three lowest in energy are kept, ties going to the smaller index.

(iv) **Periodic images.** The nine in-plane images are indexed $i=3(n_a+1)+(n_b+1)$ for offsets $n_a,n_b\in\{-1,0,1\}$, so $i=4$ is the home cell. Within one intact moiety the subgroup that keeps the minimum image is the one whose image index is shared by the largest number of its atoms, ties going to the smaller index; a moiety of one atom always keeps the minimum image; and the angular term of the ranking score is taken as zero whenever either vector in it has zero length. Coordinates are never wrapped back into the cell at any point of the workflow.

(v) **Interpolation.** A band of $M$ images is the straight line in the chosen displacements. Each interior image is refined by exactly $50$ steepest-descent steps along the exact gradient of the interpolation objective, each pair counted once, with a step length of $0.01$ and a cap of $0.05$ \AA{} applied to each atom separately: an atom whose step would exceed $0.05$ \AA{} has its own step rescaled to $0.05$ \AA, and the other atoms' steps are left unchanged. The two endpoints are held fixed. The pair set is every pair of adsorbate atoms together with every adsorbate-slab pair; adsorbate-adsorbate distances are taken from the interpolated coordinates as they are, while adsorbate-slab distances use the minimum image convention, because the slab is periodic and does not move. The reported objective value of each endpoint is zero. The coarse stage uses $6$ images and the dense stage $20$.

(vi) **The transfer metric.** Where the source leaves the distance set of its transfer metric generic, this task takes all three mutual distances among the donor, the transferring atom and the acceptor, averaged over all images of the band, with the coordinates as interpolated and no minimum image convention.

(vii) **The path coordinate.** The path coordinate of an image is the running sum, from zero at the first image, of the Frobenius norms of the differences between consecutive images, and wherever a quantity is accumulated along that coordinate, the trapezoidal rule on the images as they are is used.

(viii) **Ordering.** Candidate interpolations are ordered by the ranking metric with ties broken by initial-state index, then final-state index, then symmetry-operation index; the ten best are re-ordered by the transfer metric with the same tie-breaks; the survivors of the energy window are ordered by the ranking metric recomputed on the dense interpolation, with ties broken by the transfer metric and then by position in the previous ordering; and the single best survivor is the one that is optimised.

(ix) **The band optimisation.** The band carries $20$ images. Its force is the true force with its tangential component removed plus the difference of the two neighbouring spring extensions along the tangent, with the energy-weighted upwind tangent of Henkelman and Jonsson: the forward difference when the energy rises through the image, the backward difference when it falls through it, and at a local extremum the two differences weighted by the larger and the smaller of the two absolute energy differences, the larger weight going to the forward difference when the next image lies above the previous one. Endpoints are fixed. The first phase runs to a largest band-force vector of $0.2$ eV/\AA{} without a climbing image, the second to $10^{-4}$ eV/\AA; at the moment it stops, the interior image that is then highest in energy becomes the climbing image and **stays** the climbing image, carrying no spring force and the reverse rather than the removal of its tangential component. The spring constants are rebuilt from the current image energies at every force evaluation, so that the band force stays a continuous function of the images; holding them fixed over a block of evaluations would put a floor under the residual force that no step budget could get below.

(x) **Where the operations act.** Both the registration operation and the symmetry operations rotate the adsorbate coordinates measured from the initial-state anchoring atom and put that atom's position back afterwards; the slab is never transformed, and the initial state is never transformed. Where the source's rule for choosing the anchoring surface atom of a state admits more than one slab atom, those whose distance lies within $10^{-3}$ \AA{} of the smallest are tied, and the tie goes to the larger slab index, the slab atoms being indexed layer by layer from the top and, within a layer, with $i$ slowest and $j$ fastest.

(xi) **The order of the cascade.** Exactly these stages, in this order. Enumerate the thirty-two placements and relax them all. Apply the structure filters of (ii) to each state separately. Compute, for the survivors of one state, their relaxed energies and their minimum-image carbon-oxygen distances, and run the geometry selection on those two objectives, the energy window being measured from the lowest energy among that state's survivors. Remove redundancy from what the selection keeps and take the three lowest-energy representatives, as in (iii). Pair every retained initial state with every retained final state, align and augment each pairing, and for each augmented candidate assign the periodic images, build the coarse interpolation and evaluate both metrics. Order and shortlist as in (viii), relax the shortlisted modified final states, apply the energy window, build the dense interpolation of each survivor, evaluate both metrics again, order them, and optimise the single best. The reported quantity is read off that one optimised band.

(xii) **The reported quantity.** The barrier is the largest image energy of the optimised band minus the energy of its first image.

*What to report.* Run the cascade once on the instance above and **report the forward barrier $\Delta E^{\ddagger}$ of the selected pathway, in eV.**

In the reasoning, report the following, and little else, so that the cascade can be seen to have run. Identify the source of the workflow well enough that a reader can find it and state what you took from it. State the pair-potential interpolation objective, its distance weighting and how its target distances are built. The constructions listed in *The workflow to run* above. Then, from the run: how many of the sixteen enumerated candidates of each ensemble survive the structure filters; the lowest relaxed energy of each ensemble; the number of distinct structures each ensemble contains after redundancy removal and the three energies kept; the total number of candidate interpolations the pairing and symmetry augmentation produce; how many of them place at least one atom in a periodic image outside the home cell, and how many contain an atom that does not follow the minimum image convention, with the reason for that second count on this instance; the smallest value of the ranking metric over the coarse interpolations; the ranking metric and the transfer metric of the interpolation finally selected, which initial-state representative, final-state representative and symmetry operation it joins, and whether its end points are the lowest-energy structures of the two ensembles; whether any candidate is removed by the energy window applied to the relaxed modified final states, with the reason for that outcome on this instance; the energies of the first, the highest and the last image of the optimised band; the reaction energy; and the barrier itself. Also state, in one line each, why the ranking metric is preferred to the accumulated Euclidean displacement and what the spring law does to the distribution of images along the band.

## Output format

```
## Output format

A single final numeric answer wrapped in <final_answer>...</final_answer> tags, placed first, followed by the scientific reasoning wrapped in <reasoning>...</reasoning> tags. Put exactly one finite decimal number between the final-answer tags, with no units, words or extra lines. In the reasoning, give the sources, constructions and run results the problem asks you to report, with enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary; do not paste coordinate lists, per-image tables or the enumerated ensembles.
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

surface_calculator

Goal
----
Return the total potential energy of an adsorbate-on-slab configuration and the forces on the adsorbate atoms, for the surrogate calculator that stands in this task for a machine-learned interatomic potential. Two contributions are summed. The first is a pairwise Morse interaction, tapered smoothly to zero between an inner and an outer radius so that the periodic minimum-image sum is continuous everywhere inside the cell. The second is a bond-order over-coordination penalty that charges an energy whenever the smoothly counted number of bonds at an adsorbate atom exceeds the valence of its element; without it a purely pairwise model has no valence, the reactant and the product of a hydrogen transfer relax into the same structure, and there is no barrier between them. The slab is frozen, so pairs whose two atoms both belong to the slab are omitted and no force is returned for them, and the coordination sum runs over adsorbate neighbours only. The two in-plane directions are periodic, the surface normal is not, and a leading image axis is accepted so that a whole band of images is evaluated in one call.

```python
def surface_energy_forces(ads: "np.ndarray", slab: "np.ndarray",
                          ads_species: "np.ndarray",
                          slab_species: "np.ndarray", cell: "np.ndarray",
                          de: "np.ndarray", alpha: "np.ndarray",
                          r0: "np.ndarray", r_bond: "np.ndarray",
                          valence: "np.ndarray", k_coord: "np.ndarray",
                          f1: float = 1.3, f2: float = 2.0,
                          r_in: float = 4.0, r_out: float = 5.0) -> tuple:
    """Energy and adsorbate forces of the tapered Morse plus coordination model.

    Parameters
    ----------
    ads : numpy.ndarray
        Adsorbate positions in angstrom, shape (N, 3) or (M, N, 3) for a band
        of M images, with N >= 1.
    slab : numpy.ndarray
        Frozen slab positions in angstrom, shape (F, 3); F may be zero.
    ads_species : numpy.ndarray
        Integer species index of each adsorbate atom, shape (N,).
    slab_species : numpy.ndarray
        Integer species index of each slab atom, shape (F,).
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom.
    de, alpha, r0 : numpy.ndarray
        Symmetric (S, S) Morse well depths in eV, decay constants in 1/A and
        equilibrium lengths in A. alpha and r0 must be positive.
    r_bond : numpy.ndarray
        Symmetric non-negative (S, S) reference bond lengths of the
        coordination counter. A zero entry excludes that species pair from
        the coordination sum.
    valence : numpy.ndarray
        Non-negative valence of each species, shape (S,).
    k_coord : numpy.ndarray
        Non-negative over-coordination penalty strength of each species in eV,
        shape (S,).
    f1, f2 : float
        Multiples of r_bond at which the coordination switch leaves one and
        reaches zero; f2 must exceed f1 and both must be positive.
    r_in, r_out : float
        Inner and outer radius in A of the pair-energy taper; r_out must
        exceed r_in and both must be positive.

    Returns
    -------
    result : tuple
        (energy, forces). energy is a float for a single configuration and an
        array of shape (M,) for a band; forces has the shape of ads and
        holds the forces on the adsorbate atoms in eV/A.

    Raises
    ------
    ValueError
        If ads, slab, the species arrays or cell have the wrong shape, if a
        parameter matrix is not square, not symmetric or of a different shape
        from the others, if alpha or r0 is not positive, if r_bond, valence or
        k_coord is negative, if valence or k_coord has the wrong length, if
        f2 <= f1 or r_out <= r_in, or if a species index is out of range.
    """
    return energy, forces
```

### Step 2

pbc_displacements

Goal
----
Return, for every adsorbate atom, the displacement vector that carries it from the initial state to the final state through the periodic image that the workflow's periodic-boundary-aware interpolation assigns to it, together with the index of that image. Atoms that the reaction leaves bonded together are grouped into moieties, and the assignment must keep each moiety intact across the periodic boundary.

```python
def pbc_displacements(pos_is: "np.ndarray", pos_fs: "np.ndarray",
                      cell: "np.ndarray", moiety: "np.ndarray",
                      alpha: float = 0.68) -> tuple:
    """Periodic-image-aware displacements from an initial to a final state.

    Parameters
    ----------
    pos_is : numpy.ndarray
        Initial-state positions in angstrom, shape (N, 3) with N >= 1.
    pos_fs : numpy.ndarray
        Final-state positions in angstrom, same shape as pos_is.
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom.
    moiety : numpy.ndarray
        Integer label per atom, shape (N,), grouping the atoms that the
        reaction leaves connected. Atoms whose bonds break carry a label of
        their own.
    alpha : float
        Non-negative weight of the length term relative to the angle term in
        the ranking of candidate periodic images.

    Returns
    -------
    result : tuple
        (disp, index). disp has shape (N, 3) and holds the displacement of
        each atom through its chosen image; index has shape (N,) and holds
        that image index as a float in the range 0 to 8.

    Raises
    ------
    ValueError
        If pos_is is not an (N, 3) array with N >= 1, if pos_fs has a
        different shape, if moiety does not carry one label per atom, if cell
        is not two positive lengths, or if alpha is negative.
    """
    return disp, index
```

### Step 3

idpp_band

Goal
----
Return a band of images between an initial state and the final state reached by the supplied displacements, refined on the image-dependent pair potential, together with the value of that potential at every image.

```python
def build_idpp_band(pos_is: "np.ndarray", disp: "np.ndarray",
                    frozen: "np.ndarray", cell: "np.ndarray",
                    n_images: int, n_iter: int = 50, step: float = 0.01,
                    max_step: float = 0.05) -> tuple:
    """Linear band in the supplied displacements, refined on the IDPP surface.

    Parameters
    ----------
    pos_is : numpy.ndarray
        Initial-state positions of the moving atoms in angstrom, shape (N, 3)
        with N >= 2.
    disp : numpy.ndarray
        Displacement of each moving atom to the final state, same shape as
        pos_is.
    frozen : numpy.ndarray
        Positions of the frozen anchor atoms in angstrom, shape (F, 3); an
        empty array leaves the objective to the moving pairs alone.
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom, used for the
        moving-to-frozen distances only.
    n_images : int
        Number of images including both endpoints, at least 2.
    n_iter : int
        Non-negative number of steepest-descent steps applied to each
        interior image.
    step : float
        Positive step length of the steepest descent in A per unit gradient.
    max_step : float
        Positive cap in A on the displacement of any one atom in one step.

    Returns
    -------
    result : tuple
        (band, e_idpp). band has shape (n_images, N, 3); e_idpp has shape
        (n_images,) and is zero at both endpoints.

    Raises
    ------
    ValueError
        If pos_is is not an (N, 3) array with N >= 2, if disp has a different
        shape, if cell is not two positive lengths, if n_images is below 2,
        if n_iter is negative, or if step or max_step is not positive.
    """
    return band, e_idpp
```

### Step 4

reaction_metrics

Goal
----
Return the two reaction-distance metrics by which the workflow ranks candidate interpolations before any band optimisation is attempted: the ranking metric $\\mu$ built from the interpolation-potential profile of the band, and the transfer-specific metric $\\tau$ for a reaction in which one atom moves from a donor centre to an acceptor centre.

```python
def reaction_distance_metrics(band: "np.ndarray", e_idpp: "np.ndarray",
                              donor: int, transfer: int,
                              acceptor: int) -> tuple:
    """The two reaction-distance metrics that rank candidate interpolations.

    Parameters
    ----------
    band : numpy.ndarray
        Band coordinates in angstrom, shape (M, N, 3) with M >= 2.
    e_idpp : numpy.ndarray
        Interpolation potential of each image, shape (M,).
    donor, transfer, acceptor : int
        Indices into the atom axis of band naming the donor centre, the
        transferring atom and the acceptor centre.

    Returns
    -------
    result : tuple
        (mu, tau), both Python floats: the ranking metric mu of the
        interpolation in eV*A and its transfer metric tau in A, under the
        conventions stated for this step.

    Raises
    ------
    ValueError
        If band is not an (M, N, 3) array with M >= 2, if e_idpp does not
        carry one entry per image, or if any of the three indices lies
        outside the atom range.
    """
    return mu, tau
```

### Step 5

pareto_selection

Goal
----
Return the indices of the candidate geometries that the workflow's two-objective geometry-selection stage keeps, given one energy and one reactive separation per candidate.

```python
def select_pareto_geometries(energies: "np.ndarray",
                             distances: "np.ndarray",
                             e_window: float = 1.2, d_window: float = 4.5,
                             tol_e: float = 0.1, tol_d: float = 0.1,
                             n_fronts: int = 2) -> "np.ndarray":
    """Two-objective front selection of candidate adsorption geometries.

    Parameters
    ----------
    energies : numpy.ndarray
        Energy of each candidate in eV, shape (K,) with K >= 1.
    distances : numpy.ndarray
        Reactive separation of each candidate in angstrom, same length as
        energies.
    e_window : float
        Non-negative energy window in eV above the lowest energy in the set.
    d_window : float
        Non-negative absolute cut-off in angstrom on the reactive separation.
    tol_e, tol_d : float
        Non-negative tolerances in eV and angstrom defining the band kept
        around each undominated point.
    n_fronts : int
        Number of successive fronts to peel, at least 1.

    Returns
    -------
    keep : numpy.ndarray
        Integer indices of the retained candidates, ascending.

    Raises
    ------
    ValueError
        If energies is empty, if distances has a different length, if
        e_window, d_window, tol_e or tol_d is negative, or if n_fronts is
        below 1.
    """
    return keep
```

### Step 6

align_final_state

Goal
----
Return the translation with which the workflow aligns a final state to an initial state on a periodic slab, the registration operation that must be composed with that translation to restore an equivalent adsorption site, and the set of symmetry-equivalent aligned final states, one for each supplied point-group operation.

```python
def align_final_state(slab: "np.ndarray", ads_is: "np.ndarray",
                      ads_fs: "np.ndarray", masses: "np.ndarray",
                      frag_is: "np.ndarray", frag_fs: "np.ndarray",
                      cell: "np.ndarray", sym_ops: "np.ndarray",
                      r_cut: float = 4.0, r_tol: float = 1.0) -> tuple:
    """Anchor-based alignment of a final state to an initial state on a slab.

    Parameters
    ----------
    slab : numpy.ndarray
        Frozen slab positions in angstrom, shape (F, 3) with F >= 1.
    ads_is, ads_fs : numpy.ndarray
        Adsorbate positions of the initial and final states in angstrom, both
        of shape (N, 3) with N >= 1.
    masses : numpy.ndarray
        Positive mass of each adsorbate atom, shape (N,).
    frag_is, frag_fs : numpy.ndarray
        Integer fragment label of each adsorbate atom in the initial and in
        the final state, shape (N,); each must define exactly two fragments.
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom.
    sym_ops : numpy.ndarray
        Point-group operations of the clean slab, shape (S, 3, 3) with S >= 1.
    r_cut : float
        Positive initial radius in angstrom of the tiled anchor environment,
        grown in steps of 0.5 A until the two environments have equal size.
    r_tol : float
        Positive registration distance in angstrom below which the two
        environments count as already aligned.

    Returns
    -------
    result : tuple
        (t, r_ref, candidates). t has shape (3,), r_ref shape (3, 3) and
        candidates shape (S, N, 3), one aligned final state per supplied
        operation, in the order the operations were given.

    Raises
    ------
    ValueError
        If slab, ads_is, ads_fs, masses, frag_is, frag_fs, cell or sym_ops has
        the wrong shape, if a mass is not positive, if either fragment label
        array does not define exactly two fragments, if r_cut or r_tol is not
        positive, or if no cut-off radius makes the two environments equal in
        size.
    """
    return t, r_ref, candidates
```

### Step 7

variable_springs

Goal
----
Return the energy-scaled spring constant of every adjacent pair of images of a band, following the workflow's variable-spring scheme for its nudged elastic bands. There is one spring per adjacent pair, so a band of $M$ images yields $M-1$ constants.

```python
def variable_spring_constants(energies: "np.ndarray", k_min: float = 0.1,
                              k_max: float = 4.0) -> "np.ndarray":
    """Energy-scaled nudged-elastic-band spring constants.

    Parameters
    ----------
    energies : numpy.ndarray
        Image energies in eV, shape (M,) with M >= 2, ordered from the
        initial state to the final state.
    k_min : float
        Positive spring constant in eV/A^2 assigned at and below the
        reference energy.
    k_max : float
        Spring constant in eV/A^2 assigned at the band maximum; it must
        exceed k_min.

    Returns
    -------
    k : numpy.ndarray
        Spring constants of the M - 1 adjacent image pairs, in eV/A^2.

    Raises
    ------
    ValueError
        If energies holds fewer than two images, or if k_max does not exceed
        k_min or either is not positive.
    """
    return k
```

### Step 8

neb_band_forces

Goal
----
Return the nudged-elastic-band force on every image of a band, given the image coordinates, the image energies, the true forces of the calculator and one spring constant per adjacent pair, with an optional climbing image. The tangent is the energy-weighted upwind tangent of Henkelman and Jonsson (J. Chem. Phys. 113, 9978, 2000) and the climbing image is that of Henkelman, Uberuaga and Jonsson (J. Chem. Phys. 113, 9901, 2000).

```python
def neb_band_forces(band: "np.ndarray", energies: "np.ndarray",
                    forces: "np.ndarray", k: "np.ndarray",
                    climb: int = -1) -> "np.ndarray":
    """Nudged-elastic-band force of every image, with an optional climbing image.

    Parameters
    ----------
    band : numpy.ndarray
        Image coordinates in angstrom, shape (M, N, 3) with M >= 3.
    energies : numpy.ndarray
        True energy of each image in eV, shape (M,).
    forces : numpy.ndarray
        True force on every atom of every image in eV/A, shape of band.
    k : numpy.ndarray
        Positive spring constants of the M - 1 adjacent image pairs in
        eV/A^2.
    climb : int
        Index of the climbing image, an interior index between 1 and M - 2,
        or -1 for a band without a climbing image.

    Returns
    -------
    out : numpy.ndarray
        Band force of the shape of band, zero on both endpoints.

    Raises
    ------
    ValueError
        If band is not an (M, N, 3) array with M >= 3, if energies, forces or
        k has the wrong shape, if any spring constant is not positive, or if
        climb is neither -1 nor an interior image index.
    """
    return out
```

### Step 9

nebscape_barrier

Goal
----
Run the whole automated transition-state workflow on the benchmark surface reaction and return the forward barrier of the pathway it selects. The slab, the two adsorbate ensembles and the reaction are those of the benchmark: a hydrogen transfer from a methyl group to a hydroxyl group on a square-lattice (100) surface, with the atom order carbon, three hydrogens, oxygen, hydrogen and the third hydrogen transferring. The stages are, in order: enumerate both ensembles by placing the two fragments on the four in-cell site types with the fragment geometries, heights and in-plane rotation given; relax them all with the supplied calculator; discard structures that have changed their bonding pattern, sunk below the top layer or desorbed; run the two-objective geometry selection on energy and donor-acceptor separation; remove structural duplicates and keep the lowest-energy representatives; align every final state to every initial state and augment it by the point-group operations of the slab; assign the periodic images, build the coarse interpolation of every pairing and rank the pairings by the ranking metric; re-rank the best of those by the transfer metric; relax the modified final states and apply the energy window; recompute both metrics on the dense interpolation and take the best; and finally optimise that band with variable springs and a climbing image. The returned value is the largest image energy of the optimised band minus the energy of its first image.

```python
def run_nebscape(de: "np.ndarray", alpha: "np.ndarray", r0: "np.ndarray",
                 r_bond: "np.ndarray", valence: "np.ndarray",
                 k_coord: "np.ndarray", lattice: float = 2.8,
                 n_cell: int = 4, n_layer: int = 3, z_layer: float = 1.4,
                 z_c: float = 2.05, z_o: float = 1.95, tilt: float = 25.0,
                 z_max: float = 6.0,
                 n_rep: int = 3, n_coarse: int = 6, n_fine: int = 20,
                 n_neb: int = 20, n_pre: int = 10, e_window: float = 1.2,
                 d_window: float = 4.5, de_window: float = 0.1,
                 mic_alpha: float = 0.68, k_min: float = 0.1,
                 k_max: float = 4.0, f_switch: float = 0.2,
                 f_conv: float = 1e-4, f_relax: float = 1e-5,
                 steps_relax: int = 10000, steps_pre: int = 30000,
                 steps_ci: int = 30000) -> float:
    """Forward barrier of the pathway the automated workflow selects.

    Parameters
    ----------
    de, alpha, r0, r_bond : numpy.ndarray
        Symmetric (S, S) calculator matrices: Morse well depths in eV, decay
        constants in 1/A, equilibrium lengths in A and coordination bond
        lengths in A.
    valence, k_coord : numpy.ndarray
        Valence and over-coordination penalty strength in eV of each species,
        shape (S,).
    lattice : float
        Positive square-lattice constant of the slab in angstrom.
    n_cell : int
        Number of surface cells per direction, at least 2.
    n_layer : int
        Number of slab layers, at least 1.
    z_layer : float
        Positive interlayer spacing in angstrom.
    z_c, z_o : float
        Heights in angstrom at which the carbon and the oxygen fragments are
        placed before relaxation.
    tilt : float
        Angle in degrees through which each rigid fragment is rotated about
        the vertical axis of its own site before placement, measured from the
        x axis towards the y axis.
    z_max : float
        Height in angstrom above which an adsorbate atom counts as desorbed.
    n_rep : int
        Number of lowest-energy representatives kept per ensemble, at least 1.
    n_coarse, n_fine, n_neb : int
        Image counts of the coarse ranking interpolation, the dense ranking
        interpolation and the optimised band; each at least 3.
    n_pre : int
        Number of best candidates under the ranking metric that are re-ranked
        by the transfer metric, at least 1.
    e_window, d_window : float
        Energy window in eV and separation cut-off in angstrom of the
        geometry selection.
    de_window : float
        Energy window in eV within which a relaxed modified final state is
        accepted.
    mic_alpha : float
        Length weight of the periodic-image ranking.
    k_min, k_max : float
        Spring constant range in eV/A^2.
    f_switch, f_conv : float
        Band-force thresholds in eV/A at which the climbing image is switched
        on and at which the band counts as converged.
    f_relax : float
        Force threshold in eV/A of the geometry relaxations.
    steps_relax, steps_pre, steps_ci : int
        Step budgets of the geometry relaxations, of the non-climbing band
        pass and of the climbing band pass.

    Returns
    -------
    barrier : float
        Forward barrier of the selected pathway in eV, the largest image
        energy of the optimised band minus the energy of its first image.

    Raises
    ------
    ValueError
        If lattice or z_layer is not positive, if n_cell is below 2 or
        n_layer below 1, if any image count is below 3, if n_rep or n_pre is
        below 1, if no candidate of either ensemble survives the structure
        filter, or if no interpolation survives the selection cascade.
    """
    return barrier
```
