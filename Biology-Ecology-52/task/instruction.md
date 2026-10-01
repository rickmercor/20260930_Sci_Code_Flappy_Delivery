# Biology-Ecology-52

## Background

A food web is a directed, weighted network: the nodes are species, an arrow runs from a prey to its predator,
and the weight of the arrow is the diet coefficient, the proportion of that prey in the predator's diet. From
the weighted network one derives each species' trophic position, a network centrality defined recursively as
one plus the diet-weighted mean position of the prey, with primary producers at one; because the definition is
linear, the whole vector of positions follows from the diet matrix by solving a linear system. Measuring diet
coefficients in the field is laborious, whereas trophic positions can be estimated efficiently from the
enrichment of heavy stable isotopes, in particular nitrogen, up the food chain. The resulting positions place
every species on a trophic axis but say nothing directly about which species eats which. Recovering the network
from such node attributes is an inverse problem of the kind studied in financial, genetic and metabolic
networks: the constraints (here the positions) are far fewer than the unknowns (the coefficients), so a
reconstruction must supply additional structure, typically a probabilistic model of which links are plausible,
to select one network from the solution space. Classical theoretical models of food-web structure (cascade,
niche, nested-hierarchy models) generate networks whose trophic positions need not match measured ones, while
isotope mixing models estimate diet proportions but rely on strong assumptions about fractionation and
turnover. Reconstructions are judged against webs whose links are known, qualitatively by how many true links
are recovered and how many spurious ones are introduced, and quantitatively by how close the reconstructed
coefficients are to the true ones.

## Problem

Food webs are directed networks of who eats whom, and each species in them has a trophic position: one for
primary producers, and for a consumer one plus the diet-weighted mean of the trophic positions of its prey, so
that a consumer feeding only on producers sits at position two. Trophic positions are now routinely estimated
from stable-isotope ratios, but they do not by themselves reveal the feeding links. A recent source poses the
inverse problem of recovering the diet-coefficient matrix of a web from the vector of trophic positions alone
and solves it with a deterministic pair-decomposition scheme: for each consumer the prey that are plausible at a
given trophic specialization are decomposed into candidate prey pairs, each pair has a unique two-prey diet
consistent with the consumer's position, the pairs receive probabilities from Bayes' theorem, the unknown
trophic specialization is averaged out, and the consumer's diet row is the probability-weighted combination of the pair diets. The inputs are the trophic positions and a maximum trophic specialization; the output is a
reconstructed diet matrix, which the source assesses against known webs by the existence of links and by the
closeness of the coefficients.

In the diet-coefficient matrix Q the entry Q(i, j) is the proportion of prey j in the diet of consumer i;
consumer rows sum to one, producer rows are zero, and the diagonal is zero because cannibalism is excluded. The
trophic positions y then satisfy y = (I - Q)^-1 1. For a consumer i with position y_i the centre of its prey
is c = y_i - 1. At trophic specialization sigma the source admits as prey of i only the other species whose position lies inside a window around c whose half-width the source ties to sigma (strict inequalities: a species exactly on the window's boundary or exactly at c is not admissible), and its decomposition forms candidate prey pairs from the admissible prey so that the two-prey diet of each pair reproduces y_i. Each candidate pair carries the source's Bayesian weight at specialization sigma, a prior over the candidate pairs times the likelihood the source assigns to the two prey being eaten at that specialization. Because the specialization of a real consumer is unknown, the source averages the pair weights in its own way over sigma uniformly distributed on [0, sigma_max], each pair contributing only at the specializations at which it is a candidate and the prior being evaluated at each sigma; evaluate this average exactly (every integral involved has a closed form), not by numerical quadrature. The averaged weights yield the pair probabilities, which sum to one over the candidate pairs, and the consumer's diet row is the combination of its pair diets weighted by those probabilities; it sums to one and reproduces y_i.

Apply the method to the following synthetic web of twelve species, numbered 0 to 11. Species 0 is the only
primary producer. Species 1, 2 and 3 feed on species 0 alone. The other diets are: species 4 feeds on 0, 1, 2
with coefficients 0.25, 0.45, 0.30; species 5 on 1, 3, 4 with 0.30, 0.30, 0.40; species 6 on 2, 4, 5 with
0.50, 0.20, 0.30; species 7 on 4, 5, 6 with 0.40, 0.35, 0.25; species 8 on 3, 5, 6, 7 with 0.15, 0.25, 0.30,
0.30; species 9 on 6, 7, 8 with 0.35, 0.40, 0.25; species 10 on 7, 8, 9 with 0.30, 0.30, 0.40; species 11 on
8, 9, 10 with 0.20, 0.35, 0.45. Compute the trophic positions of this true web, discard the diets, and reconstruct the web from the positions alone with sigma_max at the upper bound on the trophic specialization that the source adopts. Use sigma_max = 1/3. Consumers whose position is exactly two
feed on the primary producer alone, producers have empty rows, and in this web no other species sits exactly
at a consumer's centre. A link of a web is any strictly positive diet coefficient.

Empirical trophic positions carry measurement error, and the source propagates such errors through its
deterministic reconstruction by simulation; quantify the local sensitivity of the reconstruction exactly instead.
Treat the positions of the eight consumers that the pair decomposition reconstructs (species 4 to 11) as the
estimated quantities, the producer and the three primary consumers being fixed at 1 and 2 by the isotope
baseline, and consider the similarity between the true and the reconstructed matrices (the sum of the
entry-wise products divided by the product of the entry-wise L2 norms) as a function of these eight positions,
the web being reconstructed from the displaced positions with the same sigma_max and compared with the
unchanged true web. At the true positions the candidate pairs of every consumer persist under small
displacements (no species sits on a window boundary or at a centre, and no pair has its two prey at equal
distances from the centre), so the similarity is differentiable there, every quantity that depends on the
positions moving with them.

For small independent errors of standard deviation s on the eight positions the expected similarity deviates
from its error-free value, to leading order, by s^2/2 times the Laplacian of the similarity with respect to the
eight positions (the sum of its second partial derivatives), the exact counterpart of the bias the source's
simulation would show.

Report this Laplacian, evaluated exactly (analytically, not by finite differences), to at least five
significant figures.

State the conventions you adopted, including the admissibility window, the form of the specialization average and the value of sigma_max, and justify each from the source. In the reasoning give the trophic positions of species 8 and 11, the candidate prey pairs of species 11 with their probabilities and its reconstructed coefficients on species 9 and on species 10, the probability of the dominant candidate pair of species 8 and its reconstructed coefficients on species 5 and on species 6, the numbers of true-positive,
false-negative and false-positive links with the true positive rate, the false positive rate and the
balanced accuracy, the mean error between the two matrices (the sum over all 144 entries of the absolute
difference divided by 144), the similarity, the true link that the reconstruction cannot recover and why, and
the Euclidean norm of the gradient of the similarity with respect to the eight positions together with its
partial derivatives with respect to the positions of species 8 and of species 10, and the second partial
derivative of the similarity with respect to the position of species 8.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

trophic_positions

Goal
----
Return the trophic positions of every species of a food web from its diet-coefficient matrix Q, whose entry Q[i, j] is the proportion of prey j in the diet of predator i (rows of consumers sum to one, rows of primary producers are zero, the diagonal is zero because cannibalism is excluded). The trophic position of a predator is one plus the diet-weighted mean of the trophic positions of its prey, and primary producers have trophic position one; solve the resulting linear system y = (I - Q)^-1 1 exactly.

```python
def trophic_positions(diet: "numpy.ndarray") -> "numpy.ndarray":
    """Return the trophic positions of every species of a food web from its diet-coefficient matrix Q, whose entry Q[i, j] is the proportion of prey j in the diet of predator i (rows of consumers sum to one, rows of primary producers are zero, the diagonal is zero because cannibalism is excluded). The trophic position of a predator is one plus the diet-weighted mean of the trophic positions of its prey, and primary producers have trophic position one; solve the resulting linear system y = (I - Q)^-1 1 exactly.

    Parameters
    ----------
    diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative diet coefficients; consumer rows sum to one, producer rows to zero, the diagonal is zero.

    Returns
    -------
    positions : numpy.ndarray
        One-dimensional array of length n holding the trophic position of each species (float64).

    Raises
    ------
    ValueError
        If diet is not square, contains a non-finite or negative entry or a nonzero diagonal entry, has a row sum other than 0 or 1, or I - Q is singular.
    """
    return positions
```

### Step 2

pair_diet_coefficients

Goal
----
Return the two diet coefficients of a predator that feeds on exactly two prey, with trophic positions y_below and y_above (y_below < y_above), given that the predator's trophic position minus one equals centre and lies between the two prey positions: the coefficients are the unique proportions summing to one whose weighted mean of the prey positions equals centre. Return the array [coefficient of the lower prey, coefficient of the upper prey].

```python
def pair_diet_coefficients(y_below: float, y_above: float, centre: float) -> "numpy.ndarray":
    """Return the two diet coefficients of a predator that feeds on exactly two prey, with trophic positions y_below and y_above (y_below < y_above), given that the predator's trophic position minus one equals centre and lies between the two prey positions: the coefficients are the unique proportions summing to one whose weighted mean of the prey positions equals centre. Return the array [coefficient of the lower prey, coefficient of the upper prey].

    Parameters
    ----------
    y_below : float
        Trophic position of the lower prey.
    y_above : float
        Trophic position of the upper prey, strictly larger than y_below.
    centre : float
        The predator's trophic position minus one; must satisfy y_below <= centre <= y_above.

    Returns
    -------
    coefficients : numpy.ndarray
        Array of shape (2,): the diet coefficients of the lower and of the upper prey (float64), summing to one.

    Raises
    ------
    ValueError
        If any input is not finite, y_above does not exceed y_below, or centre lies outside [y_below, y_above].
    """
    return coefficients
```

### Step 3

candidate_pairs

Goal
----
Return the candidate prey pairs of one predator at trophic specialization sigma according to the source's decomposition. The admissible prey of predator i are the other species whose trophic position lies inside the source's admissibility window around the centre c = positions[i] - 1, the window of half-width 3 sigma, from c - 3 sigma to c + 3 sigma (strict inequalities: a species exactly on the window's boundary or exactly at the centre is not admissible; the predator itself is never admissible); every admissible prey below the centre is paired with every admissible prey above it, so that the two-prey solution of step 02 applies to every pair. Return an integer array of shape (n_pairs, 2) whose rows are (index of the lower prey, index of the upper prey), ordered by lower-prey index and then by upper-prey index in increasing order of species index; the array has zero rows when no pair exists.

```python
def candidate_pairs(positions: "numpy.ndarray", predator: int, sigma: float) -> "numpy.ndarray":
    """Return the candidate prey pairs of one predator at trophic specialization sigma according to the source's decomposition. The admissible prey of predator i are the other species whose trophic position lies inside the source's admissibility window around the centre c = positions[i] - 1, the window of half-width 3 sigma, from c - 3 sigma to c + 3 sigma (strict inequalities: a species exactly on the window's boundary or exactly at the centre is not admissible; the predator itself is never admissible); every admissible prey below the centre is paired with every admissible prey above it, so that the two-prey solution of step 02 applies to every pair. Return an integer array of shape (n_pairs, 2) whose rows are (index of the lower prey, index of the upper prey), ordered by lower-prey index and then by upper-prey index in increasing order of species index; the array has zero rows when no pair exists.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    sigma : float
        Nonnegative trophic specialization.

    Returns
    -------
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2): rows (lower prey index, upper prey index) in the stated order.

    Raises
    ------
    ValueError
        If positions is not a finite 1-D array of at least three entries, predator is not a valid index, or sigma is negative or not finite.
    """
    return pairs
```

### Step 4

pair_thresholds

Goal
----
Return, for every given candidate prey pair of one predator (rows (lower prey index, upper prey index) as produced by step 03), the admissibility threshold of the pair: the trophic specialization above which both of its prey lie inside the source's admissibility window of step 03, so that the pair is a candidate exactly for sigma strictly above the threshold. Return one threshold per pair, in the order of the input.

```python
def pair_thresholds(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray") -> "numpy.ndarray":
    """Return, for every given candidate prey pair of one predator (rows (lower prey index, upper prey index) as produced by step 03), the admissibility threshold of the pair: the trophic specialization above which both of its prey lie inside the source's admissibility window of step 03, so that the pair is a candidate exactly for sigma strictly above the threshold. Return one threshold per pair, in the order of the input.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2) of (lower prey index, upper prey index) pairs of the predator, every lower prey below and every upper prey above the predator's centre.

    Returns
    -------
    thresholds : numpy.ndarray
        One-dimensional array of length n_pairs holding the admissibility threshold of each pair (float64).

    Raises
    ------
    ValueError
        If positions is not a finite 1-D array of at least three entries, predator is not a valid index, pairs is not an integer array of shape (n_pairs, 2) with valid indices, or a pair does not bracket the predator's centre.
    """
    return thresholds
```

### Step 5

marginal_pair_weights

Goal
----
Return the source's specialization-averaged weight of every given candidate prey pair of one predator (the pairs of step 03 at sigma = sigma_max, with their thresholds of step 04, in the same order) when the trophic specialization is unknown: the source's Bayesian weight of a pair at specialization sigma - its prior 1 / n(sigma), uniform over the n(sigma) pairs that are candidates at sigma, times its likelihood, the product of the Gaussian densities of mean zero and standard deviation sigma at the two prey offsets c - y_lower and y_upper - c, that is exp(-((c - y_lower)^2 + (y_upper - c)^2) / (2 sigma^2)) / (2 pi sigma^2) - averaged over sigma uniformly distributed on [0, sigma_max], that is integrated over sigma and divided by sigma_max, a pair contributing only above its threshold and the prior being evaluated at each sigma. Evaluate the average exactly, not by numerical quadrature. The weights are not normalised; the pair probabilities are their normalisation over the candidate pairs.

```python
def marginal_pair_weights(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                          thresholds: "numpy.ndarray", sigma_max: float) -> "numpy.ndarray":
    """Return the source's specialization-averaged weight of every given candidate prey pair of one predator (the pairs of step 03 at sigma = sigma_max, with their thresholds of step 04, in the same order) when the trophic specialization is unknown: the source's Bayesian weight of a pair at specialization sigma - its prior 1 / n(sigma), uniform over the n(sigma) pairs that are candidates at sigma, times its likelihood, the product of the Gaussian densities of mean zero and standard deviation sigma at the two prey offsets c - y_lower and y_upper - c, that is exp(-((c - y_lower)^2 + (y_upper - c)^2) / (2 sigma^2)) / (2 pi sigma^2) - averaged over sigma uniformly distributed on [0, sigma_max], that is integrated over sigma and divided by sigma_max, a pair contributing only above its threshold and the prior being evaluated at each sigma. Evaluate the average exactly, not by numerical quadrature. The weights are not normalised; the pair probabilities are their normalisation over the candidate pairs.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2): the candidate pairs of the predator at sigma_max (step 03).
    thresholds : numpy.ndarray
        One-dimensional array of length n_pairs: the admissibility thresholds of the pairs (step 04), all below sigma_max.
    sigma_max : float
        Positive upper bound of the trophic specialization.

    Returns
    -------
    weights : numpy.ndarray
        One-dimensional array of length n_pairs holding the averaged unnormalised weights (float64).

    Raises
    ------
    ValueError
        If sigma_max is not positive and finite, positions or predator are invalid, pairs is empty or malformed, or thresholds does not hold one finite value per pair in [0, sigma_max).
    """
    return weights
```

### Step 6

superpose_row

Goal
----
Return the reconstructed diet row of one predator as the source's superposition of its candidate pairs: the pair weights (step 05) normalised to probabilities summing to one, and the row of length n_species whose entry for each prey is the probability-weighted sum, over the pairs containing that prey, of the prey's coefficient in the pair's two-prey diet (step 02). The row is zero outside the candidate prey.

```python
def superpose_row(n_species: int, pairs: "numpy.ndarray", pair_diets: "numpy.ndarray",
                  weights: "numpy.ndarray") -> "numpy.ndarray":
    """Return the reconstructed diet row of one predator as the source's superposition of its candidate pairs: the pair weights (step 05) normalised to probabilities summing to one, and the row of length n_species whose entry for each prey is the probability-weighted sum, over the pairs containing that prey, of the prey's coefficient in the pair's two-prey diet (step 02). The row is zero outside the candidate prey.

    Parameters
    ----------
    n_species : int
        Number of species in the web, at least 3 (the length of the row).
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2) of (lower prey index, upper prey index) pairs, non-empty.
    pair_diets : numpy.ndarray
        Array of shape (n_pairs, 2): for each pair the two-prey diet [coefficient of the lower prey, coefficient of the upper prey].
    weights : numpy.ndarray
        One-dimensional array of length n_pairs of nonnegative pair weights with a positive sum.

    Returns
    -------
    row : numpy.ndarray
        One-dimensional array of length n_species: the reconstructed diet coefficients of the predator (float64).

    Raises
    ------
    ValueError
        If n_species is not an integer of at least 3, pairs is empty or holds an index outside the web, pair_diets is not a finite nonnegative array of shape (n_pairs, 2), or weights is not one finite nonnegative value per pair with a positive sum.
    """
    return row
```

### Step 7

reconstruction_metrics

Goal
----
Return the reconstruction metrics comparing a reconstructed diet matrix with the true one, as the array [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity]. A link is a strictly positive coefficient and the diagonal is never a link: TP counts the links present in both webs, FN the true links missing from the reconstruction, TN the off-diagonal non-links of both, FP the reconstructed links absent from the true web; TPR = TP / L_true, FPR = (L - TP) / (n (n - 1) - L_true) with L_true and L the numbers of true and reconstructed links, BA = (TPR + 1 - FPR) / 2; the mean error is the sum of the absolute differences of all n^2 entries divided by n^2, and the similarity is the sum of the entry-wise products divided by the product of the two entry-wise (Frobenius) L2 norms.

```python
def reconstruction_metrics(true_diet: "numpy.ndarray", diet: "numpy.ndarray") -> "numpy.ndarray":
    """Return the reconstruction metrics comparing a reconstructed diet matrix with the true one, as the array [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity]. A link is a strictly positive coefficient and the diagonal is never a link: TP counts the links present in both webs, FN the true links missing from the reconstruction, TN the off-diagonal non-links of both, FP the reconstructed links absent from the true web; TPR = TP / L_true, FPR = (L - TP) / (n (n - 1) - L_true) with L_true and L the numbers of true and reconstructed links, BA = (TPR + 1 - FPR) / 2; the mean error is the sum of the absolute differences of all n^2 entries divided by n^2, and the similarity is the sum of the entry-wise products divided by the product of the two entry-wise (Frobenius) L2 norms.

    Parameters
    ----------
    true_diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative true diet coefficients.
    diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative reconstructed diet coefficients.

    Returns
    -------
    metrics : numpy.ndarray
        Array of shape (9,): [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity] (float64).

    Raises
    ------
    ValueError
        If the matrices are not square of the same size with at least two species, contain a non-finite or negative entry, the true web has no link or no absent link, or the reconstructed matrix is entirely zero.
    """
    return metrics
```

### Step 8

marginal_weight_derivatives

Goal
----
Return the exact first and second derivatives of the specialization-averaged weight of every given candidate prey pair of one predator (the weights of step 05 for the pairs of step 03 at sigma = sigma_max, in the same order) with respect to the trophic position of one species, all other positions, the pair list and sigma_max held fixed. The weights are those of step 05 evaluated at the displaced positions, with the centre c = positions[predator] - 1 moving with the predator's own position, the pair list kept as given and every quantity that step 05 derives from the positions (its offsets, the thresholds of step 04 and the candidate count of the prior at each sigma) taken as the functions of the positions that steps 04 and 05 define. Return the exact derivatives, analytically and to full double precision (finite differences of any order do not meet the accuracy required of this step), as an array of shape (n_pairs, 2) whose columns are the first and the second derivative, in the order of the input pairs, zero for every pair whose weight does not depend on the species' position. Raise ValueError if the derivative does not exist: a pair whose two prey offsets are equal to within a relative tolerance of 1e-12 (its threshold switches between the two prey), or pairs whose thresholds are equal to within an absolute tolerance of 1e-12 and would move apart under the perturbation.

```python
def marginal_weight_derivatives(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                                sigma_max: float, species: int) -> "numpy.ndarray":
    """Return the exact first and second derivatives of the specialization-averaged weight of every given candidate prey pair of one predator (the weights of step 05 for the pairs of step 03 at sigma = sigma_max, in the same order) with respect to the trophic position of one species, all other positions, the pair list and sigma_max held fixed. The weights are those of step 05 evaluated at the displaced positions, with the centre c = positions[predator] - 1 moving with the predator's own position, the pair list kept as given and every quantity that step 05 derives from the positions (its offsets, the thresholds of step 04 and the candidate count of the prior at each sigma) taken as the functions of the positions that steps 04 and 05 define. Return the exact derivatives, analytically and to full double precision (finite differences of any order do not meet the accuracy required of this step), as an array of shape (n_pairs, 2) whose columns are the first and the second derivative, in the order of the input pairs, zero for every pair whose weight does not depend on the species' position. Raise ValueError if the derivative does not exist: a pair whose two prey offsets are equal to within a relative tolerance of 1e-12 (its threshold switches between the two prey), or pairs whose thresholds are equal to within an absolute tolerance of 1e-12 and would move apart under the perturbation.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2): the candidate pairs of the predator at sigma_max (step 03), non-empty.
    sigma_max : float
        Positive upper bound of the trophic specialization; every pair's threshold lies below it.
    species : int
        Index of the species whose trophic position is displaced (the predator itself or any other species).

    Returns
    -------
    derivatives : numpy.ndarray
        Array of shape (n_pairs, 2): column 0 the first and column 1 the second derivative of each pair's averaged weight with respect to the position of the species (float64).

    Raises
    ------
    ValueError
        If sigma_max is not positive and finite, positions, predator or species are invalid, pairs is empty, malformed or does not bracket the centre, a pair's threshold is not below sigma_max, a pair has prey offsets equal to within a relative tolerance of 1e-12, or pairs with thresholds equal to within an absolute tolerance of 1e-12 would move apart under the perturbation.
    """
    return derivatives
```

### Step 9

diet_row_derivatives

Goal
----
Return the exact first and second derivatives of the reconstructed diet row of one predator with respect to the trophic position of one species, all other positions and sigma_max held fixed. The row is the one steps 02-06 build from the positions: the candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02) and the superposition (step 06); the pair list is held fixed (it persists under small displacements) while the pair weights (whose derivatives are step 08), their normalisation to probabilities and the two-prey diets all move with the position. Return the exact derivatives, analytically, as an array of shape (n_species, 2) whose columns are the first and the second derivative of the row (zero outside the candidate prey). Raise ValueError if the predator has no candidate pair at sigma_max, if another species sits at the predator's centre or on the admissibility window at sigma_max to within an absolute tolerance of 1e-12 (the row has no derivative there), or if step 08 raises.

```python
def diet_row_derivatives(positions: "numpy.ndarray", predator: int, sigma_max: float,
                         species: int) -> "numpy.ndarray":
    """Return the exact first and second derivatives of the reconstructed diet row of one predator with respect to the trophic position of one species, all other positions and sigma_max held fixed. The row is the one steps 02-06 build from the positions: the candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02) and the superposition (step 06); the pair list is held fixed (it persists under small displacements) while the pair weights (whose derivatives are step 08), their normalisation to probabilities and the two-prey diets all move with the position. Return the exact derivatives, analytically, as an array of shape (n_species, 2) whose columns are the first and the second derivative of the row (zero outside the candidate prey). Raise ValueError if the predator has no candidate pair at sigma_max, if another species sits at the predator's centre or on the admissibility window at sigma_max to within an absolute tolerance of 1e-12 (the row has no derivative there), or if step 08 raises.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    sigma_max : float
        Positive upper bound of the trophic specialization.
    species : int
        Index of the species whose trophic position is displaced.

    Returns
    -------
    derivatives : numpy.ndarray
        Array of shape (n_species, 2): column 0 the first and column 1 the second derivative of the predator's reconstructed diet row with respect to the position of the species (float64).

    Raises
    ------
    ValueError
        If sigma_max is not positive and finite, positions, predator or species are invalid, the predator has no candidate pair at sigma_max, a species sits at the predator's centre or on its admissibility window at sigma_max to within an absolute tolerance of 1e-12, or the weight derivatives of step 08 do not exist.
    """
    return derivatives
```

### Step 10

similarity_laplacian

Goal
----
Orchestrator. For a true diet-coefficient matrix, compute the trophic positions of all species (step 01) and reconstruct the whole web from those positions alone: species whose position is 1 to within an absolute tolerance of 1e-12 are primary producers and get zero rows; species whose position is 2 to within the same tolerance are primary consumers and feed on the primary producers alone, in equal proportions; for every other consumer form its candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02), and its row by superposition (step 06). Then take the similarity of step 07 between the true and the reconstructed matrices (the sum of the entry-wise products divided by the product of the entry-wise L2 norms) as a function of the positions of the consumers reconstructed by pairs (the producers' and primary consumers' positions and rows are fixed), the reconstructed matrix moving with them through the row derivatives of step 09 while the true matrix is unchanged, and return its Laplacian with respect to those positions: the sum over them of the exact second partial derivatives of the similarity. Raise ValueError if the web has no producer, some other species sits at the centre of a consumer reconstructed by pairs to within an absolute tolerance of 1e-12 (the producers at a primary consumer's centre are its diet by the rule above, not an error), a consumer reconstructed by pairs has no candidate pair at sigma_max, or the derivatives do not exist (step 08 or step 09 raises). Call the earlier step functions rather than reimplementing them.

```python
def similarity_laplacian(true_diet: "numpy.ndarray", sigma_max: float) -> float:
    """Orchestrator. For a true diet-coefficient matrix, compute the trophic positions of all species (step 01) and reconstruct the whole web from those positions alone: species whose position is 1 to within an absolute tolerance of 1e-12 are primary producers and get zero rows; species whose position is 2 to within the same tolerance are primary consumers and feed on the primary producers alone, in equal proportions; for every other consumer form its candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02), and its row by superposition (step 06). Then take the similarity of step 07 between the true and the reconstructed matrices (the sum of the entry-wise products divided by the product of the entry-wise L2 norms) as a function of the positions of the consumers reconstructed by pairs (the producers' and primary consumers' positions and rows are fixed), the reconstructed matrix moving with them through the row derivatives of step 09 while the true matrix is unchanged, and return its Laplacian with respect to those positions: the sum over them of the exact second partial derivatives of the similarity. Raise ValueError if the web has no producer, some other species sits at the centre of a consumer reconstructed by pairs to within an absolute tolerance of 1e-12 (the producers at a primary consumer's centre are its diet by the rule above, not an error), a consumer reconstructed by pairs has no candidate pair at sigma_max, or the derivatives do not exist (step 08 or step 09 raises). Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    true_diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative true diet coefficients (consumer rows sum to one).
    sigma_max : float
        Positive upper bound of the trophic specialization.

    Returns
    -------
    laplacian : float
        The sum of the second partial derivatives of the similarity with respect to the positions of the consumers reconstructed by pairs, as a native Python float.

    Raises
    ------
    ValueError
        If true_diet is invalid for step 01, sigma_max is not positive and finite, the web has no producer, a species sits at the centre of a consumer reconstructed by pairs to within an absolute tolerance of 1e-12, such a consumer has no candidate pair at sigma_max, or the derivatives do not exist (equal prey offsets in a pair, a species on an admissibility window, each to within 1e-12).
    """
    return laplacian
```
