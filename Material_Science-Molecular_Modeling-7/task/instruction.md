# Material_Science-Molecular_Modeling-7

## Background

Mg–Nd precipitation can couple chemical ordering on an hcp Mg-rich parent lattice with a structural transformation toward a bcc-related ordering.
The declared paper supplies the source-specific methodology and class convention. The numerical parent-site and energy-profile data below are synthetic task data and are not copied from the paper.
Three essential paper-derived objects are intentionally not supplied below: the Figure 8 class-numbering convention, the six Figure 13 DFT class counts, and the Figure 14 DFT-versus-MLIP matrix. They must be recovered from the main paper. The Figure 13 and Figure 14 information is needed to reconstruct the exact paper confusion matrix used in the final chance-corrected comparison.
## Parent-site data
Use six parent sites:
$$
i=0,1,2,3,4,5.
$$
A chemical decoration is written as
$$
\sigma=(\sigma_0,\sigma_1,\ldots,\sigma_5),
$$
where $\sigma_i=1$ means that site $i$ contains Nd, and $\sigma_i=0$ means that it contains Mg.
Consider only decorations containing exactly two or exactly three Nd atoms.
The full parent symmetry group is not given directly. It must be generated from the following two permutations:
$$
g_A=(1,2,3,4,5,0),
$$
$$
g_B=(0,5,4,3,2,1).
$$
For a permutation $g$, the value $g_i$ gives the new position of site $i$.
The two rows above are only generators. Generate all different permutations obtained by repeatedly composing them.
## Decoration representative

For a decoration $\sigma$, define the binary code
$$
b(\sigma)=\sum_{i=0}^{5}\sigma_i2^{5-i}.
$$
Several decorations may be related by the parent symmetry group. From every such orbit, use the decoration having the largest value of $b(\sigma)$ as the representative.
## Candidate Burgers pathways
Three candidate pathway supports are given below:
$$
\mathcal{T}_0=\{2,5\},\qquad
\mathcal{T}_1=\{0,3\},\qquad
\mathcal{T}_2=\{1,4\}.
$$
The pathway labels are intentionally not given in the usual cyclic order.
Each pathway support is an unordered pair. When a site permutation acts on a pathway, apply the permutation to both sites and then treat the resulting pair as unordered.
For every set of symmetry-equivalent pathway labels, keep the smallest supplied pathway label as the representative.
Pathway equivalence must be decided from symmetry before using the energy values.

## Transformation-energy profiles
The profiles are sampled at
$$
\lambda=\left(0,\frac{1}{6},\frac{2}{6},\frac{3}{6},\frac{4}{6},\frac{5}{6},1\right).
$$
The hcp endpoint is at $\lambda=0$ and the bcc endpoint is at $\lambda=1$.
All energies are in $\mathrm{meV/atom}$.

The profile rows are intentionally given in a mixed order.

| Decoration | Pathway label | DFT profile | MLIP profile |
|---|---:|---|---|
| 110100 | 1 | (18, 16, 13, 9, 5, 2, 0) | (17, 15, 12, 8, 5, 2, 0) |
| 110000 | 0 | (0, 2, 5, 9, 13, 16, 18) | (0, 2, 6, 10, 13, 16, 19) |
| 101010 | 2 | (0, 2, 5, 9, 13, 16, 18) | (0, 2, 6, 10, 13, 16, 19) |
| 101000 | 2 | (2, 12, 21, 25, 22, 15, 8) | (1, 3, 6, 10, 14, 17, 20) |
| 100100 | 2 | (8, 0, -5, -4, 2, 10, 18) | (7, 1, -4, -3, 2, 11, 17) |
| 110000 | 2 | (18, 16, 13, 9, 5, 2, 0) | (17, 12, 6, -1, -3, 2, 5) |
| 111000 | 1 | (8, 15, 22, 25, 21, 12, 2) | (9, 14, 21, 24, 20, 11, 3) |
| 101000 | 0 | (8, 15, 22, 25, 21, 12, 2) | (3, 11, 20, 24, 21, 14, 9) |
| 101010 | 0 | (0, 2, 5, 9, 13, 16, 18) | (0, 2, 6, 10, 13, 16, 19) |
| 100100 | 1 | (18, 10, 2, -4, -5, 0, 8) | (17, 9, 1, -3, -4, 1, 7) |
| 110100 | 0 | (8, 0, -5, -4, 2, 10, 18) | (7, 1, -4, -3, 2, 11, 17) |
| 110000 | 1 | (18, 16, 13, 9, 5, 2, 0) | (17, 12, 6, -1, -3, 2, 5) |
| 101010 | 1 | (0, 2, 5, 9, 13, 16, 18) | (0, 2, 6, 10, 13, 16, 19) |
| 111000 | 2 | (2, 12, 21, 25, 22, 15, 8) | (10, 18, 24, 26, 20, 12, 4) |
| 100100 | 0 | (8, 0, -5, -4, 2, 10, 18) | (7, 1, -4, -3, 2, 11, 17) |
| 110100 | 2 | (12, 7, 1, -4, -5, -1, 8) | (7, 1, -4, -3, 2, 8, 11) |
| 101000 | 1 | (8, 15, 22, 25, 21, 12, 2) | (3, 11, 20, 24, 21, 14, 9) |
| 111000 | 0 | (8, 15, 22, 25, 21, 12, 2) | (9, 14, 21, 24, 20, 11, 3) |

## Numerical rules

- Complete the symmetry reduction before classifying the energy profiles.
- Do not decide pathway equivalence from similar energy-profile shapes.
- Match every profile by both its decoration code and its pathway label.
- Pair the DFT and MLIP profiles by the same decoration-pathway key.
- Classify the retained DFT and MLIP profiles separately.
- Use DFT classes as rows and MLIP classes as columns.
- Count every retained symmetry-distinct pathway only once.
- Do not round intermediate values.
- The final kappa-retention ratio has no unit.
## Chance-corrected comparison

For any square confusion matrix $A$ with grand total $n$, row totals $r_i$ and column totals $c_i$, use
$$
p_o(A)=\frac{\operatorname{tr}(A)}{n},\qquad
p_e(A)=\frac{\sum_i r_i c_i}{n^2},\qquad
\kappa(A)=\frac{p_o(A)-p_e(A)}{1-p_e(A)}.
$$
Let $C$ be the synthetic DFT-row/MLIP-column matrix after symmetry reduction and let $N$ be the exact integer matrix reconstructed from Figures 13 and 14. The required scalar is
$$
R_\kappa=\frac{\kappa(C)}{\kappa(N)}.
$$

Figure 14 is printed to two significant figures. For a row whose Figure 13 total is $n_i$, an integer candidate $N_{ij}$ is admissible only when $N_{ij}/n_i$, formatted with the two-significant-figure rule $\texttt{\%.2g}$, reproduces the corresponding displayed Figure 14 value under the same rule. The entire row must consist of nonnegative integers summing to $n_i$; do not independently round cells without enforcing the row total.

## Problem

A synthetic Mg-Nd ordering panel is defined by the parent-site symmetry generators, candidate Burgers pathways and shuffled DFT and MLIP energy profiles in the Scientific Background. Apply the ordering-dependent pathway reduction and the six-class energy-landscape convention from the declared source to construct the symmetry-distinct synthetic DFT-versus-MLIP confusion matrix.

Read the DFT class counts in Figure 13 and the row-normalized matrix in Figure 14, treating every Figure 14 entry as printed to two significant figures, then reconstruct the unique nonnegative integer paper matrix while enforcing every Figure 13 row total.

For the synthetic and paper matrices, calculate unweighted multiclass Cohen's kappa from the observed agreement and the row-column marginal chance agreement defined in the Scientific Background. Return the chance-corrected agreement retention $R_\kappa=\kappa_{\mathrm{synthetic}}/\kappa_{\mathrm{paper}}$, not either raw agreement fraction.

In the reasoning, report the decisive symmetry reduction, the twelve retained decoration-pathway keys, the Figure 8 class convention, both derived confusion matrices and their marginals, the two-significant-figure reconstruction check, both kappa values and their ratio while keeping source-derived and synthetic quantities separate. The two computed confusion matrices, their marginals and the twelve retained keys are required checkpoint outputs; they are derived results, not the input matrices prohibited by the Output Format Requirements below.
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

in_build_finite_group_action

Goal
----
Build the complete group of site permutations from the supplied generators. The input may contain direct permutation generators. It may also contain rotation and translation data for periodic sites. When periodic data are given, first convert every affine operation into a site permutation using exact integer calculations. After collecting all valid generators, repeatedly compose them until no new permutation is obtained. Return every different group element once. The identity permutation must be included.

```python
import numpy as np


def build_finite_group_action(
    generator_permutations: np.ndarray,
    *,
    supercell_matrix=None,
    site_numerators=None,
    coordinate_denominator=1,
    generator_rotations=None,
    generator_translation_numerators=None,
) -> np.ndarray:
    """
    Compile a finite site-permutation group from explicit and periodic-affine
    generators.

    Parameters
    ----------
    generator_permutations : np.ndarray
        Integer array of shape (n_explicit, n_sites). Row g stores the image
        of each site under one explicit generator: row_g[i] = g(i).
        Use shape (0, n_sites) when no explicit generator is supplied.
    supercell_matrix : np.ndarray or None
        Optional nonsingular integer array of shape (d, d). Its columns span
        the periodic supercell lattice H Z^d.
    site_numerators : np.ndarray or None
        Optional integer array of shape (n_sites, d). Row i represents the
        periodic coordinate x_i = site_numerators[i] /
        coordinate_denominator.
    coordinate_denominator : int
        Positive common denominator for site coordinates and affine
        translations.
    generator_rotations : np.ndarray or None
        Optional integer array of shape (n_affine, d, d) containing
        unimodular lattice-preserving affine rotation matrices.
    generator_translation_numerators : np.ndarray or None
        Optional integer array of shape (n_affine, d). Together with the
        corresponding rotation, row g acts on coordinate numerators as
        s -> R_g s + t_g.

    Returns
    -------
    group_permutations : np.ndarray
        Integer array of shape (group_order, n_sites) containing every
        generated site permutation exactly once, sorted lexicographically.
    """
    return np.empty(
        (0, generator_permutations.shape[1]),
        dtype=int,
    )
```

### Step 2

in_enumerate_decoration_orbit_data

Goal
----
A chemical ordering can be represented by a binary occupation vector. For the Mg-Nd panel, a value of 1 means Nd and a value of 0 means Mg. Two decorations should not be counted separately when a symmetry operation of the parent structure converts one into the other. All decorations connected in this way form one symmetry orbit. One representative is selected from every orbit. Here the representative is the decoration having the largest binary code. The orbit size tells how many decorations are related to the representative by parent symmetry. The symmetry operations that leave the representative unchanged form its stabilizer. For a finite group, the orbit size and stabilizer size are related by the orbit-stabilizer theorem. These values are needed in the next step because chemical ordering can reduce the symmetry available to a transformation pathway.

```python
import numpy as np


def enumerate_decoration_orbit_data(
    group_action: np.ndarray,
    occupation_counts: np.ndarray,
) -> np.ndarray:
    """
    Enumerate symmetry-distinct binary decorations and their stabilizer data.

    Parameters
    ----------
    group_action : np.ndarray
        Integer array of shape (group_order, n_sites). Each row contains one
        parent site permutation in image notation.
    occupation_counts : np.ndarray
        One-dimensional integer array containing the allowed numbers of
        occupied sites.

    Returns
    -------
    decoration_orbit_data : np.ndarray
        Integer array of shape (n_decorations, 3 + group_order).
        Column 0 contains the canonical binary decoration code.
        Column 1 contains the occupation count.
        Column 2 contains the parent-symmetry orbit size.
        The remaining columns contain 0/1 stabilizer-membership flags in
        the same order as the rows of group_action.
    """
    return np.empty(
        (
            0,
            3 + group_action.shape[0],
        ),
        dtype=int,
    )
```

### Step 3

in_reduce_undirected_axis_variants

Goal
----
Reduce the supplied undirected axis pairs separately for every decoration. Read the stabilizer flags from decoration_orbit_data, apply only the active group operations to both sites of every axis pair, and treat the mapped pair as unordered. Match every mapped pair back to axis_pairs, build the complete axis orbits, and keep the smallest label from each orbit. Return the canonical decoration code, occupation count, retained axis label and stabilizer-orbit size.

```python
import numpy as np


def reduce_undirected_axis_variants(
    decoration_orbit_data: np.ndarray,
    group_action: np.ndarray,
    axis_pairs: np.ndarray,
) -> np.ndarray:
    """
    Reduce the supplied undirected axis variants using the stabilizer
    of each chemical decoration.

    Parameters
    ----------
    decoration_orbit_data : np.ndarray
        Integer array of shape (n_decorations, 3 + group_order).
        Column 0 contains the canonical decoration code.
        Column 1 contains the occupation count.
        Column 2 contains the parent-symmetry orbit size.
        The remaining columns contain one 0/1 stabilizer flag for
        every row of group_action.
    group_action : np.ndarray
        Integer array of shape (group_order, n_sites).
        Each row is one permutation of the parent-site indices.
    axis_pairs : np.ndarray
        Integer array of shape (n_axes, 2).
        Each row contains one unordered pair of parent-site indices
        representing a candidate transformation axis.

    Returns
    -------
    retained_axis_data : np.ndarray
        Integer array of shape (n_retained, 4).
        Each row contains the canonical decoration code, occupation
        count, retained axis label and stabilizer-orbit size.
    """
    return np.empty(
        (0, 4),
        dtype=int,
    )
```

### Step 4

in_align_retained_profile_rows

Goal
----
Match every retained axis row with its energy-profile row. Use the canonical decoration code and retained axis label as the key. Keep the first four columns of retained_axis_data and append the matching DFT profile followed by the matching MLIP profile. Preserve the order of retained_axis_data. Every retained key must occur exactly once in profile_keys.

```python
import numpy as np


def align_retained_profile_rows(
    retained_axis_data: np.ndarray,
    profile_keys: np.ndarray,
    dft_profiles: np.ndarray,
    mlip_profiles: np.ndarray,
) -> np.ndarray:
    """
    Match every retained decoration-axis pair with its DFT and MLIP
    energy-profile row.

    Parameters
    ----------
    retained_axis_data : np.ndarray
        Integer array of shape (n_retained, 4).
        Each row contains the canonical decoration code, occupation
        count, retained axis label and stabilizer-orbit size.
    profile_keys : np.ndarray
        Integer array of shape (n_profiles, 2).
        Each row contains a canonical decoration code and pathway
        label identifying one energy-profile row.
    dft_profiles : np.ndarray
        Numerical array of shape (n_profiles, n_samples).
        Each row contains one DFT transformation-energy profile.
    mlip_profiles : np.ndarray
        Numerical array of shape (n_profiles, n_samples).
        Each row contains the MLIP profile paired with the DFT profile
        having the same profile key.

    Returns
    -------
    aligned_profile_rows : np.ndarray
        Floating-point array of shape
        (n_retained, 4 + 2 * n_samples).
        The first four columns contain retained-axis metadata,
        followed by the matched DFT profile and MLIP profile.
    """
    return np.empty(
        (
            0,
            4 + 2 * dft_profiles.shape[1],
        ),
        dtype=float,
    )
```

### Step 5

in_classify_transformation_landscapes

Goal
----
Classify the DFT and MLIP profiles contained in every aligned row. The first four columns are metadata. Split the remaining values into equal DFT and MLIP profiles. Use the six profile-shape rules and the supplied class_labels. Return the four metadata columns followed by the DFT class and MLIP class. Reject a profile when it does not match exactly one allowed class.

```python
import numpy as np


def classify_transformation_landscapes(
    aligned_profile_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """
    Assign one transformation-landscape class to every retained DFT
    profile and every retained MLIP profile.

    Parameters
    ----------
    aligned_profile_rows : np.ndarray
        Numerical array of shape (n_retained, 4 + 2 * n_samples).
        The first four columns contain retained-axis metadata.
        The next n_samples values contain the DFT profile and the last
        n_samples values contain the paired MLIP profile.
    class_labels : np.ndarray
        One-dimensional integer array of length 6.
        The entries give the labels used for the six transformation
        landscape classes in class-definition order.

    Returns
    -------
    classified_rows : np.ndarray
        Integer array of shape (n_retained, 6).
        Columns 0 through 3 contain the retained-axis metadata.
        Column 4 contains the DFT class and column 5 contains the
        MLIP class.
    """
    return np.empty(
        (0, 6),
        dtype=int,
    )
```

### Step 6

in_score_classification_agreement

Goal
----
Build a confusion matrix from the last two columns of classified_rows. Use the ordering supplied in class_labels. Add row totals as the last column and column totals as the last row. Store the exact-class agreement fraction in the bottom-right element. The output must therefore have one more row and one more column than the number of class labels.

```python
import numpy as np


def score_classification_agreement(
    classified_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """
    Build the DFT-versus-MLIP confusion matrix and calculate the
    exact-class agreement fraction.

    Parameters
    ----------
    classified_rows : np.ndarray
        Integer array of shape (n_retained, 6).
        The final two columns contain the DFT and MLIP class labels
        for every retained pathway.
    class_labels : np.ndarray
        One-dimensional integer array containing the allowed class
        labels in the required matrix order.

    Returns
    -------
    score_matrix : np.ndarray
        Floating-point array of shape
        (n_classes + 1, n_classes + 1).
        The upper-left block is the DFT-row/MLIP-column confusion
        matrix. The final column contains row totals. The final row
        contains column totals. The bottom-right element contains
        the exact-class agreement fraction.
    """
    return np.empty(
        (
            class_labels.size + 1,
            class_labels.size + 1,
        ),
        dtype=float,
    )
```

### Step 7

07_in_reconstruct_integer_confusion_matrix

Goal
----
Reconstruct a nonnegative integer confusion matrix from row totals and arow-normalized matrix printed to a specified number of significant figures.

```python
import numpy as np


def reconstruct_integer_confusion_matrix(
    row_counts: np.ndarray,
    row_normalized: np.ndarray,
    significant_figures: np.ndarray,
) -> np.ndarray:
    return np.empty_like(row_normalized, dtype=int)
```

### Step 8

in_compute_kappa_retention

Goal
----
Calculate the ratio of unweighted multiclass Cohen kappa for a synthetic confusion matrix and a paper confusion matrix.

```python
import numpy as np


def compute_kappa_retention(
    synthetic_score_matrix: np.ndarray,
    paper_confusion: np.ndarray,
) -> float:
    """
    Calculate chance-corrected agreement retention from two confusion matrices.

    Parameters
    ----------
    synthetic_score_matrix : np.ndarray
        Augmented square matrix produced by score_classification_agreement.
        The upper-left block contains synthetic counts, the last column and
        last row contain row and column totals, and the bottom-right entry is
        the raw agreement.
    paper_confusion : np.ndarray
        Nonnegative integer square confusion matrix with the same number of
        classes as the synthetic count block.

    Returns
    -------
    float
        Finite value kappa_synthetic / kappa_paper.

    Raises
    ------
    ValueError
        If the synthetic matrix is not a valid augmented square matrix, if its
        stored row totals, column totals or raw agreement disagree with its
        count block, if paper_confusion has the wrong class dimension or is not
        a valid count matrix, if either kappa denominator is invalid, if the
        paper kappa is zero, or if the final ratio is not finite.
    """
    return float("nan")
```

### Step 9

Run the complete Mg-Nd kappa pipeline

Goal
----
Run the complete Mg-Nd ordering and class-comparison calculation and return the chance-corrected agreement retention.

```python
import numpy as np


def resolve_mg_nd_kappa_retention(
    generator_permutations: np.ndarray,
    occupation_counts: np.ndarray,
    axis_pairs: np.ndarray,
    profile_keys: np.ndarray,
    dft_profiles: np.ndarray,
    mlip_profiles: np.ndarray,
    class_labels: np.ndarray,
    paper_row_counts: np.ndarray,
    paper_row_normalized: np.ndarray,
    paper_significant_figures: np.ndarray,
) -> float:
    return float("nan")
```
