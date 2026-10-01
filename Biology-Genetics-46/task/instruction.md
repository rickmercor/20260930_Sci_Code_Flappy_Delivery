# Block-adjusted assembly cost of a probabilistic tetraploid haplotype phasing

## Background

A haplotype is the sequence of alleles carried along one physical chromosome, and almost everything that makes genetic variation interpretable, which variants travel together, which regulatory alleles sit in cis with which coding ones, how a lineage descends from its ancestors, is a statement about haplotypes rather than about genotypes. Experimental methods that read haplotypes directly are expensive and awkward, so haplotypes are normally inferred, and one of the two ways of doing that, haplotype assembly, works from the aligned sequencing fragments of a single individual: a fragment that calls alleles at two or more heterozygous sites is direct evidence that those alleles lie on the same chromosome. In a diploid the problem is already combinatorial, and it becomes much harder in a polyploid, where the crops that matter economically, wheat, potato, strawberry, carry four, six or eight copies of each chromosome. Two things break at once. Resolving one haplotype no longer resolves its complement, so most of the genotype constraint has to be spent before it starts to help, and long tracts duplicated from the same or a closely related ancestral genome mean a fragment often cannot be assigned to a chromosome of origin at all, however good the sequencing.

The dominant response has been to make an assignment anyway. Assemblers built on graph cuts, read partitioning or greedy majority voting all commit, at some point, to a single hard allocation of fragments to haplotypes, and then optimise a combinatorial objective on top of it. That is efficient, and it is also exactly where polyploid data are weakest: when several haplotypes agree over a long stretch, the allocation is close to a coin flip, and a method that commits to one of the outcomes carries the coin flip forward silently into everything downstream. The alternative is to keep the ambiguity as a distribution and to propagate it, which is what a probabilistic assembler does. Because the space of whole-locus phasings is exponential in both the number of sites and the ploidy, this only becomes tractable if the model is written over small local objects, the phasings of individual pairs of heterozygous sites, each constrained to reproduce the called genotype at both,  with a graphical structure that links pairs sharing a site. Inference on that structure is then standard: a maximum-a-posteriori decoding when a single phasing is wanted, or repeated sampling when the point of the exercise is to say how uncertain the phasing is.

Local phasings, however, do not simply concatenate. Haplotypes are biologically unordered, so a phasing is only defined up to a permutation of haplotype labels, and two overlapping local phasings can be made consistent at their shared site by more than one permutation. Turning a set of locally decoded phasings into one global haplotype matrix is therefore a matching problem in its own right, and it is solved incrementally: the assembly grows one variant at a time from the region already phased, and each new variant is assigned by weighing the candidate extensions against the fragment evidence and against what the graphical model has already concluded. Where the fragments run out the assembly simply stops and restarts with a fresh, independent labelling, so a real output is not one haplotype matrix but a series of phased blocks, possibly with sites that no fragment pair ever reached and which stay unresolved.

That fragmentation is what makes evaluating polyploid assemblies awkward. The classical criterion, minimum error correction, counts the fragment alleles that would have to be flipped for every fragment to sit perfectly on one reconstructed haplotype, and taken literally it rewards exactly the wrong behaviour: an assembler that cuts the locus into one-site blocks can drive the count to zero without having phased anything, and an assembler that declines to resolve a site pays nothing for the omission. Comparing methods that fragment differently, or that phase different fractions of the locus, therefore requires the criterion to be generalised so that both fragmentation and incompleteness carry an explicit price. The number this task asks for is that generalised criterion, evaluated on the assembly the probabilistic pipeline produces from one small tetraploid locus.

## Problem

Assembling haplotypes from aligned sequencing fragments is harder in a polyploid genome than in a diploid one for two reasons: the haplotypes are not complementary, so $K-1$ of them have to be resolved before the called genotype fixes the last, and long stretches of near-identical sequence make the chromosome of origin of a fragment genuinely ambiguous rather than merely unknown. Probabilistic assemblers answer this by refusing to commit to a read partition at all: they put a distribution over the genotype-consistent phasings of *pairs* of heterozygous SNPs, propagate it over the graph those pairs induce, and only afterwards collapse the local phasings into one global haplotype matrix, whose quality is then reported as an error-correction cost. I have one tetraploid locus of exactly this kind, and what I want out of it is that cost.

The locus carries $L = 15$ heterozygous biallelic SNPs at ploidy $K = 4$, the per-base sequencing error rate is $\varepsilon = 0.02$, and the called counts of the alternate allele are

$$g = (3,\ 3,\ 2,\ 1,\ 1,\ 3,\ 3,\ 2,\ 2,\ 3,\ 2,\ 3,\ 1,\ 1,\ 2).$$

The 33 aligned fragments are listed below, one string per fragment, with `0` for the reference allele, `1` for the alternate allele and `-` where the fragment makes no call:

```
0-10-----------    -----00000-----    ----------00-0-
00111----------    -----0011------    ----------000--
011------------    -----0101------    ----------0000-
0110-----------    -----100-------    ----------0000-
10100----------    -----100-------    ----------1100-
1100-----------    -----1010------    -----------000-
1110-----------    -----11101-----    -----------100-
11110----------    -----11111-----    -----------111-
-010-----------    ------0101-----    --------------1
-111-----------    ------100------    --------------0
--110----------    ------1001-----    --------------1
```

Build the pair-phasing graph these fragments induce, place the read-evidence potentials on it, taking each factor as a sum over the fragments that call an allele at every one of its positions rather than as a product over them, and take the maximum-a-posteriori phasing of every vertex under the joint distribution the graph carries, one connected component at a time; then stitch those local phasings into one global haplotype matrix by the greedy connectivity-driven, variant-by-variant procedure these assemblers use, working through the SNP pairs in ascending genomic order, scoring the genotype-consistent candidates available for each newly phased variant by a weighted combination of read likelihood, minimum error correction and agreement with the phasings already inferred on the graph, with weights $1/12$, $10/12$ and $1/12$ respectively and each of the three quantities rescaled to the unit interval within the candidate set of that iteration before the combination is formed. Your final answer must be a single number: the block-adjusted minimum error correction of the resulting assembly, the evaluation criterion that keeps a raw error-correction count comparable across assemblies that differ in how far they fragment the locus and in how much of it they leave unresolved, quoted to at least six significant figures. Report also, in one line each, how many vertices the pair-phasing graph has, how the assembly divides the locus into phasing blocks, which SNPs it leaves unresolved, the two conventions that make the error-correction criterion comparable in the sense just described and what each of them contributes here, and what the same locus would cost instead if the candidates were scored on read likelihood alone; those are the few scalars that determine the final number and the few that establish it was computed rather than estimated; no further scalars are required, though you should briefly set out the model you built and the conventions you adopted wherever the construction is not forced.

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

01_enumerate_valid_phasings

Goal
----
Enumerate, in a canonical order, every haplotype phasing of a short run of heterozygous biallelic SNPs that is consistent with the called polyploid genotypes.

```python
import numpy as np


def enumerate_valid_phasings(ploidy: int, genotypes: np.ndarray) -> np.ndarray:
    """Enumerate the genotype-consistent phasings of a short run of SNPs.

    A phasing is returned in canonical form: its rows are sorted in ascending
    lexicographic order, so that two phasings differing only by a permutation
    of haplotype labels are represented by the same matrix. The enumeration is
    ordered by ascending lexicographic order of the row-major flattened matrix.

    Parameters
    ----------
    ploidy : int
        Number of haplotypes K carried by the organism (ploidy >= 1).
    genotypes : np.ndarray
        One-dimensional integer array of length P holding, for each of the P
        positions, the called count of alternate alleles across the K
        haplotypes. Every entry lies between 0 and ``ploidy`` inclusive and P
        is between 1 and 4 inclusive.

    Returns
    -------
    phasings : np.ndarray
        Integer array of shape (M, ploidy, P) with entries in {0, 1}. Entry
        ``phasings[m, k, p]`` is the allele that haplotype k carries at the
        p-th position under the m-th phasing, and every phasing satisfies
        ``phasings[m].sum(axis=0) == genotypes``.

    Raises
    ------
    ValueError
        If ``ploidy`` is not an integer greater than zero, if ``genotypes`` is
        not a one-dimensional integer-valued array whose length is between one
        and four inclusive, or if any entry of ``genotypes`` lies outside the
        range from zero to ``ploidy`` inclusive.
    """
    return phasings  # placeholder
```

### Step 2

02_compute_phasing_potentials

Goal
----
Score every candidate phasing of a short run of SNPs against the aligned fragments that cover the whole run, producing the factor potential the phasing carries in the graphical model.

```python
import numpy as np


def compute_phasing_potentials(reads: np.ndarray, phasings: np.ndarray,
                               positions: np.ndarray,
                               error_rate: float) -> np.ndarray:
    """Compute the read-evidence potential of each candidate phasing.

    A fragment contributes only if it has a called allele at every position of
    ``positions``; if no fragment does, the run carries no read evidence and
    every phasing receives the same potential of one.

    Parameters
    ----------
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0 for the reference allele, 1 for the alternate allele and
        -1 where the fragment has no called allele.
    phasings : np.ndarray
        Integer array of shape (M, K, P) with entries in {0, 1} holding the
        candidate phasings of the run.
    positions : np.ndarray
        One-dimensional integer array of length P holding the distinct SNP
        indices of the run, each a valid column index of ``reads``.
    error_rate : float
        Per-base sequencing error rate, strictly between zero and one.

    Returns
    -------
    potentials : np.ndarray
        Array of shape (M,) of non-negative floats, the potential of each
        phasing in the order the phasings were supplied.

    Raises
    ------
    ValueError
        If ``reads`` is not a two-dimensional integer-valued array with entries
        in {-1, 0, 1}, if ``phasings`` is not a three-dimensional
        integer-valued array with entries in {0, 1}, if ``positions`` is not a
        one-dimensional array of distinct valid column indices of ``reads``
        whose length equals the last axis of ``phasings``, or if ``error_rate``
        is not a finite number strictly between zero and one.
    """
    return potentials  # placeholder
```

### Step 3

03_build_snp_line_graph

Goal
----
Turn the aligned fragment matrix into the ordered vertex set of the SNP line graph, the structure on which the phasing distribution is defined.

```python
import numpy as np


def build_snp_line_graph(reads: np.ndarray) -> np.ndarray:
    """Build the ordered vertex list of the SNP line graph of a fragment set.

    Parameters
    ----------
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0 for the reference allele, 1 for the alternate allele and
        -1 where the fragment has no called allele.

    Returns
    -------
    nodes : np.ndarray
        Integer array of shape (U, 2). Row t holds the two SNP indices of the
        t-th vertex, the smaller index first. The rows are sorted in ascending
        order of the first index, ties broken by the second index. A fragment
        set that joins no pair of SNPs yields an array of shape (0, 2).

    Raises
    ------
    ValueError
        If ``reads`` is not a non-empty two-dimensional integer-valued array
        with entries in {-1, 0, 1}.
    """
    return nodes  # placeholder
```

### Step 4

04_build_transition_matrix

Goal
----
Convert the potential of the joint run spanned by two adjacent vertices into the normalised conditional distribution of the second vertex phasing given the first.

```python
import numpy as np


def build_transition_matrix(parent_positions: np.ndarray,
                            parent_phasings: np.ndarray,
                            child_positions: np.ndarray,
                            child_phasings: np.ndarray,
                            joint_positions: np.ndarray,
                            joint_phasings: np.ndarray,
                            joint_potentials: np.ndarray) -> np.ndarray:
    """Build the conditional distribution of a child phasing given a parent one.

    A joint phasing supports the pair (i, j) when the columns it holds at the
    parent positions reproduce parent phasing i as a multiset of haplotype rows
    and, at the same time, the columns it holds at the child positions
    reproduce child phasing j as a multiset of haplotype rows. Rows of the
    result whose total support vanishes are set to the uniform distribution.

    Parameters
    ----------
    parent_positions : np.ndarray
        One-dimensional integer array of the SNP indices of the parent vertex.
    parent_phasings : np.ndarray
        Integer array of shape (Mp, K, Pp) of canonical parent phasings, whose
        rows are sorted in ascending lexicographic order.
    child_positions : np.ndarray
        One-dimensional integer array of the SNP indices of the child vertex.
    child_phasings : np.ndarray
        Integer array of shape (Mc, K, Pc) of canonical child phasings, whose
        rows are sorted in ascending lexicographic order.
    joint_positions : np.ndarray
        One-dimensional integer array of the SNP indices of the joint run; it
        contains every parent and every child position.
    joint_phasings : np.ndarray
        Integer array of shape (Mj, K, Pj) of canonical phasings of the joint
        run, in the same column order as ``joint_positions``.
    joint_potentials : np.ndarray
        Array of shape (Mj,) of non-negative floats, the potential of each
        joint phasing.

    Returns
    -------
    transition : np.ndarray
        Array of shape (Mp, Mc) of non-negative floats whose rows each sum to
        one; entry (i, j) is the probability of child phasing j given parent
        phasing i.

    Raises
    ------
    ValueError
        If any phasing array is not a three-dimensional integer-valued array
        with entries in {0, 1}, if the three ploidies disagree, if any position
        array is not one-dimensional with distinct entries matching the width
        of its phasings, if the parent or child positions are not all contained
        in ``joint_positions``, or if ``joint_potentials`` is not a
        one-dimensional array of finite non-negative entries of length Mj.
    """
    return transition  # placeholder
```

### Step 5

05_decode_map_phasings

Goal
----
Decode one phasing per vertex of the SNP line graph by maximising the joint score of the directed phasing model over each connected component.

```python
import numpy as np


def decode_map_phasings(nodes: np.ndarray, node_potentials: list,
                        edges: np.ndarray, transitions: list) -> np.ndarray:
    """Decode the maximum-a-posteriori phasing index of every vertex.

    The score of an assignment of one phasing index to every vertex is the
    product of the potential each vertex gives its own index with the
    transition each edge gives the pair of indices at its endpoints. Two
    vertices lie in the same component when the undirected graph induced by
    ``edges`` connects them, components share no edge, and the score is
    therefore maximised over each component separately; a vertex belonging to
    no edge is a component on its own. Among assignments of equal score the one
    whose tuple of state indices, read in ascending vertex order, is smallest
    lexicographically is returned.

    Parameters
    ----------
    nodes : np.ndarray
        Integer array of shape (U, 2) holding the SNP pair of each vertex, in
        ascending order of the first SNP, ties broken by the second.
    node_potentials : list
        Sequence of U one-dimensional arrays of strictly positive floats;
        entry t holds the potential of every phasing of vertex t.
    edges : np.ndarray
        Integer array of shape (E, 2) of directed edges, each row holding a
        parent vertex index strictly smaller than its child vertex index.
    transitions : list
        Sequence of E two-dimensional arrays of finite non-negative floats;
        entry e has shape (Mp, Mc) for the parent and child of edge e.

    Returns
    -------
    states : np.ndarray
        Integer array of shape (U,) holding, for each vertex, the index of its
        decoded phasing within that vertex's own potential array.

    Raises
    ------
    ValueError
        If ``nodes`` is not a two-dimensional integer-valued array with two
        columns, if ``node_potentials`` does not hold one non-empty
        one-dimensional array of finite strictly positive entries per vertex,
        if ``edges`` is not a two-dimensional integer-valued array with two
        columns whose entries are valid vertex indices with the parent index
        strictly smaller than the child index, if the number or the shape of
        the transition matrices does not match the edges they belong to, if any
        transition entry is not finite and non-negative, or if any component
        admits more than 2000000 assignments.
    """
    return states  # placeholder
```

### Step 6

06_select_next_position

Goal
----
Choose the next variant to add to the growing global assembly, or open a new phasing block when the current one is exhausted, from the state of the SNP line graph alone.

```python
import numpy as np


def select_next_position(nodes: np.ndarray, node_phased: np.ndarray,
                         position_phased: np.ndarray) -> np.ndarray:
    """Decide the next move of the greedy variant-by-variant assembly.

    Two vertices of the SNP line graph are adjacent when they share exactly one
    SNP. The frontier is the set of unprocessed vertices that are adjacent to at
    least one processed vertex and that still carry at least one unphased SNP.

    When the frontier is not empty the chosen variant is the unphased SNP that
    appears in the largest number of frontier vertices, ties broken in favour of
    the smallest SNP index. When it is empty but some unprocessed vertex still
    carries an unphased SNP, a new block is opened at the first such vertex in
    the given vertex order. Otherwise no move remains.

    Parameters
    ----------
    nodes : np.ndarray
        Integer array of shape (U, 2) holding the SNP pair of each vertex in
        topological order.
    node_phased : np.ndarray
        Array of shape (U,) of zeros and ones, one where the vertex has already
        been processed.
    position_phased : np.ndarray
        Array of shape (n_snps,) of zeros and ones, one where the SNP has
        already been assigned in the global haplotype matrix.

    Returns
    -------
    move : np.ndarray
        Integer array of shape (3 + U,). Entry 0 is the action: 0 to phase a
        variant, 1 to open a new block, 2 when nothing remains. Entry 1 is the
        chosen SNP index for action 0 and -1 otherwise. Entry 2 is the chosen
        vertex index for action 1 and -1 otherwise. Entries 3 onward are one
        for each frontier vertex and zero elsewhere, and are all zero unless
        the action is 0.

    Raises
    ------
    ValueError
        If ``nodes`` is not a two-dimensional array with two columns of finite
        integer-valued entries, if ``node_phased`` or ``position_phased`` is not
        a non-empty one-dimensional array of zeros and ones of the matching
        length, or if any entry of ``nodes`` is not a valid index into
        ``position_phased``.
    """
    return move  # placeholder
```

### Step 7

07_generate_position_candidates

Goal
----
Enumerate every haplotype matrix over the SNP window spanned by the frontier vertices that both reproduces the alleles already assigned and respects the called genotypes.

```python
import numpy as np


def generate_position_candidates(positions: np.ndarray, haplotypes: np.ndarray,
                                 genotypes: np.ndarray) -> np.ndarray:
    """Enumerate the admissible haplotype matrices over a window of SNPs.

    Candidates are listed in ascending lexicographic order of the sequence of
    their free columns, each free column being read from the first haplotype
    row to the last and the earlier free columns varying more slowly than the
    later ones.

    Parameters
    ----------
    positions : np.ndarray
        One-dimensional integer array of P strictly increasing SNP indices, the
        window the candidates span.
    haplotypes : np.ndarray
        Integer array of shape (K, n_snps) holding the global haplotype matrix
        built so far, with 0 and 1 for assigned alleles and -1 for a SNP that
        has not been assigned yet. A column is treated as assigned only when
        every one of its K entries is non-negative.
    genotypes : np.ndarray
        One-dimensional integer array of length n_snps holding the called count
        of alternate alleles at each SNP.

    Returns
    -------
    candidates : np.ndarray
        Integer array of shape (n_candidates, K, P) with entries in {0, 1}.
        Every candidate reproduces ``haplotypes`` on the assigned columns of
        the window and has column sums equal to the genotypes of the window.

    Raises
    ------
    ValueError
        If ``positions`` is not a one-dimensional strictly increasing array of
        valid SNP indices, if ``haplotypes`` is not a two-dimensional
        integer-valued array with entries in {-1, 0, 1}, if ``genotypes`` is not
        a one-dimensional integer-valued array of length n_snps whose entries
        lie between zero and K inclusive, or if an already-assigned column of
        the window contradicts its called genotype.
    """
    return candidates  # placeholder
```

### Step 8

08_score_phasing_candidates

Goal
----
Rank the admissible haplotype matrices of a window by combining their read likelihood, their error-correction cost and their agreement with the phasings decoded on the graph.

```python
import numpy as np


def score_phasing_candidates(candidates: np.ndarray, positions: np.ndarray,
                             reads: np.ndarray, neighbor_pairs: np.ndarray,
                             neighbor_phasings: np.ndarray, error_rate: float,
                             weights: np.ndarray) -> np.ndarray:
    """Score the admissible haplotype matrices of one window.

    Every fragment with a called allele at one or more window positions
    contributes. For such a fragment let d_k be the number of those called
    alleles that differ from haplotype k of the candidate and let c be the
    number of them. The likelihood term sums the logarithm of the sum over k of
    the per-base error model raised to d_k times its complement raised to
    c - d_k. The error-correction term sums the smallest d_k. The agreement
    term counts the supplied neighbour vertices whose two SNPs both lie in the
    window and whose decoded phasing equals, as a multiset of haplotype rows,
    the candidate restricted to those two SNPs.

    Each of the three terms is rescaled to the unit interval across the
    candidate set, a term that takes a single value across the set being
    rescaled to zero throughout. The score adds the rescaled likelihood and
    agreement terms with their weights and subtracts the rescaled
    error-correction term with its weight.

    Parameters
    ----------
    candidates : np.ndarray
        Integer array of shape (n_candidates, K, P) with entries in {0, 1}.
    positions : np.ndarray
        One-dimensional integer array of the P distinct SNP indices of the
        window, in the column order of ``candidates``.
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0, 1 and -1 for reference, alternate and uncalled.
    neighbor_pairs : np.ndarray
        Integer array of shape (m, 2) holding the SNP pair of each neighbour
        vertex whose decoded phasing is supplied.
    neighbor_phasings : np.ndarray
        Integer array of shape (m, K, 2) holding the decoded phasing of each of
        those vertices, in the column order of ``neighbor_pairs``.
    error_rate : float
        Per-base sequencing error rate, strictly between zero and one.
    weights : np.ndarray
        One-dimensional array of three finite non-negative floats holding, in
        order, the likelihood weight, the error-correction weight and the
        agreement weight.

    Returns
    -------
    scores : np.ndarray
        Array of shape (n_candidates,) of floats, the score of each candidate
        in the order supplied.

    Raises
    ------
    ValueError
        If ``candidates`` is not a non-empty three-dimensional integer-valued
        array with entries in {0, 1}, if ``positions`` is not a one-dimensional
        array of distinct valid SNP indices whose length matches the candidate
        width, if ``reads`` is not a two-dimensional integer-valued array with
        entries in {-1, 0, 1}, if ``neighbor_pairs`` is not a two-column array
        of finite integer-valued valid SNP indices naming two distinct SNPs per
        row, if ``neighbor_phasings`` does not have shape (m, K, 2) with entries
        in {0, 1} and the same leading length as ``neighbor_pairs``, if
        ``error_rate`` is not a finite number strictly between zero and one, or
        if ``weights`` is not a one-dimensional array of three finite
        non-negative entries.
    """
    return scores  # placeholder
```

### Step 9

09_compute_block_adjusted_mec

Goal
----
Score a finished, possibly fragmented and partially phased assembly by the minimum number of read allele corrections it needs, charged for every extra block a fragment has to span.

```python
import numpy as np


def compute_block_adjusted_mec(reads: np.ndarray, haplotypes: np.ndarray,
                               blocks: np.ndarray, ploidy: int) -> float:
    """Score an assembly by its block-adjusted minimum error correction.

    A SNP is resolved when every one of the K haplotypes carries an allele at
    it. Each called allele of a fragment at an unresolved SNP costs one,
    whichever haplotype it is compared against. The resolved SNPs a fragment
    calls are grouped by the phasing block they belong to, and each group
    contributes the smallest number of mismatches any one of the K haplotypes
    leaves on it; the groups are minimised separately because the haplotype
    labels of different blocks are unrelated. The fragment is then charged one
    minus the reciprocal of the ploidy for every block it spans beyond the
    first, a fragment spanning no block being treated as spanning one. The
    total over fragments is divided by the number of called alleles in the
    whole fragment set.

    Parameters
    ----------
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0, 1 and -1 for reference, alternate and uncalled.
    haplotypes : np.ndarray
        Integer array of shape (K, n_snps) holding the reconstructed haplotypes,
        with -1 where a SNP was left unresolved.
    blocks : np.ndarray
        One-dimensional integer array of length n_snps holding the phasing
        block each SNP belongs to, numbered from one, with 0 for a SNP that
        belongs to no block. A SNP carries a positive label exactly when it is
        resolved.
    ploidy : int
        Number of haplotypes K, which must equal the first axis of
        ``haplotypes`` and be greater than zero.

    Returns
    -------
    mec : float
        The block-adjusted minimum error correction of the assembly, as a
        native Python float.

    Raises
    ------
    ValueError
        If ``reads`` or ``haplotypes`` is not a two-dimensional integer-valued
        array with entries in {-1, 0, 1}, if the two disagree on the number of
        SNPs, if ``blocks`` is not a one-dimensional array of non-negative
        integers of that same length, if some SNP carries a positive block label
        without being resolved or is resolved without carrying one, if
        ``ploidy`` is not an integer greater than zero matching the first axis
        of ``haplotypes``, or if no fragment calls any allele at all.
    """
    return mec  # placeholder
```

### Step 10

10_run_phapcompass_short_pipeline

Goal
----
Chain the sub-problem functions 01-09 end to end on the tetraploid fragment testbed and return the block-adjusted minimum error correction of the assembly they produce.

```python
import numpy as np


def run_phapcompass_short_pipeline(fragments: tuple = (
        "0-10-----------", "00111----------", "011------------",
        "0110-----------", "10100----------", "1100-----------",
        "1110-----------", "11110----------", "-010-----------",
        "-111-----------", "--110----------", "-----00000-----",
        "-----0011------", "-----0101------", "-----100-------",
        "-----100-------", "-----1010------", "-----11101-----",
        "-----11111-----", "------0101-----", "------100------",
        "------1001-----", "----------00-0-", "----------000--",
        "----------0000-", "----------0000-", "----------1100-",
        "-----------000-", "-----------100-", "-----------111-",
        "--------------1", "--------------0", "--------------1"),
        genotypes: tuple = (3, 3, 2, 1, 1, 3, 3, 2, 2, 3, 2, 3, 1, 1, 2),
        ploidy: int = 4,
        error_rate: float = 0.02,
        likelihood_weight: float = 1.0 / 12.0,
        mec_weight: float = 10.0 / 12.0,
        agreement_weight: float = 1.0 / 12.0) -> float:
    """Assemble a polyploid haplotype set from fragments and score the result.

    Parameters
    ----------
    fragments : tuple
        Sequence of equal-length strings over the alphabet {'0', '1', '-'}, one
        per aligned fragment, giving the reference allele, the alternate allele
        or no call at each of the heterozygous SNP positions in order.
    genotypes : tuple
        Sequence of integers of the same length as each fragment, holding the
        called count of alternate alleles at each SNP.
    ploidy : int
        Number of haplotypes to reconstruct (ploidy >= 1).
    error_rate : float
        Per-base sequencing error rate, strictly between zero and one.
    likelihood_weight : float
        Non-negative weight of the rescaled read-likelihood term in the
        candidate score.
    mec_weight : float
        Non-negative weight of the rescaled error-correction term, which enters
        the candidate score with a negative sign.
    agreement_weight : float
        Non-negative weight of the rescaled inference-agreement term.

    Returns
    -------
    mec : float
        The block-adjusted minimum error correction of the reconstructed
        assembly, as a native Python float.

    Raises
    ------
    ValueError
        If ``fragments`` is not a non-empty sequence of equal-length strings
        over {'0', '1', '-'}, if ``genotypes`` does not hold one integer per SNP
        in the range from zero to ``ploidy`` inclusive, if ``ploidy`` is not an
        integer greater than zero, if ``error_rate`` is not a finite number
        strictly between zero and one, or if any of the three weights is not a
        finite non-negative number.
    """
    return mec  # placeholder
```
