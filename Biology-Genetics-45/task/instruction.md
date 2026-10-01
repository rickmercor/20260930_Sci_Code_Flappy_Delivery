# Biology-Genetics-45

## Background

Biobank cohorts now hold genotypes and phenotypes for hundreds of thousands of people, and genome-wide genealogies inferred from those genotypes describe how the sampled chromosomes are related at every position. Such genealogies carry variation that was never typed or imputed, so relatedness matrices derived from them reach parts of the allele frequency spectrum that marker-based analyses miss.

The obstacle is scale. A relatedness matrix with one row and column per individual is quadratic in sample size, and fitting a mixed model against it is worse, which has held genealogy-based analyses to a few thousand samples. Work in this area therefore replaces explicit matrices with graph traversal and iterative solvers, so that only matrix-vector products are ever formed and no system is carried to convergence.

What these methods produce is an association signal for a variant after the trait has been corrected for the genetic background carried by the rest of the genome, and they are judged on whether that correction removes confounding without absorbing the signal being tested.

## Problem

An ancestral recombination graph records how sampled haplotypes are related along a chromosome, and the mutations placed on its edges include variation that genotyping arrays and imputation reference panels never observe. Testing a variant for association with a quantitative trait in a structured sample means first correcting that trait for the polygenic background the rest of the genome carries, which is what a mixed model does and what makes genealogies attractive as its basis. The inputs are one graph per chromosome with mutations on its edges, a diploid sample defined by pairs of haplotype leaves, and one phenotype vector; the number reported is the association score of a single variant under test.

Relatedness is built from the centred and frequency-scaled genotypes of the mutations the graphs carry, pooled across chromosomes, under the scaling convention this family of methods uses to put the genetic and residual variance components on a common footing. Recombination reshapes only part of a graph at a time, so a carrier set obtained at one mutation stays valid over a stretch of the sequence, and the solution should state the condition under which it does. A moment-based fit then splits the trait's variance between the two components. The correction applied to the trait is the prediction carried by every chromosome except the one under test, through their share of that same relatedness and the fitted components, so that the variants being tested are not also used to correct for themselves, and that prediction comes from solving the phenotypic covariance against the standardised phenotype by an iterative scheme that needs only products with the covariance and is run under a fixed budget of iterations rather than to convergence. The variant is scored against whatever the prediction leaves behind.

Your task is to solve one concrete deterministic example of this pipeline, using the following configuration:

- Three chromosomes with sequence coordinates spanning [0, 60), [0, 45) and [0, 50). Nodes 0 to 19 are haplotype leaves and nodes 20 to 38 are internal on each of them. Recombination means the marginal tree differs along the sequence.
- Chromosome 1 edges, as (child, parent, left, right), each stating that the child inherits the half-open interval [left, right) from the parent: (0, 20, 0, 20), (0, 21, 20, 40), (0, 20, 40, 60), (1, 20, 0, 60), (2, 21, 0, 60), (3, 21, 0, 60), (4, 22, 0, 60), (5, 22, 0, 60), (6, 23, 0, 60), (7, 23, 0, 60), (8, 24, 0, 60), (9, 24, 0, 60), (10, 25, 0, 60), (11, 25, 0, 60), (12, 26, 0, 60), (13, 26, 0, 60), (14, 27, 0, 60), (15, 27, 0, 60), (16, 28, 0, 60), (17, 28, 0, 60), (18, 29, 0, 60), (19, 29, 0, 60), (20, 30, 0, 60), (21, 30, 0, 60), (22, 31, 0, 60), (23, 31, 0, 60), (24, 32, 0, 60), (25, 32, 0, 60), (26, 33, 0, 60), (27, 33, 0, 60), (28, 34, 0, 60), (29, 34, 0, 60), (30, 35, 0, 60), (31, 35, 0, 60), (32, 36, 0, 40), (32, 37, 40, 60), (33, 36, 0, 60), (34, 37, 0, 60), (35, 37, 0, 60), (36, 38, 0, 60), (37, 38, 0, 60)
- Chromosome 1 mutations, as (child node of the carrying edge, position): (30, 5), (20, 5), (21, 5), (35, 19.999), (35, 20), (21, 25), (36, 10), (37, 39.999), (37, 40), (31, 45), (32, 30), (33, 50), (3, 12), (11, 33), (17, 55), (24, 52)
- Chromosome 2 edges, as (child, parent, left, right), each stating that the child inherits the half-open interval [left, right) from the parent: (0, 20, 0, 45), (1, 20, 0, 45), (2, 21, 0, 45), (3, 21, 0, 45), (4, 22, 0, 45), (5, 22, 0, 15), (5, 23, 15, 35), (5, 22, 35, 45), (6, 23, 0, 45), (7, 23, 0, 45), (8, 24, 0, 45), (9, 24, 0, 45), (10, 25, 0, 45), (11, 25, 0, 45), (12, 26, 0, 35), (12, 28, 35, 45), (13, 26, 0, 45), (14, 27, 0, 45), (15, 27, 0, 45), (16, 28, 0, 45), (17, 28, 0, 45), (18, 29, 0, 45), (19, 29, 0, 45), (20, 30, 0, 45), (21, 30, 0, 45), (22, 31, 0, 45), (23, 31, 0, 45), (24, 32, 0, 45), (25, 32, 0, 45), (26, 33, 0, 45), (27, 33, 0, 45), (28, 34, 0, 45), (29, 34, 0, 45), (30, 35, 0, 45), (31, 35, 0, 45), (32, 36, 0, 45), (33, 36, 0, 45), (34, 37, 0, 45), (35, 37, 0, 45), (36, 38, 0, 45), (37, 38, 0, 45)
- Chromosome 2 mutations, as (child node of the carrying edge, position): (31, 5), (22, 5), (23, 5), (35, 14.999), (35, 15), (23, 25), (36, 34.999), (36, 35), (34, 40), (30, 20), (32, 8), (28, 38), (7, 22), (14, 41), (1, 30), (26, 44)
- Chromosome 3 edges, as (child, parent, left, right), each stating that the child inherits the half-open interval [left, right) from the parent: (0, 20, 0, 50), (1, 20, 0, 50), (2, 21, 0, 50), (3, 21, 0, 50), (4, 22, 0, 50), (5, 22, 0, 50), (6, 23, 0, 50), (7, 23, 0, 50), (8, 24, 0, 50), (9, 24, 0, 50), (10, 25, 0, 50), (11, 25, 0, 50), (12, 26, 0, 50), (13, 26, 0, 50), (14, 27, 0, 50), (15, 27, 0, 50), (16, 28, 0, 50), (17, 28, 0, 50), (18, 29, 0, 10), (18, 28, 10, 30), (18, 29, 30, 50), (19, 29, 0, 50), (20, 30, 0, 50), (21, 30, 0, 50), (22, 31, 0, 50), (23, 31, 0, 50), (24, 32, 0, 50), (25, 32, 0, 50), (26, 33, 0, 50), (27, 33, 0, 50), (28, 34, 0, 50), (29, 34, 0, 50), (30, 35, 0, 30), (30, 37, 30, 50), (31, 35, 0, 50), (32, 36, 0, 50), (33, 36, 0, 50), (34, 37, 0, 50), (35, 37, 0, 50), (36, 38, 0, 50), (37, 38, 0, 50)
- Chromosome 3 mutations, as (child node of the carrying edge, position): (34, 5), (29, 5), (28, 5), (36, 9.999), (36, 10), (28, 20), (37, 29.999), (37, 30), (35, 15), (31, 35), (33, 45), (25, 40), (9, 25), (16, 48), (5, 18), (22, 12)
- The n = 10 individuals are the haplotype pairs [[0, 11], [1, 14], [2, 17], [3, 8], [4, 19], [5, 12], [6, 15], [7, 18], [9, 16], [10, 13]], in that order.
- Mutations whose minor allele count across the 20 haplotypes is below min_minor_allele_count = 2 are discarded, chromosome by chromosome.
- Frequency-dependent scaling exponent alpha = -0.25, with f the derived allele frequency of the mutation.
- Phenotype before standardisation: y = [12.26, 12.52, 12.42, 10.39, 9.6, 10.65, 11.57, 10.64, 12.51, 14.0]
- The phenotype is standardised to zero mean and y^T y = n, and the variant under test enters the score through its centred genotype column.
- Probes: num_probes = 200, taken as the rows of np.random.default_rng(101).standard_normal((200, 10))
- The chromosome under test is the first one, and the variant under test is the first of its retained variants.
- The solve starts from a zero solution, stops once the Euclidean norm of its residual falls to 1e-5 of the norm of the right-hand side, and performs at most 6 iterations, counting one iteration per update of the solution vector.

In the reasoning, report the number of retained variants on each chromosome, the relatedness divisor, the probe estimate of the trace, both variance components, the vector the solve returns and the residual trait. Your final answer must be a single number: the association score reported for the variant under test.

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

arg_carrier_genotypes

Goal
----
Sweep an ancestral recombination graph from left to right and return, for the

mutations it carries, the diploid genotype matrix together with the genomic

extent over which each carrying clade persists.

```python
def arg_carrier_genotypes(
    edges: list,
    mutations: list,
    haplotype_pairs: list,
    sequence_length: float = 100.0,
) -> tuple:
    """Build the diploid genotypes and clade extents of the mutations in an ARG.

    Args:
        edges: list of (child, parent, left, right) tuples. Each tuple states
            that node child inherits the half-open interval [left, right) from
            node parent.
        mutations: list of (child_node, position) tuples locating each mutation
            on the edge above child_node that spans that genomic position.
        haplotype_pairs: list of [hap_a, hap_b] pairs of leaf indices, one entry
            per diploid individual, in the order the individuals are reported.
        sequence_length: right end of the coordinate range, so positions lie in
            [0, sequence_length).

    Returns:
        tuple (genotypes, clade_end). genotypes holds the diploid allele counts
        with shape (n_individuals, n_mutations), columns ordered by increasing
        mutation position; mutations sharing a position keep their relative
        input order (a stable sort). clade_end has shape (n_mutations,) and is
        aligned with those columns, entry j giving the smallest position above
        the mutation at which the carrier set of column j is no longer the
        descendant set of its node, or sequence_length when no such position
        exists. Node identifiers are arbitrary non-negative integers and a leaf
        is a node without children in the marginal tree at the position.

    Raises:
        ValueError: if mutations is empty, if a mutation position lies outside
            [0, sequence_length), or if a haplotype pair does not hold exactly
            two leaf indices.
    """
    return result
```

### Step 2

standardize_arg_genotypes

Goal
----
Filter the mutations on minor allele count and return the centred,

frequency-scaled genotype matrix used to build the relatedness matrix.

```python
def standardize_arg_genotypes(
    genotypes: "np.ndarray",
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
) -> "np.ndarray":
    """Filter and frequency-scale a diploid genotype matrix.

    Args:
        genotypes: array of shape (n, p) holding diploid allele counts in
            {0, 1, 2}, one column per mutation.
        alpha: frequency-dependent scaling exponent of the method's genotype
            weight, with f the derived allele frequency across the 2 n
            haplotypes.
        min_minor_allele_count: mutations whose minor allele count across the
            2 n haplotypes is below this value are discarded.

    Returns:
        np.ndarray of shape (n, p_kept) holding the centred, frequency-scaled
        genotypes of the retained mutations, columns in their original order.

    Raises:
        ValueError: if genotypes is not two-dimensional, if any entry lies
            outside {0, 1, 2}, if min_minor_allele_count is below 1, or if no
            mutation survives the filter.
    """
    return x_std
```

### Step 3

build_arg_grm

Goal
----
Form the regional genetic relatedness matrix from the scaled genotype matrix and

return it together with its scaling factor.

```python
def build_arg_grm(x_std: "np.ndarray") -> tuple:
    """Build the regional relatedness matrix and its scaling factor.

    Args:
        x_std: array of shape (n, p) holding the centred, frequency-scaled
            genotypes of the retained mutations.

    Returns:
        tuple (grm, scale) where grm is an (n, n) float array and scale is the
        float divisor applied to X X^T.

    Raises:
        ValueError: if x_std is not two-dimensional or if it carries no
            variance, which leaves the scaling factor undefined.
    """
    return result
```

### Step 4

hutchinson_trace_squared

Goal
----
Estimate the trace of the squared relatedness matrix stochastically.

```python
def hutchinson_trace_squared(
    x_std: "np.ndarray",
    scale: float,
    num_probes: int = 200,
    seed: int = 101,
) -> float:
    """Estimate Tr R^2 stochastically from products with R.

    Args:
        x_std: array of shape (n, p) holding the scaled genotypes.
        scale: the divisor m in R = X X^T / m.
        num_probes: number of Gaussian probe vectors, taken as the rows of
            np.random.default_rng(seed).standard_normal((num_probes, n)).
        seed: seed passed to np.random.default_rng.

    Returns:
        float, the stochastic estimate of Tr R^2.

    Raises:
        ValueError: if num_probes is below 1 or if scale is not positive.
    """
    return trace_r_squared
```

### Step 5

rhe_variance_component

Goal
----
Standardise the phenotype and return the two variance components the moment

system assigns to the region and to the residual.

```python
def rhe_variance_component(
    grm: "np.ndarray",
    phenotype: "np.ndarray",
    trace_r_squared: float,
) -> tuple:
    """Fit the moment system for one region and phenotype.

    Args:
        grm: array of shape (n, n), the relatedness matrix.
        phenotype: array of shape (n,) holding the raw phenotype values, which
            are standardised here to zero mean and y^T y = n.
        trace_r_squared: the estimate of Tr R^2.

    Returns:
        tuple (sigma_g_squared, sigma_e_squared) of native floats, the genetic
        and residual variance components of the fitted model.

    Raises:
        ValueError: if phenotype length does not match the relatedness matrix,
            if the phenotype has zero variance, or if the two moment conditions
            are not independent, which leaves the estimate unidentified.
    """
    return result
```

### Step 6

loco_covariance_product

Goal
----
Return the action of the leave-one-chromosome-out phenotypic covariance on a set

of vectors, without forming that covariance.

```python
def loco_covariance_product(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    vectors: "np.ndarray",
) -> "np.ndarray":
    """Apply the leave-one-chromosome-out covariance to one or more vectors.

    Args:
        blocks: list of arrays, one per chromosome, each of shape (n, p_t)
            holding the centred and frequency-scaled genotypes of that
            chromosome's retained variants.
        focal_index: index into blocks of the chromosome under test, whose
            variants are excluded from the covariance.
        sigma_g_squared: the genetic variance component.
        sigma_e_squared: the residual variance component.
        vectors: array of shape (n,) or (n, k) to apply the covariance to.

    Returns:
        np.ndarray with the same shape as vectors, holding the product.

    Raises:
        ValueError: if blocks is empty, if focal_index is out of range, if the
            blocks disagree on the number of individuals, or if excluding the
            focal chromosome leaves no variants to build the covariance from.
    """
    return result
```

### Step 7

conjugate_gradient_solve

Goal
----
Solve the leave-one-chromosome-out system against the standardised phenotype and

return the solution together with the number of iterations spent on it.

```python
def conjugate_gradient_solve(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    phenotype: "np.ndarray",
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> tuple:
    """Solve the leave-one-chromosome-out system against the phenotype.

    Args:
        blocks: list of arrays, one per chromosome, each of shape (n, p_t)
            holding that chromosome's centred, frequency-scaled genotypes.
        focal_index: index into blocks of the chromosome under test, excluded
            from the covariance.
        sigma_g_squared: the genetic variance component.
        sigma_e_squared: the residual variance component.
        phenotype: array of shape (n,) of raw phenotype values, standardised
            here to zero mean and y^T y = n before it becomes the right-hand
            side of the system.
        tolerance: iterations stop once the Euclidean norm of the residual has
            fallen to this fraction of the norm of the right-hand side.
        max_iterations: the largest number of iterations that may be taken, which is
            the dimension of the subspace the returned vector is optimal over
            when the tolerance is not reached first.

    Returns:
        tuple (solution, iterations) where solution is an (n,) float array and
        iterations is the native int count of iterations actually taken.

    Raises:
        ValueError: if the phenotype length does not match the blocks, if the
            phenotype has zero variance, if tolerance is not positive, or if
            max_iterations is below 1.
    """
    return result
```

### Step 8

blup_residual_phenotype

Goal
----
Return the phenotype left over once the polygenic prediction from the retained

chromosomes has been subtracted.

```python
def blup_residual_phenotype(
    prediction: "np.ndarray",
    phenotype: "np.ndarray",
) -> "np.ndarray":
    """Subtract the polygenic prediction from the standardised phenotype.

    Args:
        prediction: array of shape (n,) holding the polygenic prediction made
            from the chromosomes that were retained in the fit.
        phenotype: array of shape (n,) of raw phenotype values, standardised
            here to zero mean and y^T y = n before the prediction is removed.

    Returns:
        np.ndarray of shape (n,) holding the residual phenotype.

    Raises:
        ValueError: if the prediction and the phenotype differ in length, or if
            the phenotype has zero variance.
    """
    return result
```

### Step 9

grammar_gamma_statistic

Goal
----
Return the score statistic of one variant against the residual phenotype.

```python
def grammar_gamma_statistic(
    genotype_column: "np.ndarray",
    residual: "np.ndarray",
) -> float:
    """Score one variant against a residual phenotype.

    Args:
        genotype_column: array of shape (n,) holding the centred,
            frequency-scaled genotypes of the variant under test.
        residual: array of shape (n,) holding the residual phenotype.

    Returns:
        float, the score statistic of the variant.

    Raises:
        ValueError: if the two arrays differ in length, or if the variant
            carries no variation, which leaves the statistic undefined.
    """
    return statistic
```

### Step 10

run_full_pipeline

Goal
----
Run the full ancestral recombination graph mixed model scan and return the score

statistic reported for the variant under test.

```python
def run_full_pipeline(
    chromosome_edges: list,
    chromosome_mutations: list,
    chromosome_spans: list,
    haplotype_pairs: list,
    phenotype: list,
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
    num_probes: int = 200,
    probe_seed: int = 101,
    focal_index: int = 0,
    focal_variant: int = 0,
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> float:
    """Run the whole scan and report the score of one variant.

    Args:
        chromosome_edges: list of per-chromosome edge lists, each entry a
            (child, parent, left, right) tuple.
        chromosome_mutations: list of per-chromosome mutation lists, each entry
            a (child node of the carrying edge, position) tuple.
        chromosome_spans: right end of the coordinate range of each chromosome.
        haplotype_pairs: list of [hap_a, hap_b] pairs of leaf indices, one per
            diploid individual, in the order the individuals are reported.
        phenotype: raw trait values, one per individual.
        alpha: frequency-dependent scaling exponent of the genotype weight.
        min_minor_allele_count: smallest retained minor allele count.
        num_probes: number of probe vectors.
        probe_seed: seed of the probe generator.
        focal_index: index of the chromosome under test.
        focal_variant: column of the focal chromosome's retained variants to
            score.
        tolerance: relative residual tolerance of the solve.
        max_iterations: step budget of the solve.

    Returns:
        float, the score statistic reported for the variant under test.

    Raises:
        ValueError: if the per-chromosome lists disagree in length, if
            focal_index does not name one of the chromosomes, or if
            focal_variant does not name one of its retained variants.
    """
    return statistic
```
