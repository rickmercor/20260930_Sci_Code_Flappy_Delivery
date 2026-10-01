# Biology-Genetics-34

## Background

Genome-wide association studies have linked thousands of genomic regions to complex traits and diseases, but an association signal names a neighbourhood rather than a variant. Within a region, dozens of single nucleotide polymorphisms are inherited together through linkage disequilibrium and carry almost indistinguishable statistical evidence, so resolving which of them actually causes the effect — fine-mapping — is a separate inferential problem from discovery. It matters because only the causal variant supports a mechanistic follow-up: the gene it acts through, the cell type it acts in, and whether the allele is a plausible therapeutic target.

Statistical fine-mapping is usually posed as Bayesian variable selection. A prior is placed on the effect of each variant, typically a point mass at zero mixed with one or more normal components, and the fitted model returns for every variant a posterior inclusion probability, the probability that it carries a non-zero effect given the data. Conventional methods run this one locus at a time, on windows drawn around genome-wide significant lead variants, and are therefore forced to set the prior probability of association by convention rather than estimate it; commonly as the reciprocal of the number of variants in the window. That is a strong assumption made in the one place the data could speak, and it confines the analysis to the small fraction of the heritability that has already cleared the discovery threshold.

Fitting every variant in the genome jointly changes the character of the inference. The mixing probabilities and the spread of causal effect sizes are then estimated from the data rather than assumed, information is borrowed across the genome, and the same prior is applied inside and outside significant loci, so signal can be resolved in regions a discovery scan would discard. The consequence that matters here is that a posterior inclusion probability is no longer a purely local quantity: what a given threshold on it demands of the underlying data depends on the polygenicity and effect-size distribution the model has estimated for that trait, and therefore differs from trait to trait.

That opens a question the field has long been able to answer for discovery but not for fine-mapping. Association studies have well-established power calculations that say how many samples are needed to detect an effect of a given size, and those calculations are routinely used to design the next study. No comparable analysis existed for fine-mapping, even though the practical question, how large a study must become before the variants it resolves account for a worthwhile share of the trait's heritability is exactly what determines whether a further round of data collection is worth funding. For diseases the question is harder still, because samples are ascertained on affection status while the genetic architecture is defined on an underlying continuous liability, so a headcount of cases and controls is not directly comparable to a sample size for a quantitative trait.

## Problem

Genome-wide Bayesian mixture fine-mapping assigns every variant a posterior inclusion probability (PIP) and calls a variant fine-mapped once its PIP clears a threshold, so the amount of single-variant evidence that threshold actually demands is fixed by the genome-wide mixture prior rather than by any conventional significance level. Because that prior carries the polygenicity and the spread of causal effect sizes, the same PIP threshold behaves very differently in a sparse and in a highly polygenic trait, and the size a prospective study must reach before its fine-mapped variants carry a target share of the SNP-based heritability follows from the estimated architecture and not from an association power calculation.

I have run such an analysis on a psychiatric disorder and want to know how large the next study has to be. The model fits $12{,}500{,}000$ SNPs under a five-component prior in which a variant is either null or draws its effect from one of four normal components whose variances are $10^{-5}$, $10^{-4}$, $10^{-3}$ and $10^{-2}$ times the SNP-based heritability; the posterior architecture places $40{,}000$ causal variants in the first of those four components, $3{,}000$ in the second, $200$ in the third and $10$ in the fourth, with every remaining SNP null. The SNP-based heritability is $0.24$ on the liability scale, where the phenotype has unit variance, and the prediction is to be made for a variant standing on its own, with linkage disequilibrium ignored. What is set out above is the complete specification of the model: obtain the posterior inclusion probability by deriving it from this prior together with the single-variant likelihood rather than by adopting a published parameterisation of it, and where the two disagree your own derivation governs.

Take a prospective study that calls a variant fine-mapped at $\mathrm{PIP} > 0.9$ and find the size at which the variants passing that threshold are expected to carry $75\%$ of the SNP-based heritability. The disorder has a lifetime prevalence of $1\%$ and the study ascertains equal numbers of cases and controls. Report the number of cases that study needs. Give one short line for each of the following as well, since they are what the answer is checked against: the sampling distribution you assign to the single-variant test statistic; the constants your PIP expression attaches to each non-null mixture component; the value of the test statistic at which the PIP threshold is met, and how that compares with genome-wide significance; the probability that a causal variant drawn from each of the four non-null components clears the threshold; the share of the SNP-based heritability contributed by each of those four components at the reported study size; the equivalent quantitative-trait sample size; the expected number of causal variants passing the threshold; the same case count recomputed at PIP thresholds of $0.5$ and of $0.95$, together with what the trend across the three implies; and the limit the explained share approaches as the sample size grows without bound.

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

01_mixture_probabilities

Goal
----
Turn the per-component counts of causal variants reported by a genome-wide Bayesian mixture model into the mixing probabilities of the prior, including the probability that a variant is null.

```python
import numpy as np

def mixture_probabilities(n_snps: int, causal_counts: tuple) -> np.ndarray:
    """Convert per-component causal-variant counts into prior mixing probabilities.

    Parameters
    ----------
    n_snps : int
        Total number of SNPs fitted by the model (n_snps > 0).
    causal_counts : tuple
        One non-negative integer per non-null mixture component, giving the
        number of causal variants assigned to that component. The total must
        be smaller than ``n_snps``.

    Returns
    -------
    probabilities : np.ndarray
        Shape ``(1 + len(causal_counts),)`` float array. Element 0 is the
        prior probability that a SNP is null; the remaining elements are the
        prior probabilities of the non-null components, in the given order.

    Raises
    ------
    ValueError
        If ``n_snps`` is not an integer greater than zero; if
        ``causal_counts`` is not a non-empty sequence of integers; if any
        count is negative; or if the counts sum to ``n_snps`` or more, which
        would leave no null probability.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(1 + len(causal_counts), dtype=float)  # placeholder
```

### Step 2

02_mixture_scale_factors

Goal
----
Convert the prior variance scale factors of the mixture model into the per-component shrinkage and precision constants that a single-variant posterior at a given sample size depends on.

```python
import numpy as np

def mixture_scale_factors(n: float, h2_snp: float, gamma: tuple) -> np.ndarray:
    """Build the shrinkage and precision constants of the non-null components.

    Parameters
    ----------
    n : float
        Sample size of the study, on the scale at which the phenotype has
        unit variance (n > 0).
    h2_snp : float
        SNP-based heritability of the trait (0 < h2_snp < 1).
    gamma : tuple
        Prior variance scale factors of the mixture, one per component and in
        the same order as the mixing probabilities. The first entry is the
        null component and must be exactly zero; the rest must be strictly
        positive.

    Returns
    -------
    factors : np.ndarray
        Shape ``(len(gamma) - 1, 2)`` float array with one row per non-null
        component, holding the shrinkage ratio in column 0 and the posterior
        precision in column 1.

    Raises
    ------
    ValueError
        If ``n`` is not a finite real number greater than zero; if
        ``h2_snp`` is not a real number strictly between 0 and 1; if
        ``gamma`` is not a finite sequence holding a null component and at
        least one non-null one; if the first ``gamma`` entry is not exactly
        zero; or if any later ``gamma`` entry is not strictly positive.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((len(gamma) - 1, 2), dtype=float)  # placeholder
```

### Step 3

03_pip_curve_constants

Goal
----
Reduce the prior mixing probabilities and the per-component shrinkage constants to the pair of numbers each non-null component contributes to the posterior inclusion probability of a single variant.

```python
import numpy as np

def pip_curve_constants(mixture_probs: np.ndarray, scale_factors: np.ndarray,
                        n: float) -> np.ndarray:
    """Build the amplitude and rate constants of each non-null component.

    Parameters
    ----------
    mixture_probs : np.ndarray
        Shape ``(1 + n_components,)`` prior mixing probabilities, with the
        null probability first.
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    n : float
        Sample size of the study, on the scale at which the phenotype has
        unit variance (n > 0).

    Returns
    -------
    constants : np.ndarray
        Shape ``(n_components, 2)`` float array with the amplitude in
        column 0 and the rate in column 1, one row per non-null component.

    Raises
    ------
    ValueError
        If ``n`` is not a finite real number greater than zero; if
        ``scale_factors`` does not have shape ``(n_components, 2)`` or holds
        a value that is not finite and positive; if ``mixture_probs`` does
        not hold exactly one more entry than ``scale_factors`` has rows; if
        any probability is negative or not finite; if the null probability
        is not strictly positive; or if the probabilities do not sum to 1
        within a tolerance of 1e-9.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((np.asarray(scale_factors).shape[0], 2), dtype=float)  # placeholder
```

### Step 4

04_pip_threshold_statistic

Goal
----
Invert the posterior inclusion probability of a single variant to find the chi-square statistic at which a given PIP threshold is first reached.

```python
import numpy as np

def pip_threshold_statistic(curve_constants: np.ndarray, alpha: float) -> float:
    """Find the chi-square statistic at which the PIP threshold is reached.

    Parameters
    ----------
    curve_constants : np.ndarray
        Shape ``(n_components, 2)`` array with the amplitude in column 0 and
        the rate in column 1, one row per non-null mixture component.
    alpha : float
        Posterior inclusion probability threshold (0 < alpha < 1).

    Returns
    -------
    z_threshold : float
        Smallest non-negative chi-square statistic whose posterior inclusion
        probability reaches ``alpha``, as a native Python float.

    Raises
    ------
    ValueError
        If ``curve_constants`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if ``alpha`` is not a
        real number strictly between 0 and 1; or if the threshold is not
        reached by a chi-square statistic of 1e9, which no admissible set of
        constants should require.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```

### Step 5

05_component_detection_power

Goal
----
Compute, for each non-null mixture component, the probability that a causal variant drawn from that component produces a chi-square statistic above the threshold set by the PIP cut-off.

```python
import numpy as np

def component_detection_power(scale_factors: np.ndarray, z_threshold: float) -> np.ndarray:
    """Probability that a causal variant of each component clears the threshold.

    Parameters
    ----------
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    z_threshold : float
        Chi-square statistic at which the PIP threshold is reached
        (z_threshold >= 0).

    Returns
    -------
    power : np.ndarray
        Shape ``(n_components,)`` float array of detection probabilities, one
        per non-null mixture component.

    Raises
    ------
    ValueError
        If ``scale_factors`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if any posterior
        precision does not strictly exceed its shrinkage ratio, which no
        positive sample size can produce; or if ``z_threshold`` is not a
        finite real number greater than or equal to zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(scale_factors).shape[0], dtype=float)  # placeholder
```

### Step 6

06_component_tail_second_moment

Goal
----
Compute, for each non-null mixture component, the second moment of the standardised statistic restricted to the region in which the variant is detected.

```python
import numpy as np

def component_tail_second_moment(scale_factors: np.ndarray, z_threshold: float,
                                 detection_power: np.ndarray) -> np.ndarray:
    """Second moment of the standardised statistic over the detection region.

    Parameters
    ----------
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    z_threshold : float
        Chi-square statistic at which the PIP threshold is reached
        (z_threshold >= 0).
    detection_power : np.ndarray
        Shape ``(n_components,)`` array of detection probabilities, one per
        non-null mixture component.

    Returns
    -------
    second_moment : np.ndarray
        Shape ``(n_components,)`` float array of restricted second moments,
        one per non-null mixture component.

    Raises
    ------
    ValueError
        If ``scale_factors`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if any posterior
        precision does not strictly exceed its shrinkage ratio; if
        ``detection_power`` does not hold exactly one entry per component;
        if any detection probability is not finite or falls outside the
        interval [0, 1]; or if ``z_threshold`` is not a finite real number
        greater than or equal to zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(scale_factors).shape[0], dtype=float)  # placeholder
```

### Step 7

07_component_detected_variance

Goal
----
Compute, for each non-null mixture component, the expected squared effect of a causal variant restricted to the outcomes in which that variant is detected.

```python
import numpy as np

def component_detected_variance(scale_factors: np.ndarray, detection_power: np.ndarray,
                                tail_second_moment: np.ndarray, h2_snp: float,
                                n: float) -> np.ndarray:
    """Expected squared effect of a causal variant over the detection region.

    Parameters
    ----------
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    detection_power : np.ndarray
        Shape ``(n_components,)`` array of detection probabilities.
    tail_second_moment : np.ndarray
        Shape ``(n_components,)`` array of restricted second moments of the
        standardised statistic.
    h2_snp : float
        SNP-based heritability of the trait (0 < h2_snp < 1).
    n : float
        Sample size of the study, on the scale at which the phenotype has
        unit variance (n > 0).

    Returns
    -------
    detected_variance : np.ndarray
        Shape ``(n_components,)`` float array of expected squared effects
        restricted to the detection region, one per non-null component.

    Raises
    ------
    ValueError
        If ``scale_factors`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if
        ``detection_power`` or ``tail_second_moment`` does not hold exactly
        one entry per component; if any detection probability is not finite
        or falls outside the interval [0, 1]; if any restricted second
        moment is not finite or is negative; if ``h2_snp`` is not a real
        number strictly between 0 and 1; if ``n`` is not a finite real
        number greater than zero; or if ``scale_factors`` and ``n`` describe
        different sample sizes, that is if any posterior precision minus its
        shrinkage ratio differs from ``n``.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(scale_factors).shape[0], dtype=float)  # placeholder
```

### Step 8

08_expected_heritability_explained

Goal
----
Combine the per-component detected effect variances with the prior mixing probabilities into the expected proportion of SNP-based heritability carried by the variants that pass the PIP threshold.

```python
import numpy as np

def expected_heritability_explained(mixture_probs: np.ndarray, detected_variance: np.ndarray,
                                    h2_snp: float, n_snps: int) -> float:
    """Expected proportion of SNP-based heritability carried by detected variants.

    Parameters
    ----------
    mixture_probs : np.ndarray
        Shape ``(1 + n_components,)`` prior mixing probabilities, with the
        null probability first.
    detected_variance : np.ndarray
        Shape ``(n_components,)`` array of expected squared causal effects
        restricted to the detection region.
    h2_snp : float
        SNP-based heritability of the trait (0 < h2_snp < 1).
    n_snps : int
        Total number of SNPs fitted by the model (n_snps > 0).

    Returns
    -------
    proportion : float
        Expected proportion of the SNP-based heritability carried by the
        variants that pass the threshold, as a native Python float.

    Raises
    ------
    ValueError
        If ``detected_variance`` is empty, holds a value that is not finite,
        or holds a negative value; if ``mixture_probs`` does not hold
        exactly one more entry than ``detected_variance``; if any
        probability is negative or not finite; if the probabilities do not
        sum to 1 within a tolerance of 1e-9; if ``h2_snp`` is not a real
        number strictly between 0 and 1; or if ``n_snps`` is not an integer
        greater than zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```

### Step 9

09_required_case_count

Goal
----
Convert an equivalent quantitative-trait sample size on the liability scale into the number of cases an ascertained case-control study of a disease must collect.

```python
import numpy as np

def required_case_count(n_equivalent: float, prevalence: float,
                        case_fraction: float) -> float:
    """Cases needed to match a given equivalent quantitative-trait sample size.

    Parameters
    ----------
    n_equivalent : float
        Equivalent quantitative-trait sample size on the liability scale
        (n_equivalent > 0).
    prevalence : float
        Lifetime prevalence of the disease in the population
        (0 < prevalence < 1).
    case_fraction : float
        Fraction of the ascertained sample that are cases
        (0 < case_fraction < 1).

    Returns
    -------
    n_cases : float
        Number of cases the study must collect, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is not a finite real number; if ``n_equivalent`` is
        not greater than zero; if ``prevalence`` is not strictly between 0
        and 1; or if ``case_fraction`` is not strictly between 0 and 1, a
        study of all cases or all controls being uninformative.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```

### Step 10

10_run_finemapping_power_pipeline

Goal
----
Chain the sub-problem functions 01-09 end to end and return the number of cases a prospective case-control study needs before its fine-mapped variants carry a target share of the SNP-based heritability.

```python
import numpy as np

def run_finemapping_power_pipeline(n_snps: int, causal_counts: tuple, gamma: tuple,
                                   h2_snp: float, pip_threshold: float,
                                   target_h2_fraction: float, prevalence: float,
                                   case_fraction: float) -> float:
    """Run the full prospective fine-mapping power pipeline for one trait.

    Parameters
    ----------
    n_snps : int
        Total number of SNPs fitted by the model (n_snps > 0).
    causal_counts : tuple
        One non-negative integer per non-null mixture component, giving the
        number of causal variants assigned to that component.
    gamma : tuple
        Prior variance scale factors of the mixture, one per component. The
        first entry is the null component and must be exactly zero.
    h2_snp : float
        SNP-based heritability of the trait on the liability scale
        (0 < h2_snp < 1).
    pip_threshold : float
        Posterior inclusion probability above which a variant counts as
        fine-mapped (0 < pip_threshold < 1).
    target_h2_fraction : float
        Share of the SNP-based heritability the fine-mapped variants are
        required to carry (0 < target_h2_fraction < 1).
    prevalence : float
        Lifetime prevalence of the disease in the population
        (0 < prevalence < 1).
    case_fraction : float
        Fraction of the ascertained sample that are cases
        (0 < case_fraction < 1).

    Returns
    -------
    n_cases : float
        Number of cases the prospective study must collect, as a native
        Python float.

    Raises
    ------
    ValueError
        If ``n_snps`` is not an integer; if ``causal_counts`` is not a
        non-empty sequence; if ``gamma`` does not hold exactly one more
        entry than ``causal_counts``; if any of ``h2_snp``,
        ``pip_threshold``, ``target_h2_fraction``, ``prevalence`` or
        ``case_fraction`` is not a finite real number strictly between 0 and
        1; or if the target share of heritability is out of reach, meaning
        no finite study size attains it. Errors raised by the sub-problem
        functions this step calls propagate unchanged.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-09 (``mixture_probabilities``, ``mixture_scale_factors``,
    ``pip_curve_constants``, ``pip_threshold_statistic``,
    ``component_detection_power``, ``component_tail_second_moment``,
    ``component_detected_variance``, ``expected_heritability_explained``,
    ``required_case_count``) and feed each returned value into the next,
    rather than reimplementing them. Include every import your implementation
    needs (for example ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```
