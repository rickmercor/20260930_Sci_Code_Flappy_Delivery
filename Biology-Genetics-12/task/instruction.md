# Biology-Genetics-12

## Background

Genotype imputation recovers unobserved variant states from linkage disequilibrium with observed loci. Reference-panel methods work well for represented populations, but ancestry mismatch and weak evidence for rare variants can limit their reliability.

Reference-free sequence models treat phased SNP states as discrete tokens and learn their context within each study cohort. Attention captures dependencies across a genomic segment, while physical-coordinate information distinguishes allele patterns arising in differently spaced regions and convolution supplies a local receptive field.

The resulting masked-language distribution assigns probabilities to homozygous and phase-specific heterozygous states. Such models are evaluated by concordance, dosage correlation, class-sensitive scores, or held-out likelihood under controlled missingness.

## Problem

Reference-panel genotype imputation can lose accuracy when the study ancestry is poorly represented, while a reference-free masked-genotype encoder can instead learn linkage disequilibrium from the study sequence itself. A recent phase-aware framework combines ordinal rotary position information, an actual-coordinate relative genomic positional bias, and a local convolutional bottleneck so that both dispersed and neighboring SNP dependencies contribute to masked-state predictions.

Use that framework for one frozen toy inference pass, with no fitting, to obtain probabilities for selected missing phased genotypes. Construct the framework's published directional genomic bias from each normalized coordinate vector, use it in the attention logits, and preserve the distinction between the two heterozygous phase states through scoring.

The configuration is:

- The phased allele tensor, with each innermost pair written in haplotype order, is `A = [[[0,0],[0,1],[0,0],[1,1],[1,0],[0,1],[1,1],[0,0],[1,0]],[[0,1],[0,1],[1,0],[1,1],[0,0],[1,0],[1,1],[0,1],[0,0]],[[1,1],[1,0],[1,0],[0,1],[0,0],[1,1],[0,1],[0,0],[1,1]],[[0,0],[0,0],[0,1],[1,0],[1,1],[0,1],[0,0],[1,1],[1,0]]]`. Encode `0|0`, `0|1`, `1|0`, and `1|1` as tokens 1, 2, 3, and 4; token 0 is MASK.
- The zero-based `(sample, variant)` mask list is `M = [[0,1],[0,6],[1,2],[1,7],[2,3],[2,8],[3,4],[3,6]]`, and the physical SNP positions are `p = [100,107,125,126,170,205,206,290,450]`.
- Set six-SNP windows with overlap 3 and starts 0, 3, and 6. Add CLS token 5 at coordinate `p_first - 1`, SEP token 6 at `p_last + 1`, and trailing PAD token 7 at coordinate zero, giving width 8. Within each window, normalize non-PAD coordinates affinely so CLS is zero and SEP is one; the PAD bias stays zero and PAD keys are excluded from attention.
- The hidden dimension is `d = 4`, all indexing in the formulas below is zero based, all matrix projections multiply on the right, and all calculations use float64. The frozen embedding is `E[t,r] = 0.45*sin((t+1)*(r+1)) + 0.15*cos((t+2)*(r+1))` for tokens 0 through 6, with `E[7,:] = 0`; layer-normalize every token over its hidden coordinates using population variance and `layernorm_eps = 1e-5`.
- The frozen projections are `Wq[r,c] = 0.35*sin((r+1)*(c+1)+0.2)`, `Wk[r,c] = 0.30*cos((r+1)*(c+2)-0.1)`, `Wv[r,c] = 0.40*sin((r+2)*(c+1)+0.3)`, and `Wo[r,c] = 0.25*cos((r+1)*(c+1)+0.4)`. Apply RoPE to Q and K only, pairing consecutive hidden coordinates with angle `position*rope_base^(-2*m/d)` for pair index `m`, using `rope_base = 10000` and the rotation `(x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)`.
- Set one attention head, `beta = -1.4`, scaled dot-product logits, row-wise softmax, and the frozen output projection `Wo`. The encoder has one block and follows the framework's published residual scaling and post-residual layer-normalization convention after attention and after the convolutional bottleneck.
- The convolutional bottleneck has expansion factor 2, kernel size 3, same zero-padded cross-correlation, pairwise max pooling with stride 2, nearest-neighbor upsampling by repetition, and zero dropout. Its frozen filters are `W1[o,i,k] = 0.12*sin((o+1)+(i+1)*(k+1))` for 8 output channels and `W2[o,i,k] = 0.10*cos((o+1)*(i+1)+(k+1))` for 4 output channels.
- Decode the four genotype classes with `Wd[r,c] = 0.55*sin((r+1)*(c+2)+0.25*c)` and bias `[0.15,-0.10,0.05,0.0]`, followed by softmax. For a masked site present in multiple windows, average its class probabilities before evaluation, then take the mean negative natural-log probability of its true phased state across the eight rows of `M`.

Carry out the phase-aware masked-genotype computation with the actual-coordinate bias and local bottleneck defined above. Your final answer must be a single number: the full-precision mean masked negative log-likelihood.

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

01_encode_phased_genotypes

Goal
----
Phased diploid SNP calls retain the order of the two haplotypes. GenoBERT assigns

the four allele pairs ``0|0``, ``0|1``, ``1|0``, and ``1|1`` to states 1, 2, 3,

and 4, respectively, while state 0 denotes a masked genotype. This step stores

the masked model input beside the unmasked target state so phase errors remain

distinguishable during evaluation.

Inputs

------

alleles: binary array of shape (n_samples, n_variants, 2)

masked_sites: integer array of shape (n_masked, 2) containing sample and variant indices

Returns

-------

encoded: integer array of shape (n_samples, n_variants, 2), with observed and target states

```python
def encode_phased_genotypes(
    alleles: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Encode phased allele pairs and replace selected model inputs by MASK.

    Parameters
    ----------
    alleles : np.ndarray
        Binary array with shape ``(n_samples, n_variants, 2)``.
    masked_sites : np.ndarray
        Integer array with shape ``(n_masked, 2)`` whose rows are distinct
        ``(sample_index, variant_index)`` pairs.

    Raises
    ------
    ValueError
        If alleles does not have shape ``(n_samples, n_variants, 2)``, contains
        values other than zero and one, or has no samples or variants; or if
        masked_sites does not have shape ``(n_masked, 2)``, is non-integral,
        contains duplicate rows, or contains an out-of-range index.

    Returns
    -------
    encoded : np.ndarray
        Integer array with observed states in channel 0 and unmasked target
        states in channel 1.
    """
    return encoded  # noqa: F821
```

### Step 2

02_segment_genotype_windows

Goal
----
Genome-wide genotype matrices are wide relative to cohort size, so GenoBERT

recasts each sample as a document and overlapping fixed-SNP windows as segments.

Each segment receives CLS and SEP boundary tokens, and a short terminal segment

is padded to the same width. The packed representation here retains token state,

target state, physical coordinate, and original variant index for later merging.

Inputs

------

encoded: integer array of shape (n_samples, n_variants, 2)

genomic_positions: increasing physical positions of shape (n_variants,)

Returns

-------

segments: float array of shape (n_samples, n_segments, window_size + 2, 4)

```python
def segment_genotype_windows(
    encoded: "np.ndarray",
    genomic_positions: "np.ndarray",
    window_size: int = 6,
    overlap: int = 3,
) -> "np.ndarray":
    """Form overlapping genotype segments with CLS, SEP, and PAD tokens.

    Parameters
    ----------
    encoded : np.ndarray
        Integer array of shape ``(n_samples, n_variants, 2)`` containing
        observed and target genotype states.
    genomic_positions : np.ndarray
        Strictly increasing, finite, positive physical coordinates with one
        entry per variant.
    window_size : int
        Positive number of SNP tokens per segment.
    overlap : int
        Number of SNP tokens shared by consecutive segments, in the interval
        ``[0, window_size)``.

    Raises
    ------
    ValueError
        If encoded has the wrong shape or invalid states, genomic_positions is
        not a matching strictly increasing positive vector, window_size is not
        a positive integer, or overlap is not an integer in
        ``[0, window_size)``.

    Returns
    -------
    segments : np.ndarray
        Packed float64 array whose last axis is observed token, target token,
        physical coordinate, and zero-based source variant index.
    """
    return segments  # noqa: F821
```

### Step 3

03_build_genomic_bias

Goal
----
Relative genomic positional bias uses actual SNP coordinates rather than only

ordinal token offsets. Within each segment, the virtual CLS and SEP coordinates

bound an affine normalization to zero and one; PAD positions stay zero and are

marked invalid. The resulting bias vector preserves irregular physical spacing

while remaining comparable across genomic windows.

Inputs

------

segments: packed segment array of shape (n_samples, n_segments, width, 4)

Returns

-------

bias_pack: array of shape (n_samples, n_segments, width, 2) containing bias and validity

```python
def build_genomic_bias(segments: "np.ndarray") -> "np.ndarray":
    """Normalize genomic coordinates and mark non-padding tokens.

    Parameters
    ----------
    segments : np.ndarray
        Packed segment array whose last axis contains observed token, target
        token, physical coordinate, and source variant index.

    Raises
    ------
    ValueError
        If segments is not a finite four-dimensional array with last dimension
        four, if token states are non-integral or outside 0 through 7, if a
        segment lacks one CLS and one SEP token in that order, if padding is not
        trailing, or if non-padding coordinates are not strictly increasing.

    Returns
    -------
    bias_pack : np.ndarray
        float64 array with normalized coordinate in channel 0 and a binary
        non-padding indicator in channel 1.
    """
    return bias_pack  # noqa: F821
```

### Step 4

04_project_rotary_qkv

Goal
----
Position-free content attention cannot distinguish token order. This step

reproduces the framework's late-combination rotary projection using the frozen

token and QKV maps in the main prompt. The source-specific channel pairing and

phase orientation matter at nonzero positions: the correct transform leaves the

position-zero query and key unchanged, preserves their per-token Euclidean norms,

and does not rotate values.

Inputs

------

segments: packed segment array containing observed token states

embedding_dim: positive even hidden dimension

Returns

-------

projected: array ending in axes (embedding, rotary query, rotary key, value) and hidden coordinate

```python
def project_rotary_qkv(
    segments: "np.ndarray", embedding_dim: int = 4, rope_base: float = 10000.0
) -> "np.ndarray":
    """Form deterministic Q, K, and V arrays with the source's rotary rule.

    Parameters
    ----------
    segments : np.ndarray
        Finite packed segment array of shape
        ``(n_samples, n_segments, width, 4)`` with integral token states from
        zero through seven in its first channel.
    embedding_dim : int
        Positive even hidden dimension.
    rope_base : float
        Finite base greater than one for the source-defined rotary frequencies.

    Raises
    ------
    ValueError
        If segments has the wrong shape or invalid token states, embedding_dim
        is not a positive even integer, or rope_base is not finite and greater
        than one.

    Returns
    -------
    projected : np.ndarray
        float64 array of shape
        ``(n_samples, n_segments, width, 4, embedding_dim)`` ordered as the
        normalized embedding, rotary query, rotary key, and value.
    """
    return projected  # noqa: F821
```

### Step 5

05_apply_genomic_attention

Goal
----
Actual SNP coordinates enter this head as an ordered query-key displacement,

not as unsigned distance. Under the declared row-query/column-key convention,

the source-defined bias has zero diagonal and changes sign when query and key

are exchanged. Its direction and its point of composition with content attention

are method-defining choices rather than interchangeable implementation details.

Padding keys are excluded and the result passes through the frozen output

projection from the main prompt.

Inputs

------

projected: normalized embeddings and rotary Q, K, V arrays

bias_pack: normalized genomic positions and validity mask

Returns

-------

attention_output: float array with one hidden vector per token

```python
def apply_genomic_attention(
    projected: "np.ndarray", bias_pack: "np.ndarray", beta: float = -1.4
) -> "np.ndarray":
    """Apply the source-defined coordinate-aware attention and frozen output map.

    Parameters
    ----------
    projected : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, 4, d)`` ordered
        as embedding, rotary query, rotary key, and value.
    bias_pack : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, 2)`` containing
        normalized positions and binary non-padding indicators.
    beta : float
        Finite head coefficient associated with the directional coordinate prior.

    Raises
    ------
    ValueError
        If projected or bias_pack has the wrong shape, contains non-finite
        values, has a non-binary validity channel, leaves a segment with no
        valid key, or if beta is not finite.

    Returns
    -------
    attention_output : np.ndarray
        float64 array of shape ``(n_samples, n_segments, width, d)``.
    """
    return attention_output  # noqa: F821
```

### Step 6

06_normalize_attention_residual

Goal
----
DeepNet residual scaling stabilizes the GenoBERT encoder by multiplying the

incoming representation by alpha = (2N)^(1/4), where N is encoder depth, before

adding a sublayer output. Post-residual layer normalization is performed across

the hidden coordinates of each token with population variance and a fixed

epsilon. This step applies that operation to the attention branch.

Inputs

------

projected: array containing the normalized token embedding in slot zero

attention_output: output of the relative-bias attention sublayer

Returns

-------

hidden: normalized attention-residual representation

```python
def normalize_attention_residual(
    projected: "np.ndarray", attention_output: "np.ndarray", encoder_depth: int = 1
) -> "np.ndarray":
    """Apply DeepNet scaling, an attention residual, and token-wise layer normalization.

    Parameters
    ----------
    projected : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, 4, d)`` whose
        slot zero is the input embedding.
    attention_output : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, d)``.
    encoder_depth : int
        Positive encoder depth used in the residual scaling factor.

    Raises
    ------
    ValueError
        If projected or attention_output has the wrong shape or non-finite
        entries, or if encoder_depth is not a positive integer.

    Returns
    -------
    hidden : np.ndarray
        float64 array of shape ``(n_samples, n_segments, width, d)``.
    """
    return hidden  # noqa: F821
```

### Step 7

07_apply_cnn_bottleneck

Goal
----
This local branch changes both channel width and sequence resolution before it

returns to the encoder shape. Reconstruct the source's two-convolution topology

from the main prompt and paper: activation placement, the order of pooling versus

the second convolution, and restoration before the residual are not commutative.

The declared same-padding rule also governs short boundary cases. The completed

branch uses the framework's DeepNet residual normalization.

Inputs

------

hidden: attention-residual representation of shape (n_samples, n_segments, width, d)

bottleneck_factor: positive channel expansion factor

Returns

-------

encoded_hidden: locally aggregated and normalized representation

```python
def apply_cnn_bottleneck(
    hidden: "np.ndarray",
    encoder_depth: int = 1,
    bottleneck_factor: float = 2.0,
    kernel_size: int = 3,
) -> "np.ndarray":
    """Apply the source-defined convolutional bottleneck and residual branch.

    Parameters
    ----------
    hidden : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, d)`` with an
        even token width.
    encoder_depth : int
        Positive encoder depth used in residual scaling.
    bottleneck_factor : float
        Finite positive channel multiplier for which
        ``int(d * bottleneck_factor)`` is at least one.
    kernel_size : int
        Positive odd convolution kernel size.

    Raises
    ------
    ValueError
        If hidden has the wrong shape, non-finite entries, or odd width; if
        encoder_depth is not a positive integer; if bottleneck_factor is not
        finite and positive or gives zero channels; or if kernel_size is not a
        positive odd integer.

    Returns
    -------
    encoded_hidden : np.ndarray
        float64 array with the same shape as hidden.
    """
    return encoded_hidden  # noqa: F821
```

### Step 8

08_merge_masked_probabilities

Goal
----
The masked-language head maps each encoded token back to probabilities for the

four phased genotype states. Overlapping genomic windows can provide multiple

predictions for one masked locus, so this toy inference rule averages class

probabilities across every segment occurrence before scoring. The target phase

state is carried with each occurrence and must agree across windows.

Inputs

------

hidden: final hidden vectors for all segment tokens

segments: packed segment data containing source indices and target states

masked_sites: requested sample and source-variant pairs

Returns

-------

pooled: rows containing site identity, target state, and four averaged probabilities

```python
def merge_masked_probabilities(
    hidden: "np.ndarray", segments: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Decode genotype probabilities and average overlapping masked predictions.

    Parameters
    ----------
    hidden : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, d)``.
    segments : np.ndarray
        Finite packed array of shape ``(n_samples, n_segments, width, 4)`` with
        observed token, target token, coordinate, and source index.
    masked_sites : np.ndarray
        Distinct integral ``(sample_index, variant_index)`` rows.

    Raises
    ------
    ValueError
        If hidden and segments have incompatible shapes or non-finite entries;
        if masked_sites has the wrong shape, non-integral or duplicate rows, or
        an out-of-range sample; or if a requested site has no masked segment
        occurrence or has inconsistent target states outside 1 through 4.

    Returns
    -------
    pooled : np.ndarray
        float64 array of shape ``(n_masked, 7)`` with columns sample index,
        variant index, target state, then probabilities for states 1 through 4.
    """
    return pooled  # noqa: F821
```

### Step 9

09_score_masked_likelihood

Goal
----
Masked-token training learns a categorical distribution over the four phased

genotype states at corrupted loci. Phase-aware negative log-likelihood averages

-log p(y) over the masked sites, so confusing ``0|1`` with ``1|0`` is penalized

even though both states have alternate-allele dosage one. This is the scalar loss

used to judge the frozen toy imputation pass.

Inputs

------

pooled: rows containing sample index, variant index, target state, and four probabilities

Returns

-------

masked_nll: native float mean negative log-likelihood

```python
def score_masked_likelihood(pooled: "np.ndarray") -> float:
    """Compute mean phase-aware negative log-likelihood at masked sites.

    Parameters
    ----------
    pooled : np.ndarray
        Array of shape ``(n_masked, 7)`` with nonnegative integral sample and
        variant indices, target state 1 through 4, and four class probabilities.

    Raises
    ------
    ValueError
        If pooled is not a nonempty finite array of shape ``(n_masked, 7)``, if
        index or target columns violate their integral ranges, if probabilities
        are outside ``[0, 1]`` or do not sum to one within ``1e-10``, or if a
        target-class probability is zero.

    Returns
    -------
    masked_nll : float
        Mean negative natural-log probability of the true phased state.
    """
    return masked_nll  # noqa: F821
```

### Step 10

10_run_full_pipeline

Goal
----
The deterministic imputation pipeline encodes phased genotypes, constructs

overlapping token windows, derives normalized physical-coordinate bias, combines

rotary attention with the directional genomic bias, and applies the local CNN

bottleneck. A frozen masked-language head decodes every masked occurrence,

overlap averaging consolidates repeated loci, and mean phase-aware negative

log-likelihood supplies the final scalar.

Inputs

------

beta: relative genomic positional bias coefficient

overlap: number of shared SNP tokens between consecutive six-SNP windows

Returns

-------

masked_nll: native float reported inside final_answer tags

```python
def run_full_pipeline(beta: float = -1.4, overlap: int = 3) -> float:
    """Run the frozen GenoBERT-style toy imputation pipeline.

    Parameters
    ----------
    beta : float
        Finite coefficient multiplying relative genomic positional bias.
    overlap : int
        Integer overlap in ``[0, 6)`` for six-SNP windows.

    Raises
    ------
    ValueError
        If beta is not finite or overlap is not an integer in ``[0, 6)``.

    Returns
    -------
    masked_nll : float
        Mean phase-aware negative log-likelihood over the fixed masked sites.
    """
    return masked_nll  # noqa: F821
```
