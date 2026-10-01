#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def encode_phased_genotypes(
    alleles: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Reference implementation."""
    alleles = np.asarray(alleles)
    masked_sites = np.asarray(masked_sites)
    if alleles.ndim != 3 or alleles.shape[2] != 2:
        raise ValueError("alleles must have shape (n_samples, n_variants, 2)")
    if alleles.shape[0] == 0 or alleles.shape[1] == 0:
        raise ValueError("alleles must contain at least one sample and variant")
    if not np.all((alleles == 0) | (alleles == 1)):
        raise ValueError("alleles must be binary")
    if masked_sites.ndim != 2 or masked_sites.shape[1] != 2:
        raise ValueError("masked_sites must have shape (n_masked, 2)")
    if not np.issubdtype(masked_sites.dtype, np.integer) and (
        not np.all(np.isfinite(masked_sites))
        or not np.all(masked_sites == np.floor(masked_sites))
    ):
        raise ValueError("masked_sites must contain integers")
    masked_sites = masked_sites.astype(np.int64, copy=False)
    if len({tuple(row) for row in masked_sites.tolist()}) != len(masked_sites):
        raise ValueError("masked_sites rows must be distinct")
    if masked_sites.size:
        if np.any(masked_sites[:, 0] < 0) or np.any(
            masked_sites[:, 0] >= alleles.shape[0]
        ):
            raise ValueError("masked sample index out of range")
        if np.any(masked_sites[:, 1] < 0) or np.any(
            masked_sites[:, 1] >= alleles.shape[1]
        ):
            raise ValueError("masked variant index out of range")

    target = (
        1 + 2 * alleles[:, :, 0].astype(np.int64) + alleles[:, :, 1].astype(np.int64)
    )
    observed = target.copy()
    if masked_sites.size:
        observed[masked_sites[:, 0], masked_sites[:, 1]] = 0
    return np.stack((observed, target), axis=-1)

import numpy as np


def segment_genotype_windows(
    encoded: "np.ndarray",
    genomic_positions: "np.ndarray",
    window_size: int = 6,
    overlap: int = 3,
) -> "np.ndarray":
    """Reference implementation."""
    encoded = np.asarray(encoded)
    genomic_positions = np.asarray(genomic_positions, dtype=float)
    if (
        encoded.ndim != 3
        or encoded.shape[2] != 2
        or encoded.shape[0] == 0
        or encoded.shape[1] == 0
    ):
        raise ValueError("encoded must have shape (n_samples, n_variants, 2)")
    if not np.all(encoded == np.floor(encoded)):
        raise ValueError("encoded states must be integers")
    observed = encoded[:, :, 0]
    target = encoded[:, :, 1]
    if not np.all((observed >= 0) & (observed <= 4)) or not np.all(
        (target >= 1) & (target <= 4)
    ):
        raise ValueError("encoded states are outside the genotype vocabulary")
    if not np.all((observed == 0) | (observed == target)):
        raise ValueError("an observed state must equal its target or be MASK")
    if genomic_positions.ndim != 1 or len(genomic_positions) != encoded.shape[1]:
        raise ValueError("genomic_positions must match the variant axis")
    if not np.all(np.isfinite(genomic_positions)) or np.any(genomic_positions <= 0):
        raise ValueError("genomic_positions must be finite and positive")
    if np.any(np.diff(genomic_positions) <= 0):
        raise ValueError("genomic_positions must be strictly increasing")
    if (
        not isinstance(window_size, (int, np.integer))
        or isinstance(window_size, (bool, np.bool_))
        or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer")
    if not isinstance(overlap, (int, np.integer)) or isinstance(
        overlap, (bool, np.bool_)
    ):
        raise ValueError("overlap must be an integer")  # noqa: TRY004
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be in [0, window_size)")

    stride = window_size - overlap
    starts = list(range(0, encoded.shape[1], stride))
    width = window_size + 2
    segments = np.empty((encoded.shape[0], len(starts), width, 4), dtype=np.float64)
    for segment_index, start in enumerate(starts):
        stop = min(start + window_size, encoded.shape[1])
        count = stop - start
        for sample_index in range(encoded.shape[0]):
            block = np.empty((width, 4), dtype=np.float64)
            block[:, 0:2] = 7.0
            block[:, 2] = 0.0
            block[:, 3] = -1.0
            block[0] = (5.0, 5.0, genomic_positions[start] - 1.0, -1.0)
            block[1 : count + 1, 0:2] = encoded[sample_index, start:stop]
            block[1 : count + 1, 2] = genomic_positions[start:stop]
            block[1 : count + 1, 3] = np.arange(start, stop, dtype=float)
            block[count + 1] = (6.0, 6.0, genomic_positions[stop - 1] + 1.0, -1.0)
            segments[sample_index, segment_index] = block
    return segments

import numpy as np


def build_genomic_bias(segments: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 4 or segments.shape[-1] != 4 or 0 in segments.shape[:3]:
        raise ValueError("segments must have shape (n_samples, n_segments, width, 4)")
    if not np.all(np.isfinite(segments)):
        raise ValueError("segments must be finite")
    tokens = segments[..., 0]
    if (
        not np.all(tokens == np.floor(tokens))
        or np.any(tokens < 0)
        or np.any(tokens > 7)
    ):
        raise ValueError("segment tokens must be integers from 0 through 7")

    result = np.zeros(segments.shape[:-1] + (2,), dtype=np.float64)
    for sample_index in range(segments.shape[0]):
        for segment_index in range(segments.shape[1]):
            row_tokens = tokens[sample_index, segment_index].astype(int)
            cls_indices = np.flatnonzero(row_tokens == 5)
            sep_indices = np.flatnonzero(row_tokens == 6)
            if (
                cls_indices.tolist() != [0]
                or len(sep_indices) != 1
                or sep_indices[0] <= 0
            ):
                raise ValueError(
                    "each segment must begin with one CLS and contain one later SEP"
                )
            sep_index = int(sep_indices[0])
            valid = row_tokens != 7
            if not np.all(valid[: sep_index + 1]) or np.any(valid[sep_index + 1 :]):
                raise ValueError("padding must be trailing after SEP")
            coordinates = segments[sample_index, segment_index, : sep_index + 1, 2]
            if np.any(np.diff(coordinates) <= 0):
                raise ValueError("non-padding coordinates must be strictly increasing")
            span = coordinates[-1] - coordinates[0]
            bias = (coordinates - coordinates[0]) / span
            result[sample_index, segment_index, : sep_index + 1, 0] = bias
            result[sample_index, segment_index, :, 1] = valid.astype(float)
    return result

import numpy as np


def project_rotary_qkv(
    segments: "np.ndarray", embedding_dim: int = 4, rope_base: float = 10000.0
) -> "np.ndarray":
    """Reference implementation."""
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 4 or segments.shape[-1] != 4 or 0 in segments.shape[:3]:
        raise ValueError("segments must have shape (n_samples, n_segments, width, 4)")
    if not np.all(np.isfinite(segments)):
        raise ValueError("segments must be finite")
    tokens = segments[..., 0]
    if (
        not np.all(tokens == np.floor(tokens))
        or np.any(tokens < 0)
        or np.any(tokens > 7)
    ):
        raise ValueError("token states must be integers from 0 through 7")
    if (
        not isinstance(embedding_dim, (int, np.integer))
        or isinstance(embedding_dim, (bool, np.bool_))
        or embedding_dim <= 0
        or embedding_dim % 2 != 0
    ):
        raise ValueError("embedding_dim must be a positive even integer")
    if not np.isfinite(rope_base) or rope_base <= 1.0:
        raise ValueError("rope_base must be finite and greater than one")

    token_axis = np.arange(1.0, 9.0)[:, None]
    hidden_axis = np.arange(1.0, embedding_dim + 1.0)[None, :]
    table = 0.45 * np.sin(token_axis * hidden_axis) + 0.15 * np.cos(
        (token_axis + 1.0) * hidden_axis
    )
    table[7] = 0.0
    embedded = table[tokens.astype(np.int64)]
    mean = embedded.mean(axis=-1, keepdims=True)
    variance = ((embedded - mean) ** 2).mean(axis=-1, keepdims=True)
    embedded = (embedded - mean) / np.sqrt(variance + 1e-5)

    row = np.arange(1.0, embedding_dim + 1.0)[:, None]
    col = np.arange(1.0, embedding_dim + 1.0)[None, :]
    wq = 0.35 * np.sin(row * col + 0.2)
    wk = 0.30 * np.cos(row * (col + 1.0) - 0.1)
    wv = 0.40 * np.sin((row + 1.0) * col + 0.3)
    query = embedded @ wq
    key = embedded @ wk
    value = embedded @ wv

    pair_index = np.arange(embedding_dim // 2, dtype=float)
    frequencies = rope_base ** (-2.0 * pair_index / embedding_dim)
    ordinal_position = np.arange(segments.shape[2], dtype=float)
    angles = ordinal_position[:, None] * frequencies[None, :]
    cosine = np.cos(angles)[None, None, :, :]
    sine = np.sin(angles)[None, None, :, :]

    def _rotate(values):
        even = values[..., 0::2]
        odd = values[..., 1::2]
        output = np.empty_like(values)
        output[..., 0::2] = even * cosine - odd * sine
        output[..., 1::2] = even * sine + odd * cosine
        return output

    return np.stack((embedded, _rotate(query), _rotate(key), value), axis=-2)

import numpy as np


def apply_genomic_attention(
    projected: "np.ndarray", bias_pack: "np.ndarray", beta: float = -1.4
) -> "np.ndarray":
    """Reference implementation."""
    projected = np.asarray(projected, dtype=float)
    bias_pack = np.asarray(bias_pack, dtype=float)
    if projected.ndim != 5 or projected.shape[-2] != 4 or 0 in projected.shape:
        raise ValueError(
            "projected must have shape (n_samples, n_segments, width, 4, d)"
        )
    if bias_pack.shape != projected.shape[:3] + (2,):
        raise ValueError("bias_pack shape must match projected token axes")
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(bias_pack)):
        raise ValueError("projected and bias_pack must be finite")
    valid = bias_pack[..., 1]
    if not np.all((valid == 0.0) | (valid == 1.0)):
        raise ValueError("validity indicators must be binary")
    if np.any(valid.sum(axis=-1) == 0):
        raise ValueError("each segment must have at least one valid key")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")

    query = projected[..., 1, :]
    key = projected[..., 2, :]
    value = projected[..., 3, :]
    hidden_dim = projected.shape[-1]
    bias = bias_pack[..., 0]
    relative = bias[..., None, :] - bias[..., :, None]
    scores = np.einsum("...id,...jd->...ij", query, key) / np.sqrt(float(hidden_dim))
    scores = scores + float(beta) * relative
    scores = np.where(valid[..., None, :] == 1.0, scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    attended = np.einsum("...ij,...jd->...id", weights, value)

    row = np.arange(1.0, hidden_dim + 1.0)[:, None]
    col = np.arange(1.0, hidden_dim + 1.0)[None, :]
    output_projection = 0.25 * np.cos(row * col + 0.4)
    return attended @ output_projection

import numpy as np


def normalize_attention_residual(
    projected: "np.ndarray", attention_output: "np.ndarray", encoder_depth: int = 1
) -> "np.ndarray":
    """Reference implementation."""
    projected = np.asarray(projected, dtype=float)
    attention_output = np.asarray(attention_output, dtype=float)
    if projected.ndim != 5 or projected.shape[-2] != 4 or 0 in projected.shape:
        raise ValueError(
            "projected must have shape (n_samples, n_segments, width, 4, d)"
        )
    if attention_output.shape != projected.shape[:3] + (projected.shape[-1],):
        raise ValueError(
            "attention_output shape must match projected token and hidden axes"
        )
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(attention_output)):
        raise ValueError("projected and attention_output must be finite")
    if (
        not isinstance(encoder_depth, (int, np.integer))
        or isinstance(encoder_depth, (bool, np.bool_))
        or encoder_depth <= 0
    ):
        raise ValueError("encoder_depth must be a positive integer")
    alpha = (2.0 * encoder_depth) ** 0.25
    residual = alpha * projected[..., 0, :] + attention_output
    mean = residual.mean(axis=-1, keepdims=True)
    variance = ((residual - mean) ** 2).mean(axis=-1, keepdims=True)
    return (residual - mean) / np.sqrt(variance + 1e-5)

import numpy as np


def apply_cnn_bottleneck(
    hidden: "np.ndarray",
    encoder_depth: int = 1,
    bottleneck_factor: float = 2.0,
    kernel_size: int = 3,
) -> "np.ndarray":
    """Reference implementation."""
    hidden = np.asarray(hidden, dtype=float)
    if hidden.ndim != 4 or 0 in hidden.shape:
        raise ValueError("hidden must have shape (n_samples, n_segments, width, d)")
    if not np.all(np.isfinite(hidden)):
        raise ValueError("hidden must be finite")
    if hidden.shape[2] % 2 != 0:
        raise ValueError("hidden token width must be even")
    if (
        not isinstance(encoder_depth, (int, np.integer))
        or isinstance(encoder_depth, (bool, np.bool_))
        or encoder_depth <= 0
    ):
        raise ValueError("encoder_depth must be a positive integer")
    if not np.isfinite(bottleneck_factor) or bottleneck_factor <= 0:
        raise ValueError("bottleneck_factor must be finite and positive")
    bottleneck_dim = int(hidden.shape[-1] * bottleneck_factor)
    if bottleneck_dim < 1:
        raise ValueError("bottleneck_factor gives zero channels")
    if (
        not isinstance(kernel_size, (int, np.integer))
        or isinstance(kernel_size, (bool, np.bool_))
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")

    def _same_cross_correlation(values, kernels):
        padding = kernels.shape[-1] // 2
        padded = np.pad(values, ((0, 0), (0, 0), (padding, padding), (0, 0)))
        output = np.zeros(values.shape[:-1] + (kernels.shape[0],), dtype=float)
        for kernel_index in range(kernels.shape[-1]):
            slab = padded[:, :, kernel_index : kernel_index + values.shape[2], :]
            output += np.einsum("...i,oi->...o", slab, kernels[:, :, kernel_index])
        return output

    input_dim = hidden.shape[-1]
    out1 = np.arange(1.0, bottleneck_dim + 1.0)[:, None, None]
    in1 = np.arange(1.0, input_dim + 1.0)[None, :, None]
    tap = np.arange(1.0, kernel_size + 1.0)[None, None, :]
    kernel1 = 0.12 * np.sin(out1 + in1 * tap)
    expanded = np.maximum(_same_cross_correlation(hidden, kernel1), 0.0)
    pooled = expanded.reshape(
        expanded.shape[0],
        expanded.shape[1],
        expanded.shape[2] // 2,
        2,
        expanded.shape[3],
    ).max(axis=3)

    out2 = np.arange(1.0, input_dim + 1.0)[:, None, None]
    in2 = np.arange(1.0, bottleneck_dim + 1.0)[None, :, None]
    kernel2 = 0.10 * np.cos(out2 * in2 + tap)
    decoded_half = np.maximum(_same_cross_correlation(pooled, kernel2), 0.0)
    decoded = np.repeat(decoded_half, 2, axis=2)

    alpha = (2.0 * encoder_depth) ** 0.25
    residual = alpha * hidden + decoded
    mean = residual.mean(axis=-1, keepdims=True)
    variance = ((residual - mean) ** 2).mean(axis=-1, keepdims=True)
    return (residual - mean) / np.sqrt(variance + 1e-5)

import numpy as np


def merge_masked_probabilities(
    hidden: "np.ndarray", segments: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Reference implementation."""
    hidden = np.asarray(hidden, dtype=float)
    segments = np.asarray(segments, dtype=float)
    masked_sites = np.asarray(masked_sites)
    if hidden.ndim != 4 or 0 in hidden.shape:
        raise ValueError("hidden must have shape (n_samples, n_segments, width, d)")
    if segments.shape != hidden.shape[:3] + (4,):
        raise ValueError("segments shape must match hidden token axes")
    if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(segments)):
        raise ValueError("hidden and segments must be finite")
    if masked_sites.ndim != 2 or masked_sites.shape[1] != 2:
        raise ValueError("masked_sites must have shape (n_masked, 2)")
    if not np.issubdtype(masked_sites.dtype, np.integer) and (
        not np.all(np.isfinite(masked_sites))
        or not np.all(masked_sites == np.floor(masked_sites))
    ):
        raise ValueError("masked_sites must contain integers")
    masked_sites = masked_sites.astype(np.int64, copy=False)
    if len({tuple(row) for row in masked_sites.tolist()}) != len(masked_sites):
        raise ValueError("masked_sites rows must be distinct")
    if masked_sites.size and (
        np.any(masked_sites[:, 0] < 0) or np.any(masked_sites[:, 0] >= hidden.shape[0])
    ):
        raise ValueError("masked sample index out of range")

    hidden_dim = hidden.shape[-1]
    row = np.arange(1.0, hidden_dim + 1.0)[:, None]
    genotype_class = np.arange(1.0, 5.0)[None, :]
    decoder = 0.55 * np.sin(
        row * (genotype_class + 1.0) + 0.25 * (genotype_class - 1.0)
    )
    decoder_bias = np.array([0.15, -0.10, 0.05, 0.0])
    logits = hidden @ decoder + decoder_bias
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    pooled = np.empty((len(masked_sites), 7), dtype=np.float64)
    for row_index, (sample_index, variant_index) in enumerate(masked_sites):
        source_indices = segments[sample_index, :, :, 3]
        observed_tokens = segments[sample_index, :, :, 0]
        occurrence = (source_indices == variant_index) & (observed_tokens == 0)
        if not np.any(occurrence):
            raise ValueError(
                "each requested site must have a masked segment occurrence"
            )
        targets = segments[sample_index, :, :, 1][occurrence]
        if not np.all(targets == targets[0]) or targets[0] not in (1.0, 2.0, 3.0, 4.0):
            raise ValueError("masked occurrences must have one valid target state")
        mean_probability = probabilities[sample_index][occurrence].mean(axis=0)
        pooled[row_index] = np.concatenate(
            (
                [float(sample_index), float(variant_index), float(targets[0])],
                mean_probability,
            )
        )
    return pooled

import numpy as np


def score_masked_likelihood(pooled: "np.ndarray") -> float:
    """Reference implementation."""
    pooled = np.asarray(pooled, dtype=float)
    if pooled.ndim != 2 or pooled.shape[1] != 7 or pooled.shape[0] == 0:
        raise ValueError("pooled must have nonempty shape (n_masked, 7)")
    if not np.all(np.isfinite(pooled)):
        raise ValueError("pooled must be finite")
    if not np.all(pooled[:, :3] == np.floor(pooled[:, :3])):
        raise ValueError("indices and target states must be integral")
    if np.any(pooled[:, :2] < 0):
        raise ValueError("sample and variant indices must be nonnegative")
    targets = pooled[:, 2].astype(np.int64)
    if np.any(targets < 1) or np.any(targets > 4):
        raise ValueError("target states must be from 1 through 4")
    probabilities = pooled[:, 3:]
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("each probability row must sum to one")
    target_probabilities = probabilities[np.arange(len(targets)), targets - 1]
    if np.any(target_probabilities <= 0.0):
        raise ValueError("target-class probabilities must be positive")
    return float(-np.log(target_probabilities).mean())

import numpy as np


def run_full_pipeline(beta: float = -1.4, overlap: int = 3) -> float:
    """Reference implementation chaining every earlier step."""
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    if (
        not isinstance(overlap, (int, np.integer))
        or isinstance(overlap, (bool, np.bool_))
        or overlap < 0
        or overlap >= 6
    ):
        raise ValueError("overlap must be an integer in [0, 6)")

    alleles = np.array(
        [
            [[0, 0], [0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]],
            [[0, 1], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
            [[1, 1], [1, 0], [1, 0], [0, 1], [0, 0], [1, 1], [0, 1], [0, 0], [1, 1]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 1], [1, 0]],
        ],
        dtype=int,
    )
    masked_sites = np.array(
        [[0, 1], [0, 6], [1, 2], [1, 7], [2, 3], [2, 8], [3, 4], [3, 6]],
        dtype=int,
    )
    genomic_positions = np.array(
        [100.0, 107.0, 125.0, 126.0, 170.0, 205.0, 206.0, 290.0, 450.0]
    )

    encoded = encode_phased_genotypes(alleles, masked_sites)  # noqa: F821
    segments = segment_genotype_windows(  # noqa: F821
        encoded, genomic_positions, window_size=6, overlap=overlap
    )
    bias_pack = build_genomic_bias(segments)  # noqa: F821
    projected = project_rotary_qkv(  # noqa: F821
        segments, embedding_dim=4, rope_base=10000.0
    )
    attention_output = apply_genomic_attention(  # noqa: F821
        projected, bias_pack, beta=beta
    )
    hidden = normalize_attention_residual(  # noqa: F821
        projected, attention_output, encoder_depth=1
    )
    encoded_hidden = apply_cnn_bottleneck(  # noqa: F821
        hidden, encoder_depth=1, bottleneck_factor=2.0, kernel_size=3
    )
    pooled = merge_masked_probabilities(  # noqa: F821
        encoded_hidden, segments, masked_sites
    )
    return score_masked_likelihood(pooled)  # noqa: F821
SCICODE_GOLD_EOF
