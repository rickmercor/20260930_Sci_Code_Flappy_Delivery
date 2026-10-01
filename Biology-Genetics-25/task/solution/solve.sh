#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def infer_feature_count(
    coalition_values: list[float]
) -> int:
    n = len(coalition_values)

    if n < 4 or (n & (n - 1)):
        raise ValueError(
            "coalition_values must contain a complete power-of-two "
            "coalition table representing at least two features"
        )

    return n.bit_length() - 1

def compute_pair_coalition_contrast(
    coalition_values: list[float],
    feature_i: int,
    feature_j: int,
    coalition_mask: int
) -> float:
    num_features = infer_feature_count(
        coalition_values
    )

    if (
        feature_i < 0
        or feature_j < 0
        or feature_i >= num_features
        or feature_j >= num_features
        or feature_i == feature_j
    ):
        raise ValueError(
            "feature_i and feature_j must be distinct valid feature indices"
        )

    block_size = 1 << num_features

    if coalition_mask < 0 or coalition_mask >= block_size:
        raise ValueError(
            "coalition_mask is outside the represented coalition range"
        )

    bit_i = 1 << feature_i
    bit_j = 1 << feature_j

    if coalition_mask & (bit_i | bit_j):
        raise ValueError(
            "coalition_mask must exclude both members of the feature pair"
        )

    s = coalition_mask
    si = s | bit_i
    sj = s | bit_j
    sij = s | bit_i | bit_j

    return float(
        coalition_values[sij]
        - coalition_values[si]
        - coalition_values[sj]
        + coalition_values[s]
    )

def compute_interaction_coalition_weight(
    num_features: int,
    coalition_size: int
) -> float:
    import math

    if num_features < 2:
        raise ValueError(
            "num_features must be at least two"
        )

    if coalition_size < 0 or coalition_size > num_features - 2:
        raise ValueError(
            "coalition_size must lie between 0 and num_features - 2"
        )

    numerator = (
        math.factorial(coalition_size)
        * math.factorial(
            num_features - coalition_size - 2
        )
    )

    denominator = (
        2.0
        * math.factorial(num_features - 1)
    )

    return float(numerator / denominator)

def compute_pair_shap_interaction(
    coalition_values: list[float],
    num_features: int,
    feature_i: int,
    feature_j: int
) -> float:
    import math

    if num_features < 2:
        raise ValueError(
            "num_features must be at least two"
        )

    if (
        feature_i < 0
        or feature_j < 0
        or feature_i >= num_features
        or feature_j >= num_features
        or feature_i == feature_j
    ):
        raise ValueError(
            "feature_i and feature_j must be distinct valid feature indices"
        )

    block_size = 1 << num_features

    if len(coalition_values) == 0:
        raise ValueError(
            "at least one complete coalition table is required"
        )

    if len(coalition_values) % block_size != 0:
        raise ValueError(
            "coalition_values length must be a multiple of the complete "
            "coalition-table size"
        )

    if any(
        not math.isfinite(value)
        for value in coalition_values
    ):
        raise ValueError(
            "coalition_values must contain only finite values"
        )

    number_individuals = (
        len(coalition_values) // block_size
    )

    pair_bits = (
        (1 << feature_i)
        | (1 << feature_j)
    )

    total_interaction = 0.0

    for individual_index in range(number_individuals):
        start = individual_index * block_size
        stop = start + block_size

        individual_values = coalition_values[start:stop]
        individual_interaction = 0.0

        for coalition_mask in range(block_size):
            if coalition_mask & pair_bits:
                continue

            coalition_size = bin(
                coalition_mask
            ).count("1")

            contrast = (
                compute_pair_coalition_contrast(
                    individual_values,
                    feature_i,
                    feature_j,
                    coalition_mask
                )
            )

            coefficient = (
                compute_interaction_coalition_weight(
                    num_features,
                    coalition_size
                )
            )

            individual_interaction += (
                coefficient * contrast
            )

        total_interaction += individual_interaction

    return float(
        total_interaction / number_individuals
    )

def compute_pair_raw_significance(
    observed_interaction: float,
    background_interactions: list[float]
) -> float:
    import math

    if not math.isfinite(observed_interaction):
        raise ValueError(
            "observed_interaction must be finite"
        )

    if len(background_interactions) < 2:
        raise ValueError(
            "background_interactions must contain at least two values"
        )

    if any(
        not math.isfinite(value)
        for value in background_interactions
    ):
        raise ValueError(
            "background_interactions must contain only finite values"
        )

    n = len(background_interactions)

    mean_background = (
        sum(background_interactions) / n
    )

    squared_deviations = sum(
        (value - mean_background) ** 2
        for value in background_interactions
    )

    sample_variance = (
        squared_deviations / (n - 1)
    )

    if sample_variance <= 0.0:
        raise ValueError(
            "background_interactions must have positive sample variance"
        )

    sample_sd = math.sqrt(sample_variance)

    standardized_value = (
        (observed_interaction - mean_background)
        / sample_sd
    )

    return float(
        math.erfc(
            abs(standardized_value)
            / math.sqrt(2.0)
        )
    )

def locate_pair_background(
    num_features: int,
    feature_i: int,
    feature_j: int
) -> int:
    if num_features < 2:
        raise ValueError(
            "num_features must be at least two"
        )

    if (
        feature_i < 0
        or feature_j < 0
        or feature_i >= num_features
        or feature_j >= num_features
        or feature_i == feature_j
    ):
        raise ValueError(
            "feature_i and feature_j must be distinct valid feature indices"
        )

    a = min(feature_i, feature_j)
    b = max(feature_i, feature_j)

    pair_index = 0

    for first in range(num_features - 1):
        for second in range(
            first + 1,
            num_features
        ):
            if first == a and second == b:
                return pair_index

            pair_index += 1

    raise ValueError(
        "requested feature pair could not be resolved"
    )

def compute_joint_interaction_significance(
    coalition_values: list[float],
    num_features: int,
    feature_i: int,
    feature_j: int,
    background_interactions_by_pair: list[list[float]]
) -> float:
    target_pair_index = (
        locate_pair_background(
            num_features,
            feature_i,
            feature_j
        )
    )

    expected_pair_count = (
        num_features * (num_features - 1) // 2
    )

    if (
        len(background_interactions_by_pair)
        != expected_pair_count
    ):
        raise ValueError(
            "background_interactions_by_pair must contain exactly one "
            "distribution for every unordered feature pair"
        )

    raw_significance_values = []
    pair_index = 0

    for first in range(num_features - 1):
        for second in range(
            first + 1,
            num_features
        ):
            observed_interaction = (
                compute_pair_shap_interaction(
                    coalition_values,
                    num_features,
                    first,
                    second
                )
            )

            raw_significance = (
                compute_pair_raw_significance(
                    observed_interaction,
                    background_interactions_by_pair[
                        pair_index
                    ]
                )
            )

            raw_significance_values.append(
                raw_significance
            )

            pair_index += 1

    ranked = sorted(
        [
            (value, original_index)
            for original_index, value
            in enumerate(
                raw_significance_values
            )
        ],
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    number_tests = len(ranked)

    adjusted_ranked = []

    for rank, (value, _) in enumerate(
        ranked,
        start=1
    ):
        adjusted_ranked.append(
            min(
                1.0,
                value * number_tests / rank
            )
        )

    running_minimum = 1.0

    for ranked_index in range(
        number_tests - 1,
        -1,
        -1
    ):
        running_minimum = min(
            running_minimum,
            adjusted_ranked[ranked_index]
        )

        adjusted_ranked[
            ranked_index
        ] = running_minimum

    for ranked_index, (
        _,
        original_index
    ) in enumerate(ranked):
        if original_index == target_pair_index:
            return float(
                adjusted_ranked[ranked_index]
            )

    raise ValueError(
        "target pair was not present in the evaluated pair set"
    )

def compute_target_interaction_adjusted_significance(
    coalition_values: list[float],
    feature_i: int,
    feature_j: int,
    background_interactions_by_pair: list[list[float]]
) -> float:
    num_features = (
        infer_feature_count(
            coalition_values
        )
    )

    return float(
        compute_joint_interaction_significance(
            coalition_values,
            num_features,
            feature_i,
            feature_j,
            background_interactions_by_pair
        )
    )
SCICODE_GOLD_EOF
