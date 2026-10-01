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


def build_finite_group_action(
    generator_permutations: np.ndarray,
    *,
    supercell_matrix=None,
    site_numerators=None,
    coordinate_denominator=1,
    generator_rotations=None,
    generator_translation_numerators=None,
) -> np.ndarray:
    """Reference implementation using exact quotient-lattice arithmetic."""

    def _determinant_integer(matrix: np.ndarray) -> int:
        rows = [
            [int(value) for value in row]
            for row in np.asarray(matrix, dtype=object).tolist()
        ]
        n = len(rows)

        if n == 0:
            return 1

        if any(len(row) != n for row in rows):
            raise ValueError(
                "Internal determinant input must be square."
            )

        if n == 1:
            return rows[0][0]

        work = [row[:] for row in rows]
        sign = 1
        previous_pivot = 1

        for column in range(n - 1):
            pivot_row = next(
                (
                    row
                    for row in range(column, n)
                    if work[row][column] != 0
                ),
                None,
            )

            if pivot_row is None:
                return 0

            if pivot_row != column:
                work[column], work[pivot_row] = (
                    work[pivot_row],
                    work[column],
                )
                sign *= -1

            pivot = work[column][column]

            for row in range(column + 1, n):
                for trailing_column in range(
                    column + 1,
                    n,
                ):
                    numerator = (
                        work[row][trailing_column] * pivot
                        - work[row][column]
                        * work[column][trailing_column]
                    )

                    if numerator % previous_pivot != 0:
                        raise ArithmeticError(
                            "Bareiss division was not exact."
                        )

                    work[row][trailing_column] = (
                        numerator // previous_pivot
                    )

                work[row][column] = 0

            previous_pivot = pivot

        return sign * work[-1][-1]

    def _adjugate_integer(
        matrix: np.ndarray,
    ) -> np.ndarray:
        matrix_object = np.asarray(
            matrix,
            dtype=object,
        )
        n = int(
            matrix_object.shape[0]
        )

        if n == 1:
            return np.array(
                [[1]],
                dtype=object,
            )

        adjugate = np.empty(
            (n, n),
            dtype=object,
        )

        for row in range(n):
            for column in range(n):
                minor = np.delete(
                    np.delete(
                        matrix_object,
                        column,
                        axis=0,
                    ),
                    row,
                    axis=1,
                )

                adjugate[row, column] = (
                    (-1) ** (row + column)
                ) * _determinant_integer(
                    minor
                )

        return adjugate

    if not isinstance(
        generator_permutations,
        np.ndarray,
    ):
        raise ValueError(
            "generator_permutations must be a NumPy array."
        )

    if generator_permutations.ndim != 2:
        raise ValueError(
            "generator_permutations must be two-dimensional."
        )

    if not np.issubdtype(
        generator_permutations.dtype,
        np.integer,
    ):
        raise ValueError(
            "generator_permutations must contain integers."
        )

    n_sites = int(
        generator_permutations.shape[1]
    )

    if n_sites < 1:
        raise ValueError(
            "generator_permutations must have at least one column."
        )

    expected_sites = np.arange(
        n_sites,
        dtype=int,
    )

    generators = []

    for row in generator_permutations:
        if not np.array_equal(
            np.sort(row),
            expected_sites,
        ):
            raise ValueError(
                "Every explicit generator row must be a permutation "
                "of 0 through n_sites - 1."
            )

        generators.append(
            tuple(
                int(value)
                for value in row.tolist()
            )
        )

    affine_arguments = (
        supercell_matrix,
        site_numerators,
        generator_rotations,
        generator_translation_numerators,
    )

    supplied = [
        argument is not None
        for argument in affine_arguments
    ]

    if any(supplied) and not all(supplied):
        raise ValueError(
            "The four affine-array arguments must be supplied "
            "together or all omitted."
        )

    if all(supplied):
        for name, array in zip(
            (
                "supercell_matrix",
                "site_numerators",
                "generator_rotations",
                "generator_translation_numerators",
            ),
            affine_arguments,
        ):
            if not isinstance(
                array,
                np.ndarray,
            ):
                raise ValueError(
                    f"{name} must be a NumPy array."
                )

            if not np.issubdtype(
                array.dtype,
                np.integer,
            ):
                raise ValueError(
                    f"{name} must contain integers."
                )

        if (
            supercell_matrix.ndim != 2
            or supercell_matrix.shape[0]
            != supercell_matrix.shape[1]
            or supercell_matrix.shape[0] < 1
        ):
            raise ValueError(
                "supercell_matrix must be a nonempty square array."
            )

        dimension = int(
            supercell_matrix.shape[0]
        )

        if site_numerators.shape != (
            n_sites,
            dimension,
        ):
            raise ValueError(
                "site_numerators must have shape "
                "(n_sites, dimension)."
            )

        if (
            generator_rotations.ndim != 3
            or generator_rotations.shape[1:]
            != (
                dimension,
                dimension,
            )
        ):
            raise ValueError(
                "generator_rotations must have shape "
                "(n_affine, dimension, dimension)."
            )

        n_affine = int(
            generator_rotations.shape[0]
        )

        if (
            generator_translation_numerators.shape
            != (
                n_affine,
                dimension,
            )
        ):
            raise ValueError(
                "generator_translation_numerators must have shape "
                "(n_affine, dimension)."
            )

        if (
            isinstance(
                coordinate_denominator,
                (bool, np.bool_),
            )
            or not isinstance(
                coordinate_denominator,
                (int, np.integer),
            )
            or int(
                coordinate_denominator
            ) < 1
        ):
            raise ValueError(
                "coordinate_denominator must be a positive integer."
            )

        denominator = int(
            coordinate_denominator
        )

        supercell = np.asarray(
            supercell_matrix,
            dtype=object,
        )

        determinant_supercell = (
            _determinant_integer(
                supercell
            )
        )

        if determinant_supercell == 0:
            raise ValueError(
                "supercell_matrix must be nonsingular."
            )

        adjugate_supercell = (
            _adjugate_integer(
                supercell
            )
        )

        numerator_lattice = (
            denominator * supercell
        )

        determinant_numerator_lattice = (
            _determinant_integer(
                numerator_lattice
            )
        )

        adjugate_numerator_lattice = (
            _adjugate_integer(
                numerator_lattice
            )
        )

        numerator_modulus = abs(
            determinant_numerator_lattice
        )

        def _quotient_key(
            vector: np.ndarray,
        ) -> tuple[int, ...]:
            transformed = (
                adjugate_numerator_lattice
                @ np.asarray(
                    vector,
                    dtype=object,
                )
            )

            return tuple(
                int(value) % numerator_modulus
                for value in transformed.tolist()
            )

        site_keys = [
            _quotient_key(site)
            for site in site_numerators
        ]

        if len(
            set(site_keys)
        ) != n_sites:
            raise ValueError(
                "The periodic site representatives must be distinct "
                "modulo coordinate_denominator * supercell_matrix."
            )

        site_index = {
            key: index
            for index, key in enumerate(
                site_keys
            )
        }

        determinant_modulus = abs(
            determinant_supercell
        )

        def _belongs_to_supercell_lattice(
            vector: np.ndarray,
        ) -> bool:
            transformed = (
                adjugate_supercell
                @ np.asarray(
                    vector,
                    dtype=object,
                )
            )

            return all(
                int(value) % determinant_modulus == 0
                for value in transformed.tolist()
            )

        for rotation, translation in zip(
            generator_rotations,
            generator_translation_numerators,
        ):
            rotation_object = np.asarray(
                rotation,
                dtype=object,
            )

            translation_object = np.asarray(
                translation,
                dtype=object,
            )

            if abs(
                _determinant_integer(
                    rotation_object
                )
            ) != 1:
                raise ValueError(
                    "Every generator rotation must be integral "
                    "and unimodular."
                )

            rotated_supercell = (
                rotation_object
                @ supercell
            )

            for column in range(
                dimension
            ):
                if not _belongs_to_supercell_lattice(
                    rotated_supercell[
                        :,
                        column,
                    ]
                ):
                    raise ValueError(
                        "A generator rotation does not preserve "
                        "the supercell lattice."
                    )

            induced_permutation = []

            for site in site_numerators:
                image_numerator = (
                    rotation_object
                    @ np.asarray(
                        site,
                        dtype=object,
                    )
                    + translation_object
                )

                image_key = _quotient_key(
                    image_numerator
                )

                if image_key not in site_index:
                    raise ValueError(
                        "An affine generator does not preserve "
                        "the supplied periodic site set."
                    )

                induced_permutation.append(
                    site_index[
                        image_key
                    ]
                )

            if sorted(
                induced_permutation
            ) != list(
                range(n_sites)
            ):
                raise ValueError(
                    "An affine generator does not induce a bijection."
                )

            generators.append(
                tuple(
                    induced_permutation
                )
            )

    identity = tuple(
        range(n_sites)
    )

    unique_generators = list(
        dict.fromkeys(
            generators
        )
    )

    group = {
        identity
    }

    frontier = [
        identity
    ]

    while frontier:
        current = frontier.pop()

        for generator in unique_generators:
            products = (
                tuple(
                    current[
                        generator[index]
                    ]
                    for index in range(
                        n_sites
                    )
                ),
                tuple(
                    generator[
                        current[index]
                    ]
                    for index in range(
                        n_sites
                    )
                ),
            )

            for product in products:
                if product not in group:
                    group.add(
                        product
                    )

                    frontier.append(
                        product
                    )

    return np.asarray(
        sorted(group),
        dtype=int,
    )

import numpy as np
from itertools import combinations


def enumerate_decoration_orbit_data(
    group_action: np.ndarray,
    occupation_counts: np.ndarray,
) -> np.ndarray:
    if not isinstance(group_action, np.ndarray):
        raise ValueError("group_action must be a NumPy array.")

    if group_action.ndim != 2:
        raise ValueError("group_action must be two-dimensional.")

    if not np.issubdtype(group_action.dtype, np.integer):
        raise ValueError("group_action must contain integers.")

    if not isinstance(occupation_counts, np.ndarray):
        raise ValueError("occupation_counts must be a NumPy array.")

    if occupation_counts.ndim != 1:
        raise ValueError("occupation_counts must be one-dimensional.")

    if not np.issubdtype(occupation_counts.dtype, np.integer):
        raise ValueError("occupation_counts must contain integers.")

    n_group, n_sites = group_action.shape

    if n_sites < 1:
        raise ValueError("group_action must contain at least one site.")

    expected = np.arange(n_sites, dtype=int)

    for operation in group_action:
        if not np.array_equal(np.sort(operation), expected):
            raise ValueError(
                "Every row of group_action must be a permutation."
            )

    if np.any(
        (occupation_counts < 0)
        | (occupation_counts > n_sites)
    ):
        raise ValueError(
            "occupation_counts must lie between 0 and n_sites."
        )

    def binary_code(bits):
        code = 0

        for bit in bits:
            code = (code << 1) | int(bit)

        return int(code)

    def transform(bits, operation):
        transformed = np.zeros(
            n_sites,
            dtype=int,
        )

        for site in range(n_sites):
            transformed[
                int(operation[site])
            ] = bits[site]

        return transformed

    rows = []

    for occupation_count in occupation_counts.tolist():
        seen = set()

        for occupied_sites in combinations(
            range(n_sites),
            int(occupation_count),
        ):
            decoration = np.zeros(
                n_sites,
                dtype=int,
            )

            if occupied_sites:
                decoration[
                    list(occupied_sites)
                ] = 1

            decoration_tuple = tuple(
                int(value)
                for value in decoration.tolist()
            )

            if decoration_tuple in seen:
                continue

            orbit = {}

            for operation in group_action:
                image = transform(
                    decoration,
                    operation,
                )

                orbit[
                    tuple(
                        int(value)
                        for value in image.tolist()
                    )
                ] = image

            seen.update(
                orbit.keys()
            )

            canonical = max(
                orbit.values(),
                key=binary_code,
            )

            canonical_code = binary_code(
                canonical
            )

            stabilizer_flags = []

            for operation in group_action:
                image = transform(
                    canonical,
                    operation,
                )

                stabilizer_flags.append(
                    int(
                        np.array_equal(
                            image,
                            canonical,
                        )
                    )
                )

            rows.append(
                [
                    canonical_code,
                    int(occupation_count),
                    len(orbit),
                    *stabilizer_flags,
                ]
            )

    rows.sort(
        key=lambda row: (
            list(
                occupation_counts.tolist()
            ).index(row[1]),
            -row[0],
        )
    )

    if not rows:
        return np.empty(
            (
                0,
                3 + n_group,
            ),
            dtype=int,
        )

    return np.asarray(
        rows,
        dtype=int,
    )

import numpy as np


def reduce_undirected_axis_variants(
    decoration_orbit_data: np.ndarray,
    group_action: np.ndarray,
    axis_pairs: np.ndarray,
) -> np.ndarray:
    """Reference implementation for ordering-dependent axis reduction."""

    if not isinstance(
        decoration_orbit_data,
        np.ndarray,
    ):
        raise ValueError(
            "decoration_orbit_data must be a NumPy array."
        )

    if decoration_orbit_data.ndim != 2:
        raise ValueError(
            "decoration_orbit_data must be two-dimensional."
        )

    if not np.issubdtype(
        decoration_orbit_data.dtype,
        np.integer,
    ):
        raise ValueError(
            "decoration_orbit_data must contain integers."
        )

    if not isinstance(
        group_action,
        np.ndarray,
    ):
        raise ValueError(
            "group_action must be a NumPy array."
        )

    if group_action.ndim != 2:
        raise ValueError(
            "group_action must be two-dimensional."
        )

    if not np.issubdtype(
        group_action.dtype,
        np.integer,
    ):
        raise ValueError(
            "group_action must contain integers."
        )

    if not isinstance(
        axis_pairs,
        np.ndarray,
    ):
        raise ValueError(
            "axis_pairs must be a NumPy array."
        )

    if (
        axis_pairs.ndim != 2
        or axis_pairs.shape[1] != 2
    ):
        raise ValueError(
            "axis_pairs must have shape (n_axes, 2)."
        )

    if not np.issubdtype(
        axis_pairs.dtype,
        np.integer,
    ):
        raise ValueError(
            "axis_pairs must contain integers."
        )

    group_order, n_sites = group_action.shape

    if n_sites < 1:
        raise ValueError(
            "group_action must contain at least one site."
        )

    expected_sites = np.arange(
        n_sites,
        dtype=int,
    )

    for operation in group_action:
        if not np.array_equal(
            np.sort(operation),
            expected_sites,
        ):
            raise ValueError(
                "Every row of group_action must be a permutation "
                "of 0 through n_sites - 1."
            )

    expected_width = 3 + group_order

    if decoration_orbit_data.shape[1] != expected_width:
        raise ValueError(
            "decoration_orbit_data must contain three metadata columns "
            "followed by one stabilizer flag for every group operation."
        )

    stabilizer_flags = decoration_orbit_data[
        :,
        3:,
    ]

    if np.any(
        (stabilizer_flags != 0)
        & (stabilizer_flags != 1)
    ):
        raise ValueError(
            "Stabilizer flags must be 0 or 1."
        )

    if (
        decoration_orbit_data.shape[0] > 0
        and np.any(
            np.sum(
                stabilizer_flags,
                axis=1,
            ) == 0
        )
    ):
        raise ValueError(
            "Every decoration must contain at least one "
            "stabilizer operation."
        )

    if (
        np.any(axis_pairs < 0)
        or np.any(axis_pairs >= n_sites)
    ):
        raise ValueError(
            "Every axis-pair site must be a valid site index."
        )

    normalized_pairs = [
        tuple(
            sorted(
                (
                    int(first),
                    int(second),
                )
            )
        )
        for first, second in axis_pairs.tolist()
    ]

    if any(
        first == second
        for first, second in normalized_pairs
    ):
        raise ValueError(
            "Each axis pair must contain two different sites."
        )

    if len(
        set(normalized_pairs)
    ) != len(normalized_pairs):
        raise ValueError(
            "The undirected axis pairs must be unique."
        )

    pair_to_label = {
        pair: label
        for label, pair in enumerate(
            normalized_pairs
        )
    }

    n_axes = len(
        normalized_pairs
    )

    output_rows = []

    for decoration_row in decoration_orbit_data:
        canonical_code = int(
            decoration_row[0]
        )

        occupation_count = int(
            decoration_row[1]
        )

        active_operations = np.flatnonzero(
            decoration_row[3:]
        )

        adjacency = [
            {label}
            for label in range(
                n_axes
            )
        ]

        for operation_index in active_operations.tolist():
            operation = group_action[
                int(operation_index)
            ]

            for axis_label, (
                first,
                second,
            ) in enumerate(
                normalized_pairs
            ):
                mapped_pair = tuple(
                    sorted(
                        (
                            int(
                                operation[first]
                            ),
                            int(
                                operation[second]
                            ),
                        )
                    )
                )

                if mapped_pair not in pair_to_label:
                    raise ValueError(
                        "axis_pairs must be closed under every "
                        "ordering-preserving operation."
                    )

                mapped_label = pair_to_label[
                    mapped_pair
                ]

                adjacency[
                    axis_label
                ].add(
                    mapped_label
                )

                adjacency[
                    mapped_label
                ].add(
                    axis_label
                )

        unseen = set(
            range(n_axes)
        )

        while unseen:
            start = min(
                unseen
            )

            orbit = {
                start
            }

            stack = [
                start
            ]

            while stack:
                current = stack.pop()

                for neighbour in adjacency[
                    current
                ]:
                    if neighbour not in orbit:
                        orbit.add(
                            neighbour
                        )

                        stack.append(
                            neighbour
                        )

            unseen.difference_update(
                orbit
            )

            representative = min(
                orbit
            )

            output_rows.append(
                [
                    canonical_code,
                    occupation_count,
                    representative,
                    len(orbit),
                ]
            )

    if not output_rows:
        return np.empty(
            (0, 4),
            dtype=int,
        )

    return np.asarray(
        output_rows,
        dtype=int,
    )

import numpy as np


def align_retained_profile_rows(
    retained_axis_data: np.ndarray,
    profile_keys: np.ndarray,
    dft_profiles: np.ndarray,
    mlip_profiles: np.ndarray,
) -> np.ndarray:
    """Reference implementation for key-based profile alignment."""

    arrays = {
        "retained_axis_data": retained_axis_data,
        "profile_keys": profile_keys,
        "dft_profiles": dft_profiles,
        "mlip_profiles": mlip_profiles,
    }

    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"{name} must be a NumPy array."
            )

    if (
        retained_axis_data.ndim != 2
        or retained_axis_data.shape[1] != 4
    ):
        raise ValueError(
            "retained_axis_data must have shape (n_retained, 4)."
        )

    if not np.issubdtype(
        retained_axis_data.dtype,
        np.integer,
    ):
        raise ValueError(
            "retained_axis_data must contain integers."
        )

    if (
        profile_keys.ndim != 2
        or profile_keys.shape[1] != 2
    ):
        raise ValueError(
            "profile_keys must have shape (n_profiles, 2)."
        )

    if not np.issubdtype(
        profile_keys.dtype,
        np.integer,
    ):
        raise ValueError(
            "profile_keys must contain integers."
        )

    if dft_profiles.ndim != 2:
        raise ValueError(
            "dft_profiles must be two-dimensional."
        )

    if mlip_profiles.ndim != 2:
        raise ValueError(
            "mlip_profiles must be two-dimensional."
        )

    if not np.issubdtype(
        dft_profiles.dtype,
        np.number,
    ):
        raise ValueError(
            "dft_profiles must contain numerical values."
        )

    if not np.issubdtype(
        mlip_profiles.dtype,
        np.number,
    ):
        raise ValueError(
            "mlip_profiles must contain numerical values."
        )

    if profile_keys.shape[0] != dft_profiles.shape[0]:
        raise ValueError(
            "profile_keys and dft_profiles must have the same "
            "number of rows."
        )

    if profile_keys.shape[0] != mlip_profiles.shape[0]:
        raise ValueError(
            "profile_keys and mlip_profiles must have the same "
            "number of rows."
        )

    if dft_profiles.shape[1] != mlip_profiles.shape[1]:
        raise ValueError(
            "DFT and MLIP profiles must have the same number "
            "of samples."
        )

    if dft_profiles.shape[1] < 1:
        raise ValueError(
            "Every profile must contain at least one sample."
        )

    if not np.all(
        np.isfinite(dft_profiles)
    ):
        raise ValueError(
            "dft_profiles must contain finite values."
        )

    if not np.all(
        np.isfinite(mlip_profiles)
    ):
        raise ValueError(
            "mlip_profiles must contain finite values."
        )

    profile_lookup = {}

    for row_index, key_row in enumerate(
        profile_keys
    ):
        key = (
            int(key_row[0]),
            int(key_row[1]),
        )

        if key in profile_lookup:
            raise ValueError(
                "Every profile key must occur exactly once."
            )

        profile_lookup[key] = int(
            row_index
        )

    selected_rows = []
    seen_retained_keys = set()

    for retained_row in retained_axis_data:
        key = (
            int(retained_row[0]),
            int(retained_row[2]),
        )

        if key in seen_retained_keys:
            raise ValueError(
                "retained_axis_data must not contain duplicate keys."
            )

        seen_retained_keys.add(
            key
        )

        if key not in profile_lookup:
            raise ValueError(
                "A retained decoration-axis key is missing from "
                "profile_keys."
            )

        selected_rows.append(
            profile_lookup[key]
        )

    output_width = (
        4
        + dft_profiles.shape[1]
        + mlip_profiles.shape[1]
    )

    if retained_axis_data.shape[0] == 0:
        return np.empty(
            (0, output_width),
            dtype=float,
        )

    selected_rows = np.asarray(
        selected_rows,
        dtype=int,
    )

    return np.hstack(
        (
            retained_axis_data.astype(
                float
            ),
            np.asarray(
                dft_profiles[selected_rows],
                dtype=float,
            ),
            np.asarray(
                mlip_profiles[selected_rows],
                dtype=float,
            ),
        )
    )

import numpy as np


def classify_transformation_landscapes(
    aligned_profile_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Reference implementation for six-class profile assignment."""

    if not isinstance(
        aligned_profile_rows,
        np.ndarray,
    ):
        raise ValueError(
            "aligned_profile_rows must be a NumPy array."
        )

    if aligned_profile_rows.ndim != 2:
        raise ValueError(
            "aligned_profile_rows must be two-dimensional."
        )

    if not np.issubdtype(
        aligned_profile_rows.dtype,
        np.number,
    ):
        raise ValueError(
            "aligned_profile_rows must contain numerical values."
        )

    if not isinstance(
        class_labels,
        np.ndarray,
    ):
        raise ValueError(
            "class_labels must be a NumPy array."
        )

    if (
        class_labels.ndim != 1
        or class_labels.shape != (6,)
    ):
        raise ValueError(
            "class_labels must have shape (6,)."
        )

    if not np.issubdtype(
        class_labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "class_labels must contain integers."
        )

    if len(
        set(
            int(value)
            for value in class_labels.tolist()
        )
    ) != 6:
        raise ValueError(
            "class_labels must contain six different values."
        )

    if aligned_profile_rows.shape[1] < 10:
        raise ValueError(
            "aligned_profile_rows must contain four metadata "
            "columns and two profiles having at least three "
            "samples each."
        )

    profile_width = (
        aligned_profile_rows.shape[1]
        - 4
    )

    if profile_width % 2 != 0:
        raise ValueError(
            "The DFT and MLIP profile blocks must have equal width."
        )

    n_samples = profile_width // 2

    if n_samples < 3:
        raise ValueError(
            "Every profile must contain at least three samples."
        )

    if not np.all(
        np.isfinite(
            aligned_profile_rows
        )
    ):
        raise ValueError(
            "aligned_profile_rows must contain finite values."
        )

    metadata = aligned_profile_rows[
        :,
        :4,
    ]

    if not np.all(
        metadata
        == np.rint(metadata)
    ):
        raise ValueError(
            "The first four metadata columns must be integer-valued."
        )

    def _classify_profile(
        profile: np.ndarray,
    ) -> int:
        differences = np.diff(
            profile
        )

        if np.all(
            differences < 0
        ):
            class_position = 0

        elif np.all(
            differences > 0
        ):
            class_position = 1

        else:
            first_energy = float(
                profile[0]
            )

            last_energy = float(
                profile[-1]
            )

            if first_energy == last_energy:
                raise ValueError(
                    "A nonmonotonic profile must have different "
                    "endpoint energies."
                )

            interior = profile[
                1:-1
            ]

            has_barrier = bool(
                np.max(interior)
                > max(
                    first_energy,
                    last_energy,
                )
            )

            has_intermediate = bool(
                np.min(interior)
                < min(
                    first_energy,
                    last_energy,
                )
            )

            if has_barrier == has_intermediate:
                raise ValueError(
                    "Every nonmonotonic profile must contain exactly "
                    "one allowed interior feature."
                )

            first_is_higher = (
                first_energy
                > last_energy
            )

            if has_barrier:
                class_position = (
                    2
                    if first_is_higher
                    else 3
                )

            else:
                class_position = (
                    4
                    if first_is_higher
                    else 5
                )

        return int(
            class_labels[
                class_position
            ]
        )

    output_rows = []

    for row in aligned_profile_rows:
        dft_profile = row[
            4:
            4 + n_samples
        ]

        mlip_profile = row[
            4 + n_samples:
        ]

        output_rows.append(
            [
                *(
                    int(value)
                    for value in row[:4]
                ),
                _classify_profile(
                    dft_profile
                ),
                _classify_profile(
                    mlip_profile
                ),
            ]
        )

    if not output_rows:
        return np.empty(
            (0, 6),
            dtype=int,
        )

    return np.asarray(
        output_rows,
        dtype=int,
    )

import numpy as np


def score_classification_agreement(
    classified_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Reference implementation for augmented agreement scoring."""

    if not isinstance(classified_rows, np.ndarray):
        raise ValueError("classified_rows must be a NumPy array.")

    if (
        classified_rows.ndim != 2
        or classified_rows.shape[1] != 6
    ):
        raise ValueError(
            "classified_rows must have shape (n_rows, 6)."
        )

    if not np.issubdtype(
        classified_rows.dtype,
        np.integer,
    ):
        raise ValueError(
            "classified_rows must contain integers."
        )

    if not isinstance(class_labels, np.ndarray):
        raise ValueError(
            "class_labels must be a NumPy array."
        )

    if (
        class_labels.ndim != 1
        or class_labels.size < 1
    ):
        raise ValueError(
            "class_labels must be a nonempty one-dimensional array."
        )

    if not np.issubdtype(
        class_labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "class_labels must contain integers."
        )

    labels = [
        int(value)
        for value in class_labels.tolist()
    ]

    if len(set(labels)) != len(labels):
        raise ValueError(
            "class_labels must contain different values."
        )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    n_classes = len(labels)

    confusion = np.zeros(
        (n_classes, n_classes),
        dtype=float,
    )

    for row in classified_rows:
        dft_class = int(row[-2])
        mlip_class = int(row[-1])

        if dft_class not in label_to_index:
            raise ValueError(
                "A DFT class is not present in class_labels."
            )

        if mlip_class not in label_to_index:
            raise ValueError(
                "An MLIP class is not present in class_labels."
            )

        confusion[
            label_to_index[dft_class],
            label_to_index[mlip_class],
        ] += 1.0

    total_count = float(np.sum(confusion))

    if total_count <= 0.0:
        raise ValueError(
            "At least one classified pathway is required."
        )

    score_matrix = np.zeros(
        (n_classes + 1, n_classes + 1),
        dtype=float,
    )

    score_matrix[
        :n_classes,
        :n_classes,
    ] = confusion

    score_matrix[
        :n_classes,
        n_classes,
    ] = np.sum(confusion, axis=1)

    score_matrix[
        n_classes,
        :n_classes,
    ] = np.sum(confusion, axis=0)

    score_matrix[
        n_classes,
        n_classes,
    ] = float(np.trace(confusion)) / total_count

    return score_matrix

import numpy as np


def reconstruct_integer_confusion_matrix(
    row_counts,
    row_normalized,
    significant_figures,
):
    counts = np.asarray(row_counts)
    shown = np.asarray(row_normalized, dtype=float)
    sigfigs = np.asarray(significant_figures)

    if (
        counts.ndim != 1
        or counts.size < 2
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
    ):
        raise ValueError("row_counts must be a positive integer vector")

    k = counts.size
    if (
        shown.shape != (k, k)
        or not np.all(np.isfinite(shown))
        or np.any(shown < 0)
        or np.any(shown > 1)
    ):
        raise ValueError("row_normalized must be a finite k by k array in [0,1]")

    if (
        sigfigs.shape != (k, k)
        or not np.issubdtype(sigfigs.dtype, np.integer)
        or np.any(sigfigs < 1)
        or np.any(sigfigs > 15)
    ):
        raise ValueError(
            "significant_figures must be an integer k by k array in [1,15]"
        )

    out = np.zeros((k, k), dtype=int)
    for i, n0 in enumerate(counts.tolist()):
        n = int(n0)
        candidates = []
        for displayed, s0 in zip(shown[i], sigfigs[i]):
            s = int(s0)
            target = format(float(displayed), f".{s}g")
            values = [
                x
                for x in range(n + 1)
                if format(x / n, f".{s}g") == target
            ]
            if not values:
                raise ValueError(
                    "a displayed cell has no admissible integer count"
                )
            candidates.append(values)

        solutions = []

        def search(j, remaining, prefix):
            if len(solutions) > 1:
                return
            if j == k - 1:
                if remaining in candidates[j]:
                    solutions.append(prefix + [remaining])
                return
            for value in candidates[j]:
                if value <= remaining:
                    search(j + 1, remaining - value, prefix + [value])

        search(0, n, [])
        if len(solutions) != 1:
            raise ValueError(
                "each displayed row must have a unique integer reconstruction"
            )
        out[i] = solutions[0]

    return out

import numpy as np
from fractions import Fraction


def compute_kappa_retention(
    synthetic_score_matrix,
    paper_confusion,
):
    def confusion_kappa(matrix):
        a = np.asarray(matrix, dtype=float)

        if (
            a.ndim != 2
            or a.shape[0] != a.shape[1]
            or a.shape[0] < 2
            or not np.all(np.isfinite(a))
        ):
            raise ValueError(
                "confusion matrix must be a finite square array"
            )

        if (
            np.any(a < 0)
            or not np.allclose(
                a,
                np.rint(a),
                atol=1e-12,
                rtol=0,
            )
        ):
            raise ValueError(
                "confusion counts must be nonnegative integers"
            )

        z = np.rint(a).astype(object)
        n = int(np.sum(z))

        if n <= 0:
            raise ValueError(
                "confusion matrix must contain observations"
            )

        diagonal = sum(
            int(z[i, i])
            for i in range(z.shape[0])
        )

        rows = [
            sum(
                int(v)
                for v in z[i, :]
            )
            for i in range(z.shape[0])
        ]

        cols = [
            sum(
                int(v)
                for v in z[:, j]
            )
            for j in range(z.shape[1])
        ]

        po = Fraction(
            diagonal,
            n,
        )

        pe = Fraction(
            sum(
                r * c
                for r, c in zip(
                    rows,
                    cols,
                )
            ),
            n * n,
        )

        if pe >= 1:
            raise ValueError(
                "kappa denominator must be positive"
            )

        return (
            po - pe
        ) / (
            1 - pe
        )

    score = np.asarray(
        synthetic_score_matrix,
        dtype=float,
    )

    paper = np.asarray(
        paper_confusion
    )

    if (
        score.ndim != 2
        or score.shape[0] != score.shape[1]
        or score.shape[0] < 3
    ):
        raise ValueError(
            "synthetic_score_matrix must be an augmented square matrix"
        )

    k = score.shape[0] - 1

    if paper.shape != (k, k):
        raise ValueError(
            "paper matrix class dimension must match"
        )

    syn = score[
        :k,
        :k,
    ]

    total = float(
        np.sum(syn)
    )

    if not np.allclose(
        score[:k, k],
        np.sum(
            syn,
            axis=1,
        ),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "synthetic row totals are inconsistent"
        )

    if not np.allclose(
        score[k, :k],
        np.sum(
            syn,
            axis=0,
        ),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "synthetic column totals are inconsistent"
        )

    if (
        total <= 0
        or not np.isclose(
            score[k, k],
            np.trace(syn) / total,
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ValueError(
            "synthetic raw agreement is inconsistent"
        )

    kappa_synthetic = confusion_kappa(
        syn
    )

    kappa_paper = confusion_kappa(
        paper
    )

    if abs(
        kappa_paper
    ) <= 1e-15:
        raise ValueError(
            "paper kappa must be nonzero"
        )

    result = float(
        kappa_synthetic
        / kappa_paper
    )

    if not np.isfinite(
        result
    ):
        raise ValueError(
            "kappa retention must be finite"
        )

    return result

import numpy as np


def resolve_mg_nd_kappa_retention(
    generator_permutations,
    occupation_counts,
    axis_pairs,
    profile_keys,
    dft_profiles,
    mlip_profiles,
    class_labels,
    paper_row_counts,
    paper_row_normalized,
    paper_significant_figures,
):
    group_action = build_finite_group_action(generator_permutations)
    decoration_orbit_data = enumerate_decoration_orbit_data(
        group_action,
        occupation_counts,
    )
    retained_axis_data = reduce_undirected_axis_variants(
        decoration_orbit_data,
        group_action,
        axis_pairs,
    )
    aligned_profile_rows = align_retained_profile_rows(
        retained_axis_data,
        profile_keys,
        dft_profiles,
        mlip_profiles,
    )
    classified_rows = classify_transformation_landscapes(
        aligned_profile_rows,
        class_labels,
    )
    score_matrix = score_classification_agreement(
        classified_rows,
        class_labels,
    )
    paper_confusion = reconstruct_integer_confusion_matrix(
        paper_row_counts,
        paper_row_normalized,
        paper_significant_figures,
    )
    result = compute_kappa_retention(
        score_matrix,
        paper_confusion,
    )
    if not np.isfinite(result):
        raise ValueError("final kappa retention must be finite")
    return float(result)
SCICODE_GOLD_EOF
