"""
Build the complete group of site permutations from the supplied generators. The input may contain direct permutation generators. It may also contain rotation and translation data for periodic sites. When periodic data are given, first convert every affine operation into a site permutation using exact integer calculations. After collecting all valid generators, repeatedly compose them until no new permutation is obtained. Return every different group element once. The identity permutation must be included.

A crystal structure can have many symmetry operations. When a finite set of atomic sites is used, each symmetry operation can be represented as a permutation of the site indices.Sometimes these site permutations are given directly. In a periodic crystal, the symmetry can also be given using an integer rotation and translation. After applying the symmetry operation, the transformed site must match one of the original sites after allowing translation by the supercell lattice.Periodic-site matching should therefore use exact integer lattice arithmetic instead of floating-point coordinate comparison. Two positions represent the same periodic site when their difference belongs to the supercell lattice.The given symmetry operations may only be generators and not the complete symmetry group. The full finite group is obtained by repeatedly composing the generators until no new site permutation is produced.For the six-site example used later in this task, a rotation and a reflection generate twelve different site permutations. This complete group is required for finding symmetry-equivalent chemical decorations and the symmetry operations that preserve each ordering.

Returns
-------
np.ndarray of integers with shape (group_order, n_sites), containing every generated site permutation exactly once in lexicographic row order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_finite_group_action(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return unit-test specifications for exact finite-group compilation."""
    return [
        {
            "setup": """import numpy as np

generator_permutations = np.array([
    [1, 2, 3, 4, 5, 0],
    [0, 5, 4, 3, 2, 1],
], dtype=int)""",
            "call": """build_finite_group_action(generator_permutations)""",
            "gold_call": """_oracle_build_finite_group_action(generator_permutations)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.array([
    [1, 2, 3, 0],
], dtype=int)""",
            "call": """build_finite_group_action(generator_permutations)""",
            "gold_call": """_oracle_build_finite_group_action(generator_permutations)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.empty((0, 5), dtype=int)""",
            "call": """build_finite_group_action(generator_permutations)""",
            "gold_call": """_oracle_build_finite_group_action(generator_permutations)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.empty((0, 6), dtype=int)
supercell_matrix = np.array([[6]], dtype=int)
site_numerators = np.arange(6, dtype=int).reshape(-1, 1)
coordinate_denominator = 1
generator_rotations = np.array([
    [[1]],
    [[-1]],
], dtype=int)
generator_translation_numerators = np.array([
    [1],
    [0],
], dtype=int)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

physical_order = np.array([3, 0, 5, 2, 1, 4], dtype=int)
generator_permutations = np.empty((0, 6), dtype=int)
supercell_matrix = np.array([[6]], dtype=int)
site_numerators = physical_order.reshape(-1, 1)
coordinate_denominator = 1
generator_rotations = np.array([
    [[-1]],
    [[1]],
    [[1]],
    [[1]],
    [[-1]],
], dtype=int)
generator_translation_numerators = np.array([
    [0],
    [5],
    [0],
    [1],
    [6],
], dtype=int)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.array([
    [0, 3, 2, 1],
], dtype=int)
supercell_matrix = np.array([[4]], dtype=int)
site_numerators = np.arange(4, dtype=int).reshape(-1, 1)
coordinate_denominator = 1
generator_rotations = np.array([
    [[1]],
], dtype=int)
generator_translation_numerators = np.array([
    [1],
], dtype=int)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.empty((0, 6), dtype=int)
supercell_matrix = np.array([
    [2, 1],
    [0, 3],
], dtype=int)

large = 10_000_000_000_000_000
lattice_coordinates = np.array([
    [large + i, -large + 2 * i]
    for i in range(6)
], dtype=np.int64)
base_sites = np.array([
    [0, k]
    for k in range(6)
], dtype=np.int64)

site_numerators = (
    base_sites
    + lattice_coordinates @ supercell_matrix.T
)
coordinate_denominator = 1
generator_rotations = np.array([
    [[1, 0], [0, 1]],
    [[-1, 0], [0, -1]],
], dtype=int)
generator_translation_numerators = np.array([
    [0, 1],
    [0, 0],
], dtype=int)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.empty((0, 3), dtype=int)
supercell_matrix = np.array([[3]], dtype=int)
site_numerators = np.array([
    [1],
    [3],
    [5],
], dtype=int)
coordinate_denominator = 2
generator_rotations = np.array([
    [[1]],
    [[-1]],
], dtype=int)
generator_translation_numerators = np.array([
    [2],
    [0],
], dtype=int)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np
from itertools import product

generator_permutations = np.empty((0, 8), dtype=int)

m = 10007
u = np.array([
    [1, m, 0],
    [0, 1, m],
    [0, 0, 1],
], dtype=object)
u_inverse = np.array([
    [1, -m, m * m],
    [0, 1, -m],
    [0, 0, 1],
], dtype=object)

h0 = 2 * np.eye(3, dtype=object)
supercell_matrix = np.asarray(u @ h0, dtype=np.int64)

binary_sites = np.array(
    list(product([0, 1], repeat=3)),
    dtype=object,
)
base_sites = (u @ binary_sites.T).T

large = 100_000_000_000
site_rows = []
for index, base in enumerate(base_sites):
    lattice_shift = np.array([
        large + index,
        -2 * large + 3 * index,
        3 * large - 5 * index,
    ], dtype=object)
    site_rows.append(
        base + np.asarray(supercell_matrix, dtype=object)
        @ lattice_shift
    )

scramble = [5, 0, 7, 2, 6, 1, 4, 3]
site_numerators = np.asarray(
    np.array(site_rows, dtype=object)[scramble],
    dtype=np.int64,
)
coordinate_denominator = 1

identity = np.eye(3, dtype=object)
axis_cycle = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=object)
binary_shear = np.array([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=object)

base_rotations = [
    identity,
    axis_cycle,
    binary_shear,
]
base_translations = [
    np.array([1, 0, 0], dtype=object),
    np.zeros(3, dtype=object),
    np.zeros(3, dtype=object),
]

generator_rotations = np.asarray([
    u @ rotation @ u_inverse
    for rotation in base_rotations
], dtype=np.int64)
generator_translation_numerators = np.asarray([
    u @ translation
    for translation in base_translations
], dtype=np.int64)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np
from itertools import product

generator_permutations = np.empty((0, 9), dtype=int)

coordinate_denominator = 6
m = 1_000_003
u = np.array([
    [1, m],
    [0, 1],
], dtype=object)
u_inverse = np.array([
    [1, -m],
    [0, 1],
], dtype=object)

h0 = 3 * np.eye(2, dtype=object)
supercell_matrix = np.asarray(u @ h0, dtype=np.int64)
offset = np.array([1, 5], dtype=object)

residues = np.array(
    list(product(range(3), repeat=2)),
    dtype=object,
)
base_sites = np.array([
    u @ (
        offset
        + coordinate_denominator * residue
    )
    for residue in residues
], dtype=object)

large = 50_000_000
site_rows = []
for index, base in enumerate(base_sites):
    lattice_shift = np.array([
        large + 7 * index,
        -large + 11 * index,
    ], dtype=object)
    site_rows.append(
        base
        + coordinate_denominator
        * (
            np.asarray(supercell_matrix, dtype=object)
            @ lattice_shift
        )
    )

scramble = [8, 0, 5, 2, 7, 3, 1, 6, 4]
site_numerators = np.asarray(
    np.array(site_rows, dtype=object)[scramble],
    dtype=np.int64,
)

identity = np.eye(2, dtype=object)
axis_swap = np.array([
    [0, 1],
    [1, 0],
], dtype=object)
ternary_shear = np.array([
    [1, 1],
    [0, 1],
], dtype=object)
reflection = np.array([
    [-1, 0],
    [0, 1],
], dtype=object)

base_rotations = [
    identity,
    axis_swap,
    ternary_shear,
    reflection,
]
residue_translations = [
    np.array([1, 0], dtype=object),
    np.zeros(2, dtype=object),
    np.zeros(2, dtype=object),
    np.zeros(2, dtype=object),
]

generator_rotations = []
generator_translation_numerators = []

for rotation, residue_translation in zip(
    base_rotations,
    residue_translations,
):
    translation = (
        offset
        - rotation @ offset
        + coordinate_denominator
        * residue_translation
    )
    generator_rotations.append(
        u @ rotation @ u_inverse
    )
    generator_translation_numerators.append(
        u @ translation
    )

generator_rotations = np.asarray(
    generator_rotations,
    dtype=np.int64,
)
generator_translation_numerators = np.asarray(
    generator_translation_numerators,
    dtype=np.int64,
)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

n_sites = 7
coordinate_denominator = 3
offset = 1
physical_order = np.array(
    [4, 0, 6, 2, 5, 1, 3],
    dtype=int,
)

index_of = {
    int(physical_site): index
    for index, physical_site in enumerate(
        physical_order
    )
}
explicit_swap = list(range(n_sites))
left = index_of[0]
right = index_of[1]
explicit_swap[left], explicit_swap[right] = (
    explicit_swap[right],
    explicit_swap[left],
)

generator_permutations = np.array([
    explicit_swap,
    explicit_swap,
], dtype=int)

supercell_matrix = np.array(
    [[n_sites]],
    dtype=int,
)

large = 100_000_000_000_000
lattice_shifts = np.array([
    large - 7 * index
    for index in range(n_sites)
], dtype=object)

site_numerators = np.asarray([
    [
        offset
        + coordinate_denominator
        * int(physical_site)
        + coordinate_denominator
        * n_sites
        * int(lattice_shift)
    ]
    for physical_site, lattice_shift in zip(
        physical_order,
        lattice_shifts,
    )
], dtype=np.int64)

generator_rotations = np.array([
    [[1]],
    [[-1]],
    [[1]],
], dtype=int)
generator_translation_numerators = np.array([
    [coordinate_denominator],
    [2 * offset],
    [coordinate_denominator * n_sites],
], dtype=int)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.empty(
    (0, 12),
    dtype=int,
)
coordinate_denominator = 4

m = 50021
n = 70001
u = np.array([
    [1, m, 0],
    [0, 1, n],
    [0, 0, 1],
], dtype=object)
u_inverse = np.array([
    [1, -m, m * n],
    [0, 1, -n],
    [0, 0, 1],
], dtype=object)

h0 = np.diag([2, 2, 3]).astype(object)
supercell_matrix = np.asarray(
    u @ h0,
    dtype=np.int64,
)
offset = np.array([1, 3, 2], dtype=object)

residues = []
for first in range(2):
    for second in range(2):
        for third in range(3):
            residues.append(
                np.array(
                    [first, second, third],
                    dtype=object,
                )
            )

base_sites = np.array([
    u @ (
        offset
        + coordinate_denominator * residue
    )
    for residue in residues
], dtype=object)

large = 100_000_000
site_rows = []
for index, base in enumerate(base_sites):
    lattice_shift = np.array([
        large + 13 * index,
        -2 * large + 17 * index,
        3 * large - 19 * index,
    ], dtype=object)
    site_rows.append(
        base
        + coordinate_denominator
        * (
            np.asarray(
                supercell_matrix,
                dtype=object,
            )
            @ lattice_shift
        )
    )

scramble = [
    11, 0, 7, 3, 9, 2,
    5, 8, 1, 10, 4, 6,
]
site_numerators = np.asarray(
    np.array(site_rows, dtype=object)[scramble],
    dtype=np.int64,
)

identity = np.eye(3, dtype=object)
first_axis_swap = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
], dtype=object)
binary_shear = np.array([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=object)
third_axis_reflection = np.diag(
    [1, 1, -1]
).astype(object)

base_rotations = [
    identity,
    first_axis_swap,
    binary_shear,
    identity,
    third_axis_reflection,
]
residue_translations = [
    np.array([1, 0, 0], dtype=object),
    np.zeros(3, dtype=object),
    np.zeros(3, dtype=object),
    np.array([0, 0, 1], dtype=object),
    np.zeros(3, dtype=object),
]

generator_rotations = []
generator_translation_numerators = []

for rotation, residue_translation in zip(
    base_rotations,
    residue_translations,
):
    translation = (
        offset
        - rotation @ offset
        + coordinate_denominator
        * residue_translation
    )
    generator_rotations.append(
        u @ rotation @ u_inverse
    )
    generator_translation_numerators.append(
        u @ translation
    )

generator_rotations = np.asarray(
    generator_rotations,
    dtype=np.int64,
)
generator_translation_numerators = np.asarray(
    generator_translation_numerators,
    dtype=np.int64,
)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np

generator_permutations = np.empty(
    (0, 5),
    dtype=int,
)
coordinate_denominator = 7

m = 2_000_003
u = np.array([
    [-1, m],
    [0, 1],
], dtype=object)
u_inverse = u.copy()

h0 = np.array([
    [5, 0],
    [0, 1],
], dtype=object)
supercell_matrix = np.asarray(
    u @ h0,
    dtype=np.int64,
)
offset = np.array([2, 3], dtype=object)

base_sites = np.array([
    u @ (
        offset
        + coordinate_denominator
        * np.array([residue, 0], dtype=object)
    )
    for residue in range(5)
], dtype=object)

large = 20_000_000
site_rows = []
for index, base in enumerate(base_sites):
    lattice_shift = np.array([
        large + 19 * index,
        -large + 23 * index,
    ], dtype=object)
    site_rows.append(
        base
        + coordinate_denominator
        * (
            np.asarray(
                supercell_matrix,
                dtype=object,
            )
            @ lattice_shift
        )
    )

scramble = [3, 0, 4, 1, 2]
site_numerators = np.asarray(
    np.array(site_rows, dtype=object)[scramble],
    dtype=np.int64,
)

identity = np.eye(2, dtype=object)
infinite_order_lift = np.array([
    [2, 5],
    [1, 3],
], dtype=object)

base_rotations = [
    identity,
    infinite_order_lift,
    infinite_order_lift @ infinite_order_lift,
]
residue_translations = [
    np.array([1, 0], dtype=object),
    np.zeros(2, dtype=object),
    np.zeros(2, dtype=object),
]

generator_rotations = []
generator_translation_numerators = []

for rotation, residue_translation in zip(
    base_rotations,
    residue_translations,
):
    translation = (
        offset
        - rotation @ offset
        + coordinate_denominator
        * residue_translation
    )
    generator_rotations.append(
        u @ rotation @ u_inverse
    )
    generator_translation_numerators.append(
        u @ translation
    )

generator_rotations = np.asarray(
    generator_rotations,
    dtype=np.int64,
)
generator_translation_numerators = np.asarray(
    generator_translation_numerators,
    dtype=np.int64,
)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
        {
            "setup": """import numpy as np
from itertools import product

generator_permutations = np.empty(
    (0, 4),
    dtype=int,
)
coordinate_denominator = 2

large = 4_000_000_000_000_000
supercell_matrix = np.array([
    [large, large - 1],
    [large + 1, large],
], dtype=np.int64)

offset = np.array([1, -1], dtype=object)
binary_residues = np.array(
    list(product([0, 1], repeat=2)),
    dtype=object,
)
supercell_object = np.asarray(
    supercell_matrix,
    dtype=object,
)

site_rows = np.array([
    offset + supercell_object @ residue
    for residue in binary_residues
], dtype=object)

scramble = [3, 0, 2, 1]
site_numerators = np.asarray(
    site_rows[scramble],
    dtype=np.int64,
)

generator_rotations = np.array([
    [[1, 0], [0, 1]],
    [[1, 0], [0, 1]],
], dtype=int)
generator_translation_numerators = np.array([
    supercell_matrix[:, 0],
    supercell_matrix[:, 1],
], dtype=np.int64)""",
            "call": """build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
            "gold_call": """_oracle_build_finite_group_action(
    generator_permutations,
    supercell_matrix=supercell_matrix,
    site_numerators=site_numerators,
    coordinate_denominator=coordinate_denominator,
    generator_rotations=generator_rotations,
    generator_translation_numerators=generator_translation_numerators,
)""",
        },
    ]
