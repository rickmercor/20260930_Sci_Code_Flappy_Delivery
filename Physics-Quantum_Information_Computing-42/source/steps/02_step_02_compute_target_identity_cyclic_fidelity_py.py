"""
Compute the source cyclic fidelity between the fixed three-qubit benchmark target circuit and the identity candidate. The fixed target topology and chronological gate ordering are part of this subproblem's public contract; the five numeric inputs supply only the five rotation angles.

The fixed three-qubit target circuit for this subproblem is, in chronological order,



RX_0(theta_rx0),

CX_{0->1},

RX_1(theta_rx1),

CX_{1->2},

RX_2(theta_rx2),

RZ_0(theta_rz0),

CX_{0->2},

RZ_2(theta_rz2).



Computational-basis states use ordering |q0 q1 q2>, with q0 the most-significant qubit.



The five scalar inputs are exactly the five rotation angles appearing above. The directed-CX locations and gate ordering are fixed by this subproblem and are not additional inputs.



Use the source computational-basis cyclic construction to compare the target with the identity candidate. The public output is a representation-independent fidelity scalar. Boolean variable numbering, auxiliary-variable names, equivalent auxiliary polarities, and CNF clause ordering are implementation details and are not part of the public result.

Returns
-------
float: representation-independent three-qubit target-versus-identity Jamiołkowski fidelity produced by the source computational-basis cyclic weighted-model count
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_target_identity_cyclic_fidelity(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
) -> float:
    """
    Compute the source cyclic fidelity of the fixed three-qubit target
    against the identity candidate.

    Parameters
    ----------
    theta_rx0 : float
        Finite RX angle for qubit 0.
    theta_rx1 : float
        Finite RX angle for qubit 1.
    theta_rx2 : float
        Finite RX angle for qubit 2.
    theta_rz0 : float
        Finite RZ angle for qubit 0.
    theta_rz2 : float
        Finite RZ angle for qubit 2.

    Returns
    -------
    fidelity : float
        Source cyclic Jamiołkowski fidelity between the fixed target
        circuit and the identity candidate. The returned value lies in
        [0, 1].

    Raises
    ------
    ValueError
        If any rotation angle is not finite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import cmath
import collections
import math
import numpy as np


class _PrivateTargetBuilder02:
    """
    Private source-style target compiler.

    The variable allocation used here is intentionally not part of the public
    Step-02 contract. It may be replaced by any semantically equivalent CNF.
    """

    def __init__(self):
        self.n = 3
        self.var = 0

        self.clauses = []
        self.weights = {}

        self.x = [
            self._new_var()
            for _ in range(3)
        ]

        self.x_init = list(
            self.x
        )

    def _new_var(self):
        self.var += 1
        return self.var

    def _clause(self, literals):
        self.clauses.append(
            tuple(
                int(v)
                for v in literals
            )
        )

    def _weight(
        self,
        variable,
        true_weight,
        false_weight,
    ):
        wt = complex(
            true_weight
        )

        wf = complex(
            false_weight
        )

        if variable in self.weights:
            old_t, old_f = (
                self.weights[
                    variable
                ]
            )

            wt *= old_t
            wf *= old_f

        self.weights[
            variable
        ] = (
            wt,
            wf,
        )

    def _rx(
        self,
        qubit,
        theta,
    ):
        old = self.x[
            qubit
        ]

        X = self._new_var()
        w = self._new_var()

        # w <-> (X <-> old)
        #
        # Equivalent four-clause XOR/equality encoding.
        self._clause(
            [X, w, old]
        )

        self._clause(
            [X, -w, -old]
        )

        self._clause(
            [-X, w, -old]
        )

        self._clause(
            [-X, -w, old]
        )

        self._weight(
            w,
            math.cos(
                theta / 2.0
            ),
            -1j
            * math.sin(
                theta / 2.0
            ),
        )

        self.x[
            qubit
        ] = X

    def _rz(
        self,
        qubit,
        theta,
    ):
        # Source convention:
        # diag(1, exp(i theta)).
        self._weight(
            self.x[
                qubit
            ],
            cmath.exp(
                1j * theta
            ),
            1.0,
        )

    def _cx(
        self,
        control,
        target,
    ):
        xc = self.x[
            control
        ]

        xt = self.x[
            target
        ]

        Xt = self._new_var()

        # Xt <-> xc XOR xt.
        self._clause(
            [
                Xt,
                xc,
                -xt,
            ]
        )

        self._clause(
            [
                Xt,
                -xc,
                xt,
            ]
        )

        self._clause(
            [
                -Xt,
                xc,
                xt,
            ]
        )

        self._clause(
            [
                -Xt,
                -xc,
                -xt,
            ]
        )

        self.x[
            target
        ] = Xt


def _pack_private_target02(
    builder,
):
    """
    Retain the old deterministic 12-column serialization only as a PRIVATE
    implementation detail for later oracle helpers.

    Public Step 02 never returns this table.
    """
    rows = []

    header = np.zeros(
        12,
        dtype=float,
    )

    header[:] = [
        0,
        builder.var,
        builder.n,
        builder.x[0],
        builder.x[1],
        builder.x[2],
        builder.x_init[0],
        builder.x_init[1],
        builder.x_init[2],
        0,
        len(
            builder.clauses
        ),
        0,
    ]

    rows.append(
        header
    )

    for clause in (
        builder.clauses
    ):
        row = np.zeros(
            12,
            dtype=float,
        )

        row[0] = 1
        row[1] = len(
            clause
        )

        row[
            2:2 + len(
                clause
            )
        ] = clause

        rows.append(
            row
        )

    for variable in sorted(
        builder.weights
    ):
        wt, wf = (
            builder.weights[
                variable
            ]
        )

        row = np.zeros(
            12,
            dtype=float,
        )

        row[:6] = [
            2,
            variable,
            wt.real,
            wt.imag,
            wf.real,
            wf.imag,
        ]

        rows.append(
            row
        )

    return np.vstack(
        rows
    )


def _private_compile_target_wcnf02(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
) -> np.ndarray:
    """
    Private target-adjoint compiler retained for later oracle steps.
    Its serialization is not a public scientific contract.
    """
    angles = np.asarray(
        [
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            angles
        )
    ):
        raise ValueError(
            "all rotation angles must be finite"
        )

    builder = (
        _PrivateTargetBuilder02()
    )

    # ------------------------------------------------------------
    # Target adjoint in reverse chronological order.
    # ------------------------------------------------------------
    builder._rz(
        2,
        -theta_rz2,
    )

    builder._cx(
        0,
        2,
    )

    builder._rz(
        0,
        -theta_rz0,
    )

    builder._rx(
        2,
        -theta_rx2,
    )

    builder._cx(
        1,
        2,
    )

    builder._rx(
        1,
        -theta_rx1,
    )

    builder._cx(
        0,
        1,
    )

    builder._rx(
        0,
        -theta_rx0,
    )

    return (
        _pack_private_target02(
            builder
        )
    )


def _canonical_clauses02(
    clauses,
):
    result = []

    for clause in clauses:
        literals = set(
            clause
        )

        # Tautological clause.
        if any(
            -literal in literals
            for literal in literals
        ):
            continue

        result.append(
            tuple(
                sorted(
                    literals,
                    key=lambda x: (
                        abs(x),
                        x < 0,
                    ),
                )
            )
        )

    return tuple(
        sorted(
            result,
            key=lambda c: (
                len(c),
                c,
            ),
        )
    )


def _bcp02(
    clauses,
):
    work = [
        set(
            clause
        )
        for clause in clauses
    ]

    assignment = {}

    while True:
        if any(
            len(clause) == 0
            for clause in work
        ):
            return (
                (),
                assignment,
                True,
            )

        unit = None

        for clause in work:
            if len(clause) == 1:
                unit = next(
                    iter(
                        clause
                    )
                )
                break

        if unit is None:
            break

        variable = abs(
            unit
        )

        value = (
            unit > 0
        )

        if (
            variable
            in assignment
            and assignment[
                variable
            ]
            != value
        ):
            return (
                (),
                assignment,
                True,
            )

        assignment[
            variable
        ] = value

        next_work = []

        for clause in work:
            if unit in clause:
                continue

            if -unit in clause:
                reduced = set(
                    clause
                )

                reduced.remove(
                    -unit
                )

                if not reduced:
                    return (
                        (),
                        assignment,
                        True,
                    )

                next_work.append(
                    reduced
                )

            else:
                next_work.append(
                    clause
                )

        work = next_work

    return (
        _canonical_clauses02(
            work
        ),
        assignment,
        False,
    )


def _variables02(
    clauses,
):
    return {
        abs(
            literal
        )
        for clause in clauses
        for literal in clause
    }


def _components02(
    clauses,
):
    if not clauses:
        return []

    by_variable = (
        collections.defaultdict(
            list
        )
    )

    for index, clause in enumerate(
        clauses
    ):
        for literal in clause:
            by_variable[
                abs(
                    literal
                )
            ].append(
                index
            )

    seen = set()
    components = []

    for start in range(
        len(
            clauses
        )
    ):
        if start in seen:
            continue

        stack = [
            start
        ]

        seen.add(
            start
        )

        indices = []

        while stack:
            index = (
                stack.pop()
            )

            indices.append(
                index
            )

            for literal in clauses[
                index
            ]:
                for other in (
                    by_variable[
                        abs(
                            literal
                        )
                    ]
                ):
                    if (
                        other
                        not in seen
                    ):
                        seen.add(
                            other
                        )

                        stack.append(
                            other
                        )

        components.append(
            tuple(
                clauses[
                    index
                ]
                for index in indices
            )
        )

    return components


def _wmc02(
    clauses,
    counted,
    weights,
    cache,
):
    (
        simplified,
        propagated,
        unsat,
    ) = _bcp02(
        clauses
    )

    if unsat:
        return 0j

    counted = set(
        counted
    )

    factor = (
        1.0 + 0.0j
    )

    # ------------------------------------------------------------
    # Weights of propagated variables.
    # ------------------------------------------------------------
    for variable, value in (
        propagated.items()
    ):
        if variable in counted:
            wt, wf = (
                weights.get(
                    variable,
                    (
                        1.0 + 0.0j,
                        1.0 + 0.0j,
                    ),
                )
            )

            factor *= (
                wt
                if value
                else wf
            )

            counted.remove(
                variable
            )

    active = (
        _variables02(
            simplified
        )
    )

    # ------------------------------------------------------------
    # Free counted variables.
    # ------------------------------------------------------------
    for variable in list(
        counted
    ):
        if (
            variable
            not in active
        ):
            wt, wf = (
                weights.get(
                    variable,
                    (
                        1.0 + 0.0j,
                        1.0 + 0.0j,
                    ),
                )
            )

            factor *= (
                wt + wf
            )

            counted.remove(
                variable
            )

    if not simplified:
        return factor

    key = (
        simplified,
        tuple(
            sorted(
                counted
            )
        ),
    )

    if key in cache:
        return (
            factor
            * cache[
                key
            ]
        )

    components = (
        _components02(
            simplified
        )
    )

    if len(
        components
    ) > 1:
        result = (
            1.0 + 0.0j
        )

        for component in (
            components
        ):
            component_vars = (
                _variables02(
                    component
                )
            )

            result *= (
                _wmc02(
                    component,
                    counted
                    & component_vars,
                    weights,
                    cache,
                )
            )

    else:
        occurrence = (
            collections.Counter(
                abs(
                    literal
                )
                for clause
                in simplified
                for literal
                in clause
                if abs(
                    literal
                )
                in counted
            )
        )

        if not occurrence:
            result = (
                1.0 + 0.0j
            )

        else:
            variable = max(
                occurrence,
                key=lambda value: (
                    occurrence[
                        value
                    ],
                    -value,
                ),
            )

            wt, wf = (
                weights.get(
                    variable,
                    (
                        1.0 + 0.0j,
                        1.0 + 0.0j,
                    ),
                )
            )

            result = (
                wt
                * _wmc02(
                    simplified
                    + (
                        (
                            variable,
                        ),
                    ),
                    counted
                    - {
                        variable
                    },
                    weights,
                    cache,
                )
                +
                wf
                * _wmc02(
                    simplified
                    + (
                        (
                            -variable,
                        ),
                    ),
                    counted
                    - {
                        variable
                    },
                    weights,
                    cache,
                )
            )

    cache[
        key
    ] = result

    return (
        factor
        * result
    )


def _oracle_compute_target_identity_cyclic_fidelity(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
) -> float:
    angles = np.asarray(
        [
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            angles
        )
    ):
        raise ValueError(
            "all rotation angles must be finite"
        )

    table = (
        _private_compile_target_wcnf02(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        )
    )

    n_variables = int(
        round(
            table[
                0,
                1,
            ]
        )
    )

    final_x = [
        int(
            round(
                value
            )
        )
        for value in table[
            0,
            3:6,
        ]
    ]

    initial_x = [
        int(
            round(
                value
            )
        )
        for value in table[
            0,
            6:9,
        ]
    ]

    clauses = []
    weights = {}

    for row in table[
        1:
    ]:
        kind = int(
            round(
                row[0]
            )
        )

        if kind == 1:
            width = int(
                round(
                    row[1]
                )
            )

            clauses.append(
                tuple(
                    int(
                        round(
                            value
                        )
                    )
                    for value in row[
                        2:2
                        + width
                    ]
                )
            )

        elif kind == 2:
            variable = int(
                round(
                    row[1]
                )
            )

            weights[
                variable
            ] = (
                complex(
                    row[2],
                    row[3],
                ),
                complex(
                    row[4],
                    row[5],
                ),
            )

    # ------------------------------------------------------------
    # Cyclic comparison with the identity candidate:
    # final target-adjoint state equals initial state.
    # ------------------------------------------------------------
    for q in range(3):
        clauses.append(
            (
                final_x[q],
                -initial_x[q],
            )
        )

        clauses.append(
            (
                -final_x[q],
                initial_x[q],
            )
        )

    z = _wmc02(
        tuple(
            clauses
        ),
        set(
            range(
                1,
                n_variables + 1,
            )
        ),
        weights,
        {},
    )

    fidelity = (
        abs(z) ** 2
        / 64.0
    )

    return float(
        fidelity
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

theta_rx0 = 0.8453
theta_rx1 = 0.1729
theta_rx2 = 0.6331
theta_rz0 = 0.4217
theta_rz2 = 0.2876
""",
            "call": "compute_target_identity_cyclic_fidelity(theta_rx0, theta_rx1, theta_rx2, theta_rz0, theta_rz2)",
            "gold_call": "_oracle_compute_target_identity_cyclic_fidelity(theta_rx0, theta_rx1, theta_rx2, theta_rz0, theta_rz2)",
        },
        {
            "setup": """import numpy as np

theta_rx0 = 0.0
theta_rx1 = 0.0
theta_rx2 = 0.0
theta_rz0 = 0.0
theta_rz2 = 0.0
""",
            "call": "compute_target_identity_cyclic_fidelity(theta_rx0, theta_rx1, theta_rx2, theta_rz0, theta_rz2)",
            "gold_call": "_oracle_compute_target_identity_cyclic_fidelity(theta_rx0, theta_rx1, theta_rx2, theta_rz0, theta_rz2)",
        },
        {
            "setup": """import numpy as np

theta_rx0 = np.nan
theta_rx1 = 0.0
theta_rx2 = 0.0
theta_rz0 = 0.0
theta_rz2 = 0.0

def run_model():
    try:
        compute_target_identity_cyclic_fidelity(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_target_identity_cyclic_fidelity(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
