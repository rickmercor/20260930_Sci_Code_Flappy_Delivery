"""
Compute the Jamiołkowski fidelity of one semantic source-valid synthesis

schedule using the paper's computational-basis weighted Boolean construction.



The public schedule contains only gate identities and qubit operands. The

oracle privately compiles the target adjoint and the fixed candidate through

gate-specific Boolean relations and complex literal weights, adds cyclic

final-to-initial equality, and performs weighted model counting.



No private CNF variable numbering or selector-literal numbering appears in

the public contract. Direct unitary multiplication is not used to obtain the

returned value.

Condition the source synthesis construction on the supplied semantic schedule,

compose it with the target comparison construction, and evaluate the resulting

weighted Boolean model.



Return the corresponding representation-independent circuit fidelity.

Matrix multiplication may be used only as an independent check.

Returns
-------
float: Jamiołkowski fidelity |Tr(U V†)|^2/64 obtained from the source computational-basis weighted Boolean encoding of the fixed semantic schedule
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_fixed_schedule_cyclic_fidelity(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
    schedule: np.ndarray,
) -> float:
    """
    Compute source weighted-Boolean cyclic fidelity for one schedule.

    Returns
    -------
    float
        Three-qubit Jamiołkowski fidelity.

    Raises
    ------
    ValueError
        If an angle is non-finite or the schedule is malformed or
        source-invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import cmath
import collections
import math
import numpy as np


class _FixedBuilder04:
    def __init__(self, target_table):
        table = np.asarray(
            target_table,
            dtype=float,
        )

        if (
            table.ndim != 2
            or table.shape[1] != 12
            or int(round(table[0, 0])) != 0
            or int(round(table[0, 2])) != 3
        ):
            raise ValueError(
                "invalid private target encoding"
            )

        self.var = int(
            round(table[0, 1])
        )

        self.x = [
            int(round(v))
            for v in table[
                0,
                3:6,
            ]
        ]

        self.x_init = [
            int(round(v))
            for v in table[
                0,
                6:9,
            ]
        ]

        self.clauses = []
        self.weights = {}

        for row in table[1:]:
            kind = int(
                round(row[0])
            )

            if kind == 1:
                k = int(
                    round(row[1])
                )

                self.clauses.append(
                    tuple(
                        int(round(v))
                        for v in row[
                            2:2 + k
                        ]
                    )
                )

            elif kind == 2:
                variable = int(
                    round(row[1])
                )

                self.weights[
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

    def new_var(self):
        self.var += 1
        return self.var

    def clause(self, values):
        self.clauses.append(
            tuple(
                int(v)
                for v in values
            )
        )

    def weight(
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

    def identity(self, qubit):
        # Fixed semantic identity: unchanged wire.
        return

    def h(self, qubit):
        old = self.x[
            qubit
        ]

        X = self.new_var()
        h = self.new_var()

        # h <-> (X and old)
        self.clause(
            [-h, X]
        )
        self.clause(
            [-h, old]
        )
        self.clause(
            [h, -X, -old]
        )

        self.weight(
            h,
            -1.0
            / math.sqrt(2.0),
            +1.0
            / math.sqrt(2.0),
        )

        self.x[
            qubit
        ] = X

    def t(self, qubit):
        self.weight(
            self.x[
                qubit
            ],
            cmath.exp(
                1j
                * math.pi
                / 4.0
            ),
            1.0,
        )

    def tdg(self, qubit):
        self.weight(
            self.x[
                qubit
            ],
            cmath.exp(
                -1j
                * math.pi
                / 4.0
            ),
            1.0,
        )

    def cx(
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

        Xt = self.new_var()

        # Xt <-> xc XOR xt
        self.clause(
            [Xt, xc, -xt]
        )
        self.clause(
            [Xt, -xc, xt]
        )
        self.clause(
            [-Xt, xc, xt]
        )
        self.clause(
            [-Xt, -xc, -xt]
        )

        self.x[
            target
        ] = Xt

    def close_cyclic(self):
        for q in range(3):
            final = self.x[q]
            initial = (
                self.x_init[q]
            )

            self.clause(
                [final, -initial]
            )
            self.clause(
                [-final, initial]
            )


def _canonical04(clauses):
    result = []

    for clause in clauses:
        literals = set(
            clause
        )

        if any(
            -lit in literals
            for lit in literals
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


def _bcp04(clauses):
    work = [
        set(c)
        for c in clauses
    ]

    assignment = {}

    while True:
        if any(
            len(c) == 0
            for c in work
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
                    iter(clause)
                )
                break

        if unit is None:
            break

        var = abs(unit)
        value = unit > 0

        if (
            var in assignment
            and assignment[var]
            != value
        ):
            return (
                (),
                assignment,
                True,
            )

        assignment[
            var
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
        _canonical04(
            work
        ),
        assignment,
        False,
    )


def _vars04(clauses):
    return {
        abs(lit)
        for clause in clauses
        for lit in clause
    }


def _components04(clauses):
    if not clauses:
        return []

    by_var = (
        collections.defaultdict(
            list
        )
    )

    for index, clause in enumerate(
        clauses
    ):
        for lit in clause:
            by_var[
                abs(lit)
            ].append(
                index
            )

    seen = set()
    answer = []

    for start in range(
        len(clauses)
    ):
        if start in seen:
            continue

        stack = [
            start
        ]

        seen.add(start)

        group = []

        while stack:
            index = (
                stack.pop()
            )

            group.append(
                index
            )

            for lit in clauses[
                index
            ]:
                for other in by_var[
                    abs(lit)
                ]:
                    if other not in seen:
                        seen.add(
                            other
                        )

                        stack.append(
                            other
                        )

        answer.append(
            tuple(
                clauses[i]
                for i in group
            )
        )

    return answer


def _wmc04(
    clauses,
    counted,
    weights,
    cache,
):
    (
        simplified,
        propagated,
        unsat,
    ) = _bcp04(
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

    active = _vars04(
        simplified
    )

    for variable in list(
        counted
    ):
        if variable not in active:
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
            sorted(counted)
        ),
    )

    if key in cache:
        return (
            factor
            * cache[key]
        )

    components = (
        _components04(
            simplified
        )
    )

    if len(components) > 1:
        result = (
            1.0 + 0.0j
        )

        for component in components:
            result *= _wmc04(
                component,
                _vars04(
                    component
                )
                & counted,
                weights,
                cache,
            )

    else:
        occurrence = (
            collections.Counter(
                abs(lit)
                for clause in simplified
                for lit in clause
                if abs(lit)
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
                key=lambda v: (
                    occurrence[v],
                    -v,
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
                * _wmc04(
                    simplified
                    + (
                        (
                            variable,
                        ),
                    ),
                    counted
                    - {variable},
                    weights,
                    cache,
                )
                +
                wf
                * _wmc04(
                    simplified
                    + (
                        (
                            -variable,
                        ),
                    ),
                    counted
                    - {variable},
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


def _oracle_compute_fixed_schedule_cyclic_fidelity(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
    schedule: np.ndarray,
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
            "rotation angles must be finite"
        )

    if (
        _oracle_validate_source_schedule(
            schedule
        )
        != 1.0
    ):
        raise ValueError(
            "schedule is not source-valid"
        )

    normalized = (
        _normalize_schedule03(
            schedule
        )
    )

    target = (
        _private_compile_target_wcnf02(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        )
    )

    builder = _FixedBuilder04(
        target
    )

    for layer in normalized:
        operations = [
            tuple(
                int(x)
                for x in row
            )
            for row in layer
            if row[0] != -1
        ]

        # Operations inside a legal layer are disjoint and commute.
        for (
            gate_code,
            q0,
            q1,
        ) in operations:
            if gate_code == 0:
                builder.identity(
                    q0
                )

            elif gate_code == 1:
                builder.h(
                    q0
                )

            elif gate_code == 2:
                builder.t(
                    q0
                )

            elif gate_code == 3:
                builder.tdg(
                    q0
                )

            elif gate_code == 4:
                builder.cx(
                    q0,
                    q1,
                )

    builder.close_cyclic()

    z = _wmc04(
        tuple(
            builder.clauses
        ),
        set(
            range(
                1,
                builder.var + 1,
            )
        ),
        builder.weights,
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

schedule = np.array([
    [
        [1, 0, -1],
        [1, 1, -1],
        [1, 2, -1],
    ],
    [
        [2, 0, -1],
        [4, 2,  1],
        [-1,-1,-1],
    ],
    [
        [1, 0, -1],
        [1, 1, -1],
        [2, 2, -1],
    ],
    [
        [4, 0,  1],
        [1, 2, -1],
        [-1,-1,-1],
    ],
], dtype=float)
""",
            "call": "compute_fixed_schedule_cyclic_fidelity(0.8453,0.1729,0.6331,0.4217,0.2876,schedule)",
            "gold_call": "_oracle_compute_fixed_schedule_cyclic_fidelity(0.8453,0.1729,0.6331,0.4217,0.2876,schedule)",
        },
        {
            "setup": """import numpy as np

schedule = np.array([
    [
        [4,0,1],
        [0,2,-1],
        [-1,-1,-1],
    ],
], dtype=float)
""",
            "call": "compute_fixed_schedule_cyclic_fidelity(0.8453,0.1729,0.6331,0.4217,0.2876,schedule)",
            "gold_call": "_oracle_compute_fixed_schedule_cyclic_fidelity(0.8453,0.1729,0.6331,0.4217,0.2876,schedule)",
        },
        {
            "setup": """import numpy as np

schedule = np.array([
    [
        [1,0,-1],
        [1,1,-1],
        [-1,-1,-1],
    ],
], dtype=float)

def run_model():
    try:
        compute_fixed_schedule_cyclic_fidelity(
            0.8453,0.1729,0.6331,0.4217,0.2876,schedule
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_fixed_schedule_cyclic_fidelity(
            0.8453,0.1729,0.6331,0.4217,0.2876,schedule
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
