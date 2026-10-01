#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import cmath
import math
import numpy as np


def reconstruct_source_gate_encodings(
    theta_rx: float,
    theta_rz: float,
) -> np.ndarray:
    values = np.asarray(
        [theta_rx, theta_rz],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "rotation angles must be finite"
        )

    I = np.eye(
        2,
        dtype=complex,
    )

    # Source Hadamard semantics:
    # h <-> (q and q')
    # W(h) = -1/sqrt(2)
    # W(~h) = +1/sqrt(2)
    H = (
        np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=complex,
        )
        / math.sqrt(2.0)
    )

    T = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                1j
                * math.pi
                / 4.0
            ),
        ]
    )

    Tdg = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                -1j
                * math.pi
                / 4.0
            ),
        ]
    )

    c = math.cos(
        theta_rx / 2.0
    )

    s = (
        -1j
        * math.sin(
            theta_rx / 2.0
        )
    )

    RX = np.array(
        [
            [c, s],
            [s, c],
        ],
        dtype=complex,
    )

    # Source computational-basis RZ convention.
    RZ_source = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                1j
                * theta_rz
            ),
        ]
    )

    H0 = np.kron(
        H,
        I,
    )

    T0 = np.kron(
        T,
        I,
    )

    Tdg0 = np.kron(
        Tdg,
        I,
    )

    RX0 = np.kron(
        RX,
        I,
    )

    RZ0 = np.kron(
        RZ_source,
        I,
    )

    CX01 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )

    gates = np.stack(
        [
            H0,
            T0,
            Tdg0,
            RX0,
            RZ0,
            CX01,
        ],
        axis=0,
    )

    return np.stack(
        [
            gates.real,
            gates.imag,
        ],
        axis=-1,
    ).astype(float)

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


def compute_target_identity_cyclic_fidelity(
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

import numpy as np


def _normalize_schedule03(schedule):
    a = np.asarray(schedule, dtype=float)

    if (
        a.ndim != 3
        or a.shape[1:] != (3, 3)
        or a.shape[0] < 1
        or a.shape[0] > 4
    ):
        raise ValueError(
            "schedule must have shape (depth,3,3) with depth from 1 through 4"
        )

    if not np.all(np.isfinite(a)):
        raise ValueError("schedule entries must be finite")

    if not np.allclose(a, np.round(a), atol=1e-12, rtol=0.0):
        raise ValueError("schedule entries must be integers")

    a = np.round(a).astype(int)

    for layer in a:
        for gate_code, q0, q1 in layer:
            if gate_code == -1:
                if q0 != -1 or q1 != -1:
                    raise ValueError(
                        "unused slot must be exactly [-1,-1,-1]"
                    )

            elif gate_code in (0, 1, 2, 3):
                if q0 not in (0, 1, 2) or q1 != -1:
                    raise ValueError(
                        "single-qubit gates require q0 in {0,1,2} and q1=-1"
                    )

            elif gate_code == 4:
                if (
                    q0 not in (0, 1, 2)
                    or q1 not in (0, 1, 2)
                    or q0 == q1
                ):
                    raise ValueError(
                        "CX requires distinct control and target in {0,1,2}"
                    )

            else:
                raise ValueError("invalid gate code")

    return a


def _layer_actions03(layer):
    """
    Return:
      action[q] = ("single", code) or ("cx", control, target)
      cx_list = [(control,target), ...]
    """
    action = {}
    cx_list = []

    for gate_code, q0, q1 in layer:
        if gate_code == -1:
            continue

        if gate_code in (0, 1, 2, 3):
            if q0 in action:
                return None, None

            action[q0] = (
                "single",
                int(gate_code),
            )

        else:
            control = int(q0)
            target = int(q1)

            if control in action or target in action:
                return None, None

            marker = (
                "cx",
                control,
                target,
            )

            action[control] = marker
            action[target] = marker

            cx_list.append(
                (control, target)
            )

    if set(action) != {0, 1, 2}:
        return None, None

    if len(cx_list) > 1:
        return None, None

    # A complete 3-qubit layer must therefore contain either
    # three single-qubit operations or one CX + one single.
    real_slots = int(
        np.sum(
            np.asarray(layer)[:, 0]
            != -1
        )
    )

    if len(cx_list) == 0:
        if real_slots != 3:
            return None, None

    else:
        if real_slots != 2:
            return None, None

    return action, cx_list


def _single_code03(action, qubit):
    value = action[qubit]

    if value[0] != "single":
        return None

    return value[1]


def _incident_cx03(action, qubit):
    return (
        action[qubit][0]
        == "cx"
    )


def validate_source_schedule(
    schedule: np.ndarray,
) -> float:
    a = _normalize_schedule03(
        schedule
    )

    actions = []
    cx_layers = []

    for layer in a:
        action, cx_list = (
            _layer_actions03(
                layer
            )
        )

        if action is None:
            return 0.0

        actions.append(action)
        cx_layers.append(cx_list)

    depth = len(actions)

    for layer_index in range(
        1,
        depth,
    ):
        previous = actions[
            layer_index - 1
        ]

        current = actions[
            layer_index
        ]

        # --------------------------------------------------------
        # Per-qubit adjacent gate pruning.
        # --------------------------------------------------------
        for q in range(3):
            prev_code = (
                _single_code03(
                    previous,
                    q,
                )
            )

            curr_code = (
                _single_code03(
                    current,
                    q,
                )
            )

            # H followed immediately by H.
            if (
                prev_code == 1
                and curr_code == 1
            ):
                return 0.0

            # T followed by Tdg.
            if (
                prev_code == 2
                and curr_code == 3
            ):
                return 0.0

            # Tdg followed by T.
            if (
                prev_code == 3
                and curr_code == 2
            ):
                return 0.0

            # Identity-padding canonicalization:
            # previous I => current I unless a current CX touches q.
            if prev_code == 0:
                if (
                    curr_code != 0
                    and not _incident_cx03(
                        current,
                        q,
                    )
                ):
                    return 0.0

        # --------------------------------------------------------
        # Consecutive-CX pruning.
        # --------------------------------------------------------
        previous_cx = set(
            cx_layers[
                layer_index - 1
            ]
        )

        current_cx = set(
            cx_layers[
                layer_index
            ]
        )

        # Same directed CX in consecutive layers.
        if (
            previous_cx
            & current_cx
        ):
            return 0.0

        # No current CX immediately after both endpoints were I.
        for control, target in (
            current_cx
        ):
            prev_control = (
                _single_code03(
                    previous,
                    control,
                )
            )

            prev_target = (
                _single_code03(
                    previous,
                    target,
                )
            )

            if (
                prev_control == 0
                and prev_target == 0
            ):
                return 0.0

    # ------------------------------------------------------------
    # Three-layer T/CX/Tdg pruning, active from layer 3 onward.
    # ------------------------------------------------------------
    for layer_index in range(
        2,
        depth,
    ):
        two_back = actions[
            layer_index - 2
        ]

        previous_cx = cx_layers[
            layer_index - 1
        ]

        current = actions[
            layer_index
        ]

        for control, target in (
            previous_cx
        ):
            old_code = (
                _single_code03(
                    two_back,
                    control,
                )
            )

            current_code = (
                _single_code03(
                    current,
                    control,
                )
            )

            # Tdg -- CX(control,*) -- T
            if (
                old_code == 3
                and current_code == 2
            ):
                return 0.0

            # T -- CX(control,*) -- Tdg
            if (
                old_code == 2
                and current_code == 3
            ):
                return 0.0

    return 1.0

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


def compute_fixed_schedule_cyclic_fidelity(
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
        validate_source_schedule(
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

import cmath
import collections
import math
import numpy as np


# PRIVATE SOURCE-STYLE BOOLEAN BUILDER

class _SourceSynthesisBuilder05:
    """
    Private computational-basis weighted-Boolean builder.

    Variable numbering and clause ordering are implementation details only.
    The public Step-05 contract exposes only representation-independent
    fidelity/feasibility certificates.
    """

    def __init__(self):
        self.n = 3

        self.var = 0

        self.clauses = []

        self.weights = {}

        # selector variable -> (layer, gate_name, q0, q1)
        self.selector_meta = {}

        # Per-layer selector lookup used by symmetry-breaking clauses.
        self.layer_vars = {}

        self.layer = 0

        # Initial computational-basis variables.
        self.x = [
            self.new_var()
            for _ in range(
                self.n
            )
        ]

        self.x_init = list(
            self.x
        )

    def new_var(
        self,
        selector=False,
        meta=None,
    ):
        self.var += 1

        if selector:
            if meta is None:
                raise ValueError(
                    "selector variables require metadata"
                )

            self.selector_meta[
                self.var
            ] = meta

        return self.var

    def clause(
        self,
        literals,
    ):
        values = tuple(
            int(v)
            for v in literals
        )

        if len(values) < 1:
            raise ValueError(
                "empty clauses are not accepted by the builder"
            )

        self.clauses.append(
            values
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

        # Multiple phase operations may act on the same unchanged
        # computational-basis variable. Their literal weights multiply.
        if variable in self.weights:
            old_true, old_false = (
                self.weights[
                    variable
                ]
            )

            wt *= old_true
            wf *= old_false

        self.weights[
            variable
        ] = (
            wt,
            wf,
        )

    # TARGET GATE ENCODINGS

    def target_rx(
        self,
        qubit,
        theta,
    ):
        """
        Source computational-basis RX encoding.

        Auxiliary w satisfies

            w <-> (X <-> x)

        with

            W(w)     = cos(theta/2)
            W(not w) = -i sin(theta/2).
        """

        old = self.x[
            qubit
        ]

        X = self.new_var()

        w = self.new_var()

        # w <-> (X <-> old)
        self.clause(
            [
                X,
                w,
                old,
            ]
        )

        self.clause(
            [
                X,
                -w,
                -old,
            ]
        )

        self.clause(
            [
                -X,
                w,
                -old,
            ]
        )

        self.clause(
            [
                -X,
                -w,
                old,
            ]
        )

        self.weight(
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

    def target_rz(
        self,
        qubit,
        theta,
    ):
        """
        Source computational-basis RZ convention:

            diag(1, exp(i theta)).
        """

        self.weight(
            self.x[
                qubit
            ],
            cmath.exp(
                1j * theta
            ),
            1.0,
        )

    def target_cx(
        self,
        control,
        target,
    ):
        """
        Directed CX:

            new_target <-> control XOR old_target.

        The control wire is unchanged.
        """

        xc = self.x[
            control
        ]

        xt = self.x[
            target
        ]

        Xt = self.new_var()

        self.clause(
            [
                Xt,
                xc,
                -xt,
            ]
        )

        self.clause(
            [
                Xt,
                -xc,
                xt,
            ]
        )

        self.clause(
            [
                -Xt,
                xc,
                xt,
            ]
        )

        self.clause(
            [
                -Xt,
                -xc,
                -xt,
            ]
        )

        self.x[
            target
        ] = Xt

    def compile_target_adjoint(
        self,
        theta_rx0,
        theta_rx1,
        theta_rx2,
        theta_rz0,
        theta_rz2,
    ):
        """
        Encode V† for the chronological target

        RX0(a),
        CX0->1,
        RX1(b),
        CX1->2,
        RX2(c),
        RZ0(d),
        CX0->2,
        RZ2(e).

        Therefore V† is encoded in reverse order with negated
        rotation angles.
        """

        self.target_rz(
            2,
            -theta_rz2,
        )

        self.target_cx(
            0,
            2,
        )

        self.target_rz(
            0,
            -theta_rz0,
        )

        self.target_rx(
            2,
            -theta_rx2,
        )

        self.target_cx(
            1,
            2,
        )

        self.target_rx(
            1,
            -theta_rx1,
        )

        self.target_cx(
            0,
            1,
        )

        self.target_rx(
            0,
            -theta_rx0,
        )

    # SOURCE PARAMETRIC SYNTHESIS LAYERS

    def _exactly_one(
        self,
        variables,
    ):
        """
        Standard exactly-one CNF:
        one positive clause plus all pairwise exclusions.
        """

        self.clause(
            variables
        )

        for i in range(
            len(
                variables
            )
        ):
            for j in range(
                i + 1,
                len(
                    variables
                ),
            ):
                self.clause(
                    [
                        -variables[i],
                        -variables[j],
                    ]
                )

    def add_source_layer(
        self,
    ):
        """
        Add one source computational-basis parametric synthesis layer.

        Gate choices:
            I, H, T, Tdg, directed CX.

        Weighted auxiliaries:
            R  : -1
            U  : 1/sqrt(2)
            Wp : (1+i)/sqrt(2)
            Wn : (1-i)/sqrt(2)

        Negative literal weights are 1.
        """

        n = self.n

        self.layer += 1

        layer = self.layer

        old_x = list(
            self.x
        )

        # --------------------------------------------------------
        # State and weighted auxiliary variables.
        # --------------------------------------------------------

        X = [
            self.new_var()
            for _ in range(
                n
            )
        ]

        R = [
            self.new_var()
            for _ in range(
                n
            )
        ]

        U = [
            self.new_var()
            for _ in range(
                n
            )
        ]

        Wp = [
            self.new_var()
            for _ in range(
                n
            )
        ]

        Wn = [
            self.new_var()
            for _ in range(
                n
            )
        ]

        inv_sqrt2 = (
            1.0
            / math.sqrt(
                2.0
            )
        )

        for k in range(
            n
        ):
            self.weight(
                R[k],
                -1.0,
                1.0,
            )

            self.weight(
                U[k],
                inv_sqrt2,
                1.0,
            )

            self.weight(
                Wp[k],
                complex(
                    inv_sqrt2,
                    inv_sqrt2,
                ),
                1.0,
            )

            self.weight(
                Wn[k],
                complex(
                    inv_sqrt2,
                    -inv_sqrt2,
                ),
                1.0,
            )

        # --------------------------------------------------------
        # Single-qubit selector variables.
        # --------------------------------------------------------

        idg = [
            self.new_var(
                selector=True,
                meta=(
                    layer,
                    "id",
                    k,
                    -1,
                ),
            )
            for k in range(
                n
            )
        ]

        hg = [
            self.new_var(
                selector=True,
                meta=(
                    layer,
                    "h",
                    k,
                    -1,
                ),
            )
            for k in range(
                n
            )
        ]

        tg = [
            self.new_var(
                selector=True,
                meta=(
                    layer,
                    "t",
                    k,
                    -1,
                ),
            )
            for k in range(
                n
            )
        ]

        tdg = [
            self.new_var(
                selector=True,
                meta=(
                    layer,
                    "tdg",
                    k,
                    -1,
                ),
            )
            for k in range(
                n
            )
        ]

        # --------------------------------------------------------
        # Directed CX selector variables.
        # --------------------------------------------------------

        cg = [
            [
                None
                for _ in range(
                    n
                )
            ]
            for _ in range(
                n
            )
        ]

        for control in range(
            n
        ):
            for target in range(
                n
            ):
                if (
                    control
                    == target
                ):
                    continue

                cg[
                    control
                ][
                    target
                ] = self.new_var(
                    selector=True,
                    meta=(
                        layer,
                        "cx",
                        control,
                        target,
                    ),
                )

        self.layer_vars[
            layer
        ] = {
            "id": idg,
            "h": hg,
            "t": tg,
            "tdg": tdg,
            "cx": cg,
        }

        # --------------------------------------------------------
        # Per-qubit conditional gate semantics.
        # --------------------------------------------------------

        for k in range(
            n
        ):
            # IDENTITY

            # X == old_x when id selector is true.
            self.clause(
                [
                    X[k],
                    -idg[k],
                    -old_x[k],
                ]
            )

            self.clause(
                [
                    -X[k],
                    -idg[k],
                    old_x[k],
                ]
            )

            # No weighted auxiliary contribution.
            self.clause(
                [
                    -R[k],
                    -idg[k],
                ]
            )

            self.clause(
                [
                    -U[k],
                    -idg[k],
                ]
            )

            self.clause(
                [
                    -Wn[k],
                    -idg[k],
                ]
            )

            self.clause(
                [
                    -Wp[k],
                    -idg[k],
                ]
            )

            # HADAMARD
            #
            # R <-> (X and old_x)
            # U = true

            self.clause(
                [
                    -R[k],
                    X[k],
                    -hg[k],
                ]
            )

            self.clause(
                [
                    -R[k],
                    -hg[k],
                    old_x[k],
                ]
            )

            self.clause(
                [
                    R[k],
                    -X[k],
                    -hg[k],
                    -old_x[k],
                ]
            )

            self.clause(
                [
                    U[k],
                    -hg[k],
                ]
            )

            self.clause(
                [
                    -Wn[k],
                    -hg[k],
                ]
            )

            self.clause(
                [
                    -Wp[k],
                    -hg[k],
                ]
            )

            # T / T-DAGGER

            # State bit unchanged for both gates.
            self.clause(
                [
                    X[k],
                    -tdg[k],
                    -old_x[k],
                ]
            )

            self.clause(
                [
                    X[k],
                    -tg[k],
                    -old_x[k],
                ]
            )

            self.clause(
                [
                    -X[k],
                    -tdg[k],
                    old_x[k],
                ]
            )

            self.clause(
                [
                    -X[k],
                    -tg[k],
                    old_x[k],
                ]
            )

            self.clause(
                [
                    -R[k],
                    -tdg[k],
                ]
            )

            self.clause(
                [
                    -R[k],
                    -tg[k],
                ]
            )

            self.clause(
                [
                    -U[k],
                    -tdg[k],
                ]
            )

            self.clause(
                [
                    -U[k],
                    -tg[k],
                ]
            )

            # T does not use Wn.
            self.clause(
                [
                    -Wn[k],
                    -tg[k],
                ]
            )

            # Tdg does not use Wp.
            self.clause(
                [
                    -Wp[k],
                    -tdg[k],
                ]
            )

            # Wn follows old_x for Tdg.
            self.clause(
                [
                    Wn[k],
                    -tdg[k],
                    -old_x[k],
                ]
            )

            self.clause(
                [
                    -Wn[k],
                    -tdg[k],
                    old_x[k],
                ]
            )

            # Wp follows old_x for T.
            self.clause(
                [
                    Wp[k],
                    -tg[k],
                    -old_x[k],
                ]
            )

            self.clause(
                [
                    -Wp[k],
                    -tg[k],
                    old_x[k],
                ]
            )

            # DIRECTED CX WITH k AS CONTROL

            for target in range(
                n
            ):
                if (
                    target
                    == k
                ):
                    continue

                gate = (
                    cg[k][target]
                )

                # Control unchanged.
                self.clause(
                    [
                        X[k],
                        -gate,
                        -old_x[k],
                    ]
                )

                self.clause(
                    [
                        -X[k],
                        -gate,
                        old_x[k],
                    ]
                )

                # Target is control XOR target.
                self.clause(
                    [
                        X[target],
                        -gate,
                        old_x[k],
                        -old_x[target],
                    ]
                )

                self.clause(
                    [
                        X[target],
                        -gate,
                        -old_x[k],
                        old_x[target],
                    ]
                )

                self.clause(
                    [
                        -X[target],
                        -gate,
                        old_x[k],
                        old_x[target],
                    ]
                )

                self.clause(
                    [
                        -X[target],
                        -gate,
                        -old_x[k],
                        -old_x[target],
                    ]
                )

                # CX itself carries no auxiliary gate weight.
                for auxiliary in (
                    R,
                    U,
                    Wn,
                    Wp,
                ):
                    self.clause(
                        [
                            -auxiliary[k],
                            -gate,
                        ]
                    )

                    self.clause(
                        [
                            -auxiliary[
                                target
                            ],
                            -gate,
                        ]
                    )

            # EXACTLY ONE ACTION INVOLVING EACH QUBIT

            controllers = [
                idg[k],
                hg[k],
                tg[k],
                tdg[k],
            ]

            # Outgoing CX selectors.
            controllers += [
                cg[k][other]
                for other in range(
                    n
                )
                if (
                    other
                    != k
                )
            ]

            # Incoming CX selectors.
            controllers += [
                cg[other][k]
                for other in range(
                    n
                )
                if (
                    other
                    != k
                )
            ]

            self._exactly_one(
                controllers
            )

            # CROSS-LAYER SYMMETRY BREAKING

            if (
                layer
                >= 2
            ):
                previous = (
                    self.layer_vars[
                        layer - 1
                    ]
                )

                # No H immediately after H.
                self.clause(
                    [
                        -hg[k],
                        -previous[
                            "h"
                        ][k],
                    ]
                )

                # No T immediately after Tdg.
                self.clause(
                    [
                        -tg[k],
                        -previous[
                            "tdg"
                        ][k],
                    ]
                )

                # No Tdg immediately after T.
                self.clause(
                    [
                        -tdg[k],
                        -previous[
                            "t"
                        ][k],
                    ]
                )

                # Identity-padding canonicalization:
                #
                # previous I implies current I unless a current
                # directed CX is incident on qubit k.
                incident_current_cx = (
                    [
                        cg[k][other]
                        for other in range(
                            n
                        )
                        if (
                            other
                            != k
                        )
                    ]
                    +
                    [
                        cg[other][k]
                        for other in range(
                            n
                        )
                        if (
                            other
                            != k
                        )
                    ]
                )

                self.clause(
                    [
                        -previous[
                            "id"
                        ][k],
                        idg[k],
                    ]
                    + incident_current_cx
                )

            # ----------------------------------------------------
            # CX-specific cross-layer pruning.
            # ----------------------------------------------------

            for target in range(
                n
            ):
                if (
                    target
                    == k
                ):
                    continue

                current_cx = (
                    cg[k][target]
                )

                if (
                    layer
                    >= 2
                ):
                    previous = (
                        self.layer_vars[
                            layer - 1
                        ]
                    )

                    # Do not repeat exactly the same directed CX.
                    self.clause(
                        [
                            -current_cx,
                            -previous[
                                "cx"
                            ][k][target],
                        ]
                    )

                    # Do not introduce CX immediately after both
                    # endpoints were identities.
                    self.clause(
                        [
                            -current_cx,
                            -previous[
                                "id"
                            ][k],
                            -previous[
                                "id"
                            ][target],
                        ]
                    )

                if (
                    layer
                    >= 3
                ):
                    previous_1 = (
                        self.layer_vars[
                            layer - 1
                        ]
                    )

                    previous_2 = (
                        self.layer_vars[
                            layer - 2
                        ]
                    )

                    # Prune:
                    # Tdg(k) -- CX(k,target) -- T(k)
                    self.clause(
                        [
                            -previous_1[
                                "cx"
                            ][k][target],
                            -previous_2[
                                "tdg"
                            ][k],
                            -tg[k],
                        ]
                    )

                    # Prune:
                    # T(k) -- CX(k,target) -- Tdg(k)
                    self.clause(
                        [
                            -previous_1[
                                "cx"
                            ][k][target],
                            -previous_2[
                                "t"
                            ][k],
                            -tdg[k],
                        ]
                    )

        # New layer becomes current state.
        self.x = X

    # CYCLIC CLOSURE

    def close_cyclic(
        self,
    ):
        """
        Add equality between final and initial computational-basis bits.
        """

        for q in range(
            self.n
        ):
            final = (
                self.x[q]
            )

            initial = (
                self.x_init[q]
            )

            self.clause(
                [
                    final,
                    -initial,
                ]
            )

            self.clause(
                [
                    -final,
                    initial,
                ]
            )


# BOOLEAN-SOLVER HELPERS

def _canonical_clauses05(
    clauses,
):
    clean = []

    for clause in clauses:
        literals = set(
            clause
        )

        # Remove tautological clauses.
        if any(
            -literal
            in literals
            for literal
            in literals
        ):
            continue

        clean.append(
            tuple(
                sorted(
                    literals,
                    key=lambda value: (
                        abs(
                            value
                        ),
                        value < 0,
                    ),
                )
            )
        )

    return tuple(
        sorted(
            clean,
            key=lambda clause: (
                len(
                    clause
                ),
                clause,
            ),
        )
    )


def _bcp05(
    clauses,
):
    """
    Boolean constraint propagation.

    Returns
    -------
    simplified_clauses, propagated_assignment, unsat
    """

    work = [
        set(
            clause
        )
        for clause in clauses
    ]

    assignment = {}

    while True:
        if any(
            len(
                clause
            )
            == 0
            for clause
            in work
        ):
            return (
                (),
                assignment,
                True,
            )

        unit = None

        for clause in work:
            if (
                len(
                    clause
                )
                == 1
            ):
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
                # Clause satisfied.
                continue

            if (
                -unit
                in clause
            ):
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
        _canonical_clauses05(
            work
        ),
        assignment,
        False,
    )


def _variables05(
    clauses,
):
    return {
        abs(
            literal
        )
        for clause
        in clauses
        for literal
        in clause
    }


def _components05(
    clauses,
):
    """
    Connected components of the variable-clause incidence graph.
    """

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

            for literal in (
                clauses[index]
            ):
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
                for index
                in indices
            )
        )

    return components


# THRESHOLD EXCEPTION

class _ThresholdReached05(
    Exception
):
    """
    Internal early-exit signal carrying an actual feasible selector witness.
    """

    def __init__(
        self,
        z,
        assignment,
    ):
        super().__init__(
            "fidelity threshold reached"
        )

        self.z = complex(
            z
        )

        self.assignment = dict(
            assignment
        )


# D4MAX-STYLE OPTIMIZER

class _D4Style05:
    """
    d4Max-style recursive maximum weighted-model-counting solver.

    X = synthesis selector / optimization variables.
    Y = remaining weighted-counting variables.

    The source computational-basis objective is complex. Branches are
    therefore compared by |z|, because Jamiołkowski fidelity is monotonic in
    |z|^2.
    """

    def __init__(
        self,
        weights,
        selector_meta,
        n_qubits,
        threshold,
    ):
        self.weights = dict(
            weights
        )

        self.meta = dict(
            selector_meta
        )

        self.n = int(
            n_qubits
        )

        self.threshold = float(
            threshold
        )

        self.total_selectors = set(
            self.meta
        )

        self.wmc_cache = {}

        self.max_cache = {}

        # Deterministic search preference only.
        # It is not part of the public scientific result.
        self.gate_priority = {
            "h": 0,
            "t": 1,
            "cx": 2,
            "tdg": 3,
            "id": 4,
        }

    def _weight(
        self,
        variable,
        value,
    ):
        wt, wf = (
            self.weights.get(
                variable,
                (
                    1.0 + 0.0j,
                    1.0 + 0.0j,
                ),
            )
        )

        return (
            wt
            if value
            else wf
        )

    def _branch_key(
        self,
        variable,
    ):
        (
            layer,
            gate,
            q0,
            q1,
        ) = self.meta[
            variable
        ]

        return (
            layer,
            self.gate_priority[
                gate
            ],
            q0,
            q1,
            variable,
        )

    def _check_threshold(
        self,
        total_z,
        path,
        enabled,
    ):
        if not enabled:
            return

        # Only raise once a complete selector assignment is available.
        if (
            len(
                path
            )
            < len(
                self.total_selectors
            )
        ):
            return

        fidelity = (
            abs(
                total_z
            )
            ** 2
            / float(
                2
                ** (
                    2
                    * self.n
                )
            )
        )

        if (
            fidelity
            >= self.threshold
            - 1e-15
        ):
            raise _ThresholdReached05(
                total_z,
                path,
            )

    # WEIGHTED MODEL COUNTING BASE CASE

    def _wmc(
        self,
        clauses,
        counted,
    ):
        (
            simplified,
            propagated,
            unsat,
        ) = _bcp05(
            clauses
        )

        if unsat:
            return 0j

        counted = set(
            counted
        )

        factor = (
            1.0
            + 0.0j
        )

        # --------------------------------------------------------
        # Propagated counted variables.
        # --------------------------------------------------------

        for (
            variable,
            value,
        ) in propagated.items():
            if (
                variable
                in counted
            ):
                factor *= (
                    self._weight(
                        variable,
                        value,
                    )
                )

                counted.remove(
                    variable
                )

        active = (
            _variables05(
                simplified
            )
        )

        # --------------------------------------------------------
        # Free counted variables.
        #
        # A free counted variable contributes W(v)+W(not v).
        # --------------------------------------------------------

        for variable in list(
            counted
        ):
            if (
                variable
                not in active
            ):
                wt, wf = (
                    self.weights.get(
                        variable,
                        (
                            1.0
                            + 0.0j,
                            1.0
                            + 0.0j,
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

        if (
            key
            in self.wmc_cache
        ):
            return (
                factor
                * self.wmc_cache[
                    key
                ]
            )

        components = (
            _components05(
                simplified
            )
        )

        # --------------------------------------------------------
        # Component decomposition.
        # --------------------------------------------------------

        if (
            len(
                components
            )
            > 1
        ):
            result = (
                1.0
                + 0.0j
            )

            for component in (
                components
            ):
                component_variables = (
                    _variables05(
                        component
                    )
                )

                result *= (
                    self._wmc(
                        component,
                        counted
                        & component_variables,
                    )
                )

        # --------------------------------------------------------
        # Weighted Shannon expansion.
        # --------------------------------------------------------

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
                    1.0
                    + 0.0j
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
                    self.weights.get(
                        variable,
                        (
                            1.0
                            + 0.0j,
                            1.0
                            + 0.0j,
                        ),
                    )
                )

                result = (
                    wt
                    * self._wmc(
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
                    )
                    +
                    wf
                    * self._wmc(
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
                    )
                )

        self.wmc_cache[
            key
        ] = result

        return (
            factor
            * result
        )

    # MAXIMUM WEIGHTED-MODEL-COUNTING RECURSION

    def solve(
        self,
        clauses,
        X,
        Y,
        prefix=1.0 + 0.0j,
        path=None,
        threshold_enabled=True,
    ):
        if path is None:
            path = {}

        (
            simplified,
            propagated,
            unsat,
        ) = _bcp05(
            clauses
        )

        if unsat:
            return (
                0j,
                {},
            )

        X = set(
            X
        )

        Y = set(
            Y
        )

        factor = (
            1.0
            + 0.0j
        )

        fixed = {}

        current_path = dict(
            path
        )

        # --------------------------------------------------------
        # Account for propagated variables.
        # --------------------------------------------------------

        for (
            variable,
            value,
        ) in propagated.items():
            if variable in X:
                factor *= (
                    self._weight(
                        variable,
                        value,
                    )
                )

                X.remove(
                    variable
                )

                fixed[
                    variable
                ] = value

                current_path[
                    variable
                ] = value

            elif (
                variable
                in Y
            ):
                factor *= (
                    self._weight(
                        variable,
                        value,
                    )
                )

                Y.remove(
                    variable
                )

        active = (
            _variables05(
                simplified
            )
        )

        # --------------------------------------------------------
        # Free optimization variables.
        #
        # Selector weights are unit weights in this synthesis
        # construction. Choose true on a magnitude tie to maintain a
        # deterministic complete witness.
        # --------------------------------------------------------

        for variable in list(
            X
        ):
            if (
                variable
                not in active
            ):
                wt, wf = (
                    self.weights.get(
                        variable,
                        (
                            1.0
                            + 0.0j,
                            1.0
                            + 0.0j,
                        ),
                    )
                )

                value = (
                    abs(
                        wt
                    )
                    >= abs(
                        wf
                    )
                )

                factor *= (
                    wt
                    if value
                    else wf
                )

                X.remove(
                    variable
                )

                fixed[
                    variable
                ] = value

                current_path[
                    variable
                ] = value

        # --------------------------------------------------------
        # Free counted variables.
        # --------------------------------------------------------

        for variable in list(
            Y
        ):
            if (
                variable
                not in active
            ):
                wt, wf = (
                    self.weights.get(
                        variable,
                        (
                            1.0
                            + 0.0j,
                            1.0
                            + 0.0j,
                        ),
                    )
                )

                factor *= (
                    wt + wf
                )

                Y.remove(
                    variable
                )

        # --------------------------------------------------------
        # Formula satisfied.
        # --------------------------------------------------------

        if not simplified:
            total_z = (
                prefix
                * factor
            )

            self._check_threshold(
                total_z,
                current_path,
                threshold_enabled,
            )

            return (
                factor,
                fixed,
            )

        key = (
            simplified,
            tuple(
                sorted(
                    X
                )
            ),
            tuple(
                sorted(
                    Y
                )
            ),
        )

        # --------------------------------------------------------
        # Cached residual formula.
        # --------------------------------------------------------

        if (
            key
            in self.max_cache
        ):
            (
                core_z,
                core_assignment,
            ) = (
                self.max_cache[
                    key
                ]
            )

            merged = dict(
                fixed
            )

            merged.update(
                core_assignment
            )

            full_path = dict(
                current_path
            )

            full_path.update(
                core_assignment
            )

            total_z = (
                prefix
                * factor
                * core_z
            )

            self._check_threshold(
                total_z,
                full_path,
                threshold_enabled,
            )

            return (
                factor
                * core_z,
                merged,
            )

        active_X = (
            _variables05(
                simplified
            )
            & X
        )

        # --------------------------------------------------------
        # No optimization variable remains:
        # weighted-model-counting base case.
        # --------------------------------------------------------

        if not active_X:
            core_z = (
                self._wmc(
                    simplified,
                    Y,
                )
            )

            core_assignment = {}

        else:
            components = (
                _components05(
                    simplified
                )
            )

            # ----------------------------------------------------
            # Independent connected components.
            #
            # Threshold checking is disabled inside individual
            # components because only the product corresponds to the
            # full cyclic count.
            # ----------------------------------------------------

            if (
                len(
                    components
                )
                > 1
            ):
                core_z = (
                    1.0
                    + 0.0j
                )

                core_assignment = {}

                for component in (
                    components
                ):
                    component_variables = (
                        _variables05(
                            component
                        )
                    )

                    (
                        component_z,
                        component_assignment,
                    ) = self.solve(
                        component,
                        X
                        & component_variables,
                        Y
                        & component_variables,
                        prefix=(
                            1.0
                            + 0.0j
                        ),
                        path={},
                        threshold_enabled=False,
                    )

                    core_z *= (
                        component_z
                    )

                    core_assignment.update(
                        component_assignment
                    )

            # ----------------------------------------------------
            # Branch on one selector / optimization variable.
            # ----------------------------------------------------

            else:
                variable = min(
                    active_X,
                    key=(
                        self._branch_key
                    ),
                )

                # Positive branch first.
                (
                    z_true,
                    assignment_true,
                ) = self.solve(
                    simplified
                    + (
                        (
                            variable,
                        ),
                    ),
                    X,
                    Y,
                    prefix=(
                        prefix
                        * factor
                    ),
                    path=current_path,
                    threshold_enabled=(
                        threshold_enabled
                    ),
                )

                (
                    z_false,
                    assignment_false,
                ) = self.solve(
                    simplified
                    + (
                        (
                            -variable,
                        ),
                    ),
                    X,
                    Y,
                    prefix=(
                        prefix
                        * factor
                    ),
                    path=current_path,
                    threshold_enabled=(
                        threshold_enabled
                    ),
                )

                # Fidelity is monotonic in |z|^2.
                if (
                    abs(
                        z_true
                    )
                    >= abs(
                        z_false
                    )
                    - 1e-15
                ):
                    core_z = (
                        z_true
                    )

                    core_assignment = (
                        assignment_true
                    )

                else:
                    core_z = (
                        z_false
                    )

                    core_assignment = (
                        assignment_false
                    )

        self.max_cache[
            key
        ] = (
            core_z,
            core_assignment,
        )

        merged = dict(
            fixed
        )

        merged.update(
            core_assignment
        )

        full_path = dict(
            current_path
        )

        full_path.update(
            core_assignment
        )

        total_z = (
            prefix
            * factor
            * core_z
        )

        self._check_threshold(
            total_z,
            full_path,
            threshold_enabled,
        )

        return (
            factor
            * core_z,
            merged,
        )


# PRIVATE SEMANTIC WITNESS CONVERSION

def _private_assignment_to_schedule05(
    selector_meta,
    assignment,
    depth,
):
    """
    Convert the private Boolean selector assignment to the public semantic
    schedule representation.

    Output shape:
        (depth, 3, 3)

    Each slot:
        [-1,-1,-1] = unused
        [0,q,-1]    = I(q)
        [1,q,-1]    = H(q)
        [2,q,-1]    = T(q)
        [3,q,-1]    = Tdg(q)
        [4,c,t]     = CX(c->t)
    """

    gate_code = {
        "id": 0,
        "h": 1,
        "t": 2,
        "tdg": 3,
        "cx": 4,
    }

    layers = {
        layer: []
        for layer in range(
            1,
            depth + 1,
        )
    }

    for variable in sorted(
        selector_meta
    ):
        if not assignment.get(
            variable,
            False,
        ):
            continue

        (
            layer,
            name,
            q0,
            q1,
        ) = selector_meta[
            variable
        ]

        if name == "cx":
            row = [
                4,
                q0,
                q1,
            ]

        else:
            row = [
                gate_code[
                    name
                ],
                q0,
                -1,
            ]

        layers[
            layer
        ].append(
            row
        )

    schedule = np.full(
        (
            depth,
            3,
            3,
        ),
        -1.0,
        dtype=float,
    )

    for layer in range(
        1,
        depth + 1,
    ):
        operations = (
            layers[
                layer
            ]
        )

        # Deterministic semantic display order.
        # The ordering inside a legal parallel layer has no physical
        # significance because its operations act on disjoint qubits.
        operations.sort(
            key=lambda row: (
                row[0] == 4,
                row[1],
                row[2],
                row[0],
            )
        )

        if (
            len(
                operations
            )
            not in (
                2,
                3,
            )
        ):
            raise ValueError(
                "selector assignment does not describe a legal three-qubit layer"
            )

        for (
            index,
            row,
        ) in enumerate(
            operations
        ):
            schedule[
                layer - 1,
                index,
                :,
            ] = row

    return schedule


# PRIVATE FULL FIXED-DEPTH OPTIMIZER WITH ACTUAL WITNESS

def _private_optimize_fixed_depth_with_witness05(
    theta_rx0,
    theta_rx1,
    theta_rx2,
    theta_rz0,
    theta_rz2,
    depth,
    threshold,
):
    """
    Run the complete private fixed-depth paper-faithful Boolean optimization.

    Returns
    -------
    tuple
        (
            threshold_met,
            actual_fidelity,
            complex_count,
            private_selector_assignment,
            semantic_schedule,
        )
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
            "rotation angles must be finite"
        )

    if (
        isinstance(
            depth,
            (
                bool,
                np.bool_,
            ),
        )
        or not isinstance(
            depth,
            (
                int,
                np.integer,
            ),
        )
        or not (
            1
            <= int(
                depth
            )
            <= 4
        )
    ):
        raise ValueError(
            "depth must be an integer from 1 through 4"
        )

    try:
        threshold = float(
            threshold
        )

    except Exception as exc:
        raise ValueError(
            "threshold must be finite and in (0,1]"
        ) from exc

    if (
        not np.isfinite(
            threshold
        )
        or threshold <= 0.0
        or threshold > 1.0
    ):
        raise ValueError(
            "threshold must be finite and in (0,1]"
        )

    builder = (
        _SourceSynthesisBuilder05()
    )

    # ------------------------------------------------------------
    # Source target-adjoint encoding.
    # ------------------------------------------------------------

    builder.compile_target_adjoint(
        theta_rx0,
        theta_rx1,
        theta_rx2,
        theta_rz0,
        theta_rz2,
    )

    # ------------------------------------------------------------
    # Source parametric synthesis layers.
    # ------------------------------------------------------------

    for _ in range(
        int(
            depth
        )
    ):
        builder.add_source_layer()

    # ------------------------------------------------------------
    # Cyclic trace closure.
    # ------------------------------------------------------------

    builder.close_cyclic()

    selector_variables = set(
        builder.selector_meta
    )

    counted_variables = (
        set(
            range(
                1,
                builder.var + 1,
            )
        )
        - selector_variables
    )

    solver = _D4Style05(
        builder.weights,
        builder.selector_meta,
        builder.n,
        threshold,
    )

    threshold_met = False

    try:
        (
            z,
            assignment,
        ) = solver.solve(
            tuple(
                builder.clauses
            ),
            selector_variables,
            counted_variables,
        )

    except _ThresholdReached05 as hit:
        threshold_met = True

        z = (
            hit.z
        )

        assignment = (
            hit.assignment
        )

    actual_fidelity = (
        abs(
            z
        )
        ** 2
        / 64.0
    )

    schedule = (
        _private_assignment_to_schedule05(
            builder.selector_meta,
            assignment,
            int(
                depth
            ),
        )
    )

    return (
        bool(
            threshold_met
        ),
        float(
            actual_fidelity
        ),
        complex(
            z
        ),
        dict(
            assignment
        ),
        schedule,
    )


# STEP-05 ORACLE FUNCTION

def optimize_fixed_depth_d4max(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
    depth: int,
    threshold: float,
) -> np.ndarray:
    """
    Representation-independent Step-05 oracle.

    Returns
    -------
    np.ndarray
        [threshold_met, certificate]

        Rejected depth:
            [0, exact_global_optimum]

        Feasible depth:
            [1, requested_threshold]

        The actual first-found witness fidelity remains private because it
        may depend on search order even though the existence certificate does
        not.
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
            "rotation angles must be finite"
        )

    if (
        isinstance(
            depth,
            (
                bool,
                np.bool_,
            ),
        )
        or not isinstance(
            depth,
            (
                int,
                np.integer,
            ),
        )
        or not (
            1
            <= int(
                depth
            )
            <= 4
        )
    ):
        raise ValueError(
            "depth must be an integer from 1 through 4"
        )

    try:
        threshold = float(
            threshold
        )

    except Exception as exc:
        raise ValueError(
            "threshold must be finite and in (0,1]"
        ) from exc

    if (
        not np.isfinite(
            threshold
        )
        or threshold <= 0.0
        or threshold > 1.0
    ):
        raise ValueError(
            "threshold must be finite and in (0,1]"
        )

    (
        threshold_met,
        actual_fidelity,
        _,
        _,
        _,
    ) = (
        _private_optimize_fixed_depth_with_witness05(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
            int(
                depth
            ),
            threshold,
        )
    )

    if threshold_met:
        # Representation-independent feasibility certificate.
        return np.asarray(
            [
                1.0,
                threshold,
            ],
            dtype=float,
        )

    # If threshold was not reached, the d4Max recursion was exhaustive,
    # so actual_fidelity is the fixed-depth global optimum.
    return np.asarray(
        [
            0.0,
            actual_fidelity,
        ],
        dtype=float,
    )

import numpy as np


def search_source_depths_d4max(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
    epsilon: float,
    max_depth: int,
) -> np.ndarray:
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

    if not np.all(np.isfinite(angles)):
        raise ValueError(
            "rotation angles must be finite"
        )

    try:
        epsilon = float(epsilon)
    except Exception as exc:
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        ) from exc

    if (
        not np.isfinite(epsilon)
        or epsilon < 0.0
        or epsilon >= 1.0
    ):
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        )

    if (
        isinstance(max_depth, (bool, np.bool_))
        or not isinstance(max_depth, (int, np.integer))
        or int(max_depth) not in {1, 2, 3, 4}
    ):
        raise ValueError(
            "max_depth must be an integer in {1,2,3,4}"
        )

    threshold = 1.0 - epsilon

    rows = []

    for depth in range(
        1,
        int(max_depth) + 1,
    ):
        result = (
            optimize_fixed_depth_d4max(
                theta_rx0,
                theta_rx1,
                theta_rx2,
                theta_rz0,
                theta_rz2,
                depth,
                threshold,
            )
        )

        result = np.asarray(
            result,
            dtype=float,
        )

        if (
            result.shape != (2,)
            or not np.all(np.isfinite(result))
        ):
            raise ValueError(
                "fixed-depth optimizer returned a malformed result"
            )

        flag = float(result[0])
        certificate = float(result[1])

        if flag not in (0.0, 1.0):
            raise ValueError(
                "fixed-depth threshold flag must be exactly 0 or 1"
            )

        if (
            certificate < 0.0
            or certificate > 1.0
        ):
            raise ValueError(
                "fixed-depth certificate must lie in [0,1]"
            )

        rows.append(
            [
                float(depth),
                flag,
                certificate,
            ]
        )

        if flag == 1.0:
            return np.asarray(
                rows,
                dtype=float,
            )

    raise ValueError(
        "no searched depth reaches the requested threshold"
    )

import numpy as np


def select_first_feasible_depth(
    depth_results: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(
        depth_results,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.shape[0] < 1
        or values.shape[0] > 4
    ):
        raise ValueError(
            "depth_results must have shape (m,3) with 1 <= m <= 4"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "depth_results must be finite"
        )

    try:
        epsilon = float(epsilon)
    except Exception as exc:
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        ) from exc

    if (
        not np.isfinite(epsilon)
        or epsilon < 0.0
        or epsilon >= 1.0
    ):
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        )

    threshold = 1.0 - epsilon

    # Validate the complete table before selecting a feasible row.
    for index, row in enumerate(values):
        expected_depth = float(
            index + 1
        )

        depth = float(row[0])
        flag = float(row[1])
        certificate = float(row[2])

        if depth != expected_depth:
            raise ValueError(
                "depth rows must be consecutive integers starting at 1"
            )

        # Review-3 repair:
        # do this BEFORE the feasible branch.
        if flag not in (0.0, 1.0):
            raise ValueError(
                "threshold_met must be exactly 0 or 1"
            )

        if (
            certificate < 0.0
            or certificate > 1.0
        ):
            raise ValueError(
                "certificate must lie in [0,1]"
            )

        if (
            flag == 0.0
            and certificate
            >= threshold - 1e-12
        ):
            raise ValueError(
                "a rejected depth must have global optimum below threshold"
            )

        if (
            flag == 1.0
            and not np.isclose(
                certificate,
                threshold,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "a feasible public certificate must equal the requested threshold"
            )

    rejected = []

    for row in values:
        depth = int(row[0])
        flag = int(row[1])
        certificate = float(row[2])

        if flag == 1:
            return np.asarray(
                [
                    float(depth)
                ]
                + rejected,
                dtype=float,
            )

        rejected.append(
            certificate
        )

    raise ValueError(
        "no feasible depth is present"
    )

import cmath
import math
import numpy as np


def run_paper_faithful_quokka_synthesis(
    epsilon: float = 0.10,
    max_depth: int = 4,
    theta_rx0: float = 0.8453,
    theta_rx1: float = 0.1729,
    theta_rx2: float = 0.6331,
    theta_rz0: float = 0.4217,
    theta_rz2: float = 0.2876,
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

    if not np.all(np.isfinite(angles)):
        raise ValueError(
            "rotation angles must be finite"
        )

    if (
        isinstance(max_depth, (bool, np.bool_))
        or not isinstance(max_depth, (int, np.integer))
        or int(max_depth) not in {1, 2, 3, 4}
    ):
        raise ValueError(
            "max_depth must be an integer in {1,2,3,4}"
        )

    try:
        epsilon = float(epsilon)
    except Exception as exc:
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        ) from exc

    if (
        not np.isfinite(epsilon)
        or epsilon < 0.0
        or epsilon >= 1.0
    ):
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        )

    I = np.eye(
        2,
        dtype=complex,
    )

    H = (
        np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=complex,
        )
        / math.sqrt(2.0)
    )

    T = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                1j * math.pi / 4.0
            ),
        ]
    )

    Tdg = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                -1j * math.pi / 4.0
            ),
        ]
    )

    def rx(theta):
        c = math.cos(
            theta / 2.0
        )

        s = (
            -1j
            * math.sin(
                theta / 2.0
            )
        )

        return np.array(
            [
                [c, s],
                [s, c],
            ],
            dtype=complex,
        )

    def rz(theta):
        return np.diag(
            [
                cmath.exp(
                    -1j * theta / 2.0
                ),
                cmath.exp(
                    +1j * theta / 2.0
                ),
            ]
        )

    def op1(
        gate,
        qubit,
    ):
        factors = [
            I,
            I,
            I,
        ]

        factors[
            qubit
        ] = gate

        return np.kron(
            np.kron(
                factors[0],
                factors[1],
            ),
            factors[2],
        )

    def cx3(
        control,
        target,
    ):
        matrix = np.zeros(
            (8, 8),
            dtype=complex,
        )

        for basis in range(8):
            bits = [
                (basis >> 2) & 1,
                (basis >> 1) & 1,
                basis & 1,
            ]

            out = list(
                bits
            )

            if bits[
                control
            ]:
                out[
                    target
                ] ^= 1

            output_index = (
                4 * out[0]
                + 2 * out[1]
                + out[2]
            )

            matrix[
                output_index,
                basis,
            ] = 1.0

        return matrix

    # STEP 01: VALUE-SENSITIVE GATE-SEMANTIC CHECK

    gate_check = (
        reconstruct_source_gate_encodings(
            theta_rx0,
            theta_rz0,
        )
    )

    rz_source = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                1j * theta_rz0
            ),
        ]
    )

    expected_gates = np.stack(
        [
            np.kron(
                H,
                I,
            ),
            np.kron(
                T,
                I,
            ),
            np.kron(
                Tdg,
                I,
            ),
            np.kron(
                rx(theta_rx0),
                I,
            ),
            np.kron(
                rz_source,
                I,
            ),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=complex,
            ),
        ],
        axis=0,
    )

    expected_gate_check = np.stack(
        [
            expected_gates.real,
            expected_gates.imag,
        ],
        axis=-1,
    ).astype(float)

    gate_check = np.asarray(
        gate_check,
        dtype=float,
    )

    if (
        gate_check.shape
        != expected_gate_check.shape
    ):
        raise ValueError(
            "source gate semantic check returned an invalid shape"
        )

    if not np.allclose(
        gate_check,
        expected_gate_check,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "source gate semantic check disagrees with independent matrices"
        )

    # INDEPENDENT PHYSICAL TARGET

    physical_target = np.eye(
        8,
        dtype=complex,
    )

    target_gates = [
        op1(
            rx(theta_rx0),
            0,
        ),
        cx3(
            0,
            1,
        ),
        op1(
            rx(theta_rx1),
            1,
        ),
        cx3(
            1,
            2,
        ),
        op1(
            rx(theta_rx2),
            2,
        ),
        op1(
            rz(theta_rz0),
            0,
        ),
        cx3(
            0,
            2,
        ),
        op1(
            rz(theta_rz2),
            2,
        ),
    ]

    for gate in target_gates:
        physical_target = (
            gate
            @ physical_target
        )

    # STEP 02: VALUE-SENSITIVE IDENTITY-FIDELITY CHECK

    identity_fidelity = (
        compute_target_identity_cyclic_fidelity(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        )
    )

    expected_identity_fidelity = (
        abs(
            np.trace(
                physical_target
            )
        )
        ** 2
        / 64.0
    )

    if not np.isclose(
        identity_fidelity,
        expected_identity_fidelity,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "target cyclic fidelity disagrees with the independent target matrix"
        )

    # STEP 06: INCREASING-DEPTH SEARCH

    depth_results = (
        search_source_depths_d4max(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
            epsilon,
            int(max_depth),
        )
    )

    # STEP 07: MINIMUM-DEPTH SELECTION

    selected = (
        select_first_feasible_depth(
            depth_results,
            epsilon,
        )
    )

    selected_depth = int(
        round(
            float(
                selected[0]
            )
        )
    )

    threshold = (
        1.0 - epsilon
    )

    # STEP 05: RECOVER AN ACTUAL SOLVER WITNESS

    (
        threshold_met,
        solver_fidelity,
        _,
        _,
        schedule,
    ) = (
        _private_optimize_fixed_depth_with_witness05(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
            selected_depth,
            threshold,
        )
    )

    if not threshold_met:
        raise ValueError(
            "selected depth did not produce a threshold witness"
        )

    # STEP 03: SEMANTIC SCHEDULE VALIDATION

    if (
        validate_source_schedule(
            schedule
        )
        != 1.0
    ):
        raise ValueError(
            "solver witness failed source schedule validation"
        )

    # STEP 04: FIXED-SCHEDULE BOOLEAN FIDELITY

    boolean_fidelity = (
        compute_fixed_schedule_cyclic_fidelity(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
            schedule,
        )
    )

    if not np.isclose(
        boolean_fidelity,
        solver_fidelity,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "fixed-schedule Boolean WMC disagrees with fixed-depth witness"
        )

    if (
        boolean_fidelity
        < threshold
        - 1e-12
    ):
        raise ValueError(
            "selected witness is below the requested threshold"
        )

    # FINAL INDEPENDENT UNITARY CROSS-CHECK

    single_gate = {
        0: I,
        1: H,
        2: T,
        3: Tdg,
    }

    candidate = np.eye(
        8,
        dtype=complex,
    )

    normalized_schedule = np.asarray(
        schedule,
        dtype=int,
    )

    for layer in normalized_schedule:
        layer_matrix = np.eye(
            8,
            dtype=complex,
        )

        for (
            gate_code,
            q0,
            q1,
        ) in layer:
            gate_code = int(
                gate_code
            )

            q0 = int(
                q0
            )

            q1 = int(
                q1
            )

            if gate_code == -1:
                continue

            if gate_code in (
                0,
                1,
                2,
                3,
            ):
                layer_matrix = (
                    op1(
                        single_gate[
                            gate_code
                        ],
                        q0,
                    )
                    @ layer_matrix
                )

            elif gate_code == 4:
                layer_matrix = (
                    cx3(
                        q0,
                        q1,
                    )
                    @ layer_matrix
                )

            else:
                raise ValueError(
                    "solver witness contains an invalid gate code"
                )

        candidate = (
            layer_matrix
            @ candidate
        )

    matrix_fidelity = (
        abs(
            np.trace(
                candidate
                @ physical_target.conj().T
            )
        )
        ** 2
        / 64.0
    )

    if not np.isclose(
        matrix_fidelity,
        boolean_fidelity,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "physical matrix check disagrees with Boolean fidelity"
        )

    return float(
        selected_depth
    )
SCICODE_GOLD_EOF
