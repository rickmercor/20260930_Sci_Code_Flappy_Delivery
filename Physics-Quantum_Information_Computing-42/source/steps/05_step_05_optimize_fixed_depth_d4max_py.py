"""
Solve one fixed-depth source synthesis optimization for the fixed three-qubit benchmark target. The benchmark topology is fixed; the inputs supply its five rotation angles, the synthesis depth, and the required fidelity threshold.

The fixed three-qubit target circuit used by this subproblem is, in chronological order,



RX_0(theta_rx0),

CX_{0->1},

RX_1(theta_rx1),

CX_{1->2},

RX_2(theta_rx2),

RZ_0(theta_rz0),

CX_{0->2},

RZ_2(theta_rz2).



Computational-basis ordering is |q0 q1 q2>, with q0 the most-significant qubit.



The five angle inputs are finite real numbers. The directed-CX locations and target gate ordering above are fixed and are not additional inputs.



The supported synthesis depth is an integer in {1,2,3,4}. Depth zero is not supported.



The fidelity threshold is a finite real number satisfying



0 < threshold <= 1.



Optimize the source fixed-depth synthesis problem at exactly the requested depth. The optimization must operate over the symbolic source selector construction rather than externally enumerating all complete candidate circuit strings.



The returned fixed-depth fidelity is the global optimum established by the source optimization at that depth, and the feasibility indicator must be consistent with the supplied threshold.

Returns
-------
np.ndarray of shape (2,): [threshold_met, certificate], where rejected depths return their exact global optimum and feasible depths return the requested threshold as a representation-independent existence certificate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


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
    Optimize one fixed synthesis depth for the fixed three-qubit target.

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
    depth : int
        Synthesis depth. Must be exactly one of {1, 2, 3, 4}.
    threshold : float
        Required fidelity threshold. Must be finite and satisfy
        0 < threshold <= 1.

    Returns
    -------
    result : np.ndarray
        The fixed-depth optimization result using the representation
        specified by this step's Expected Return Line. Its feasibility
        field must be Boolean-valued (0 or 1), and its fidelity
        certificate must lie in [0, 1].

    Raises
    ------
    ValueError
        If any rotation angle is not finite; if depth is not an integer
        in {1, 2, 3, 4}; or if threshold is not finite or does not
        satisfy 0 < threshold <= 1.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import cmath
import collections
import math
import numpy as np


# ================================================================
# PRIVATE SOURCE-STYLE BOOLEAN BUILDER
# ================================================================

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

    # ============================================================
    # TARGET GATE ENCODINGS
    # ============================================================

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

    # ============================================================
    # SOURCE PARAMETRIC SYNTHESIS LAYERS
    # ============================================================

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
            # ====================================================
            # IDENTITY
            # ====================================================

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

            # ====================================================
            # HADAMARD
            #
            # R <-> (X and old_x)
            # U = true
            # ====================================================

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

            # ====================================================
            # T / T-DAGGER
            # ====================================================

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

            # ====================================================
            # DIRECTED CX WITH k AS CONTROL
            # ====================================================

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

            # ====================================================
            # EXACTLY ONE ACTION INVOLVING EACH QUBIT
            # ====================================================

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

            # ====================================================
            # CROSS-LAYER SYMMETRY BREAKING
            # ====================================================

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

    # ============================================================
    # CYCLIC CLOSURE
    # ============================================================

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


# ================================================================
# BOOLEAN-SOLVER HELPERS
# ================================================================

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


# ================================================================
# THRESHOLD EXCEPTION
# ================================================================

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


# ================================================================
# D4MAX-STYLE OPTIMIZER
# ================================================================

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

    # ============================================================
    # WEIGHTED MODEL COUNTING BASE CASE
    # ============================================================

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

    # ============================================================
    # MAXIMUM WEIGHTED-MODEL-COUNTING RECURSION
    # ============================================================

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


# ================================================================
# PRIVATE SEMANTIC WITNESS CONVERSION
# ================================================================

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


# ================================================================
# PRIVATE FULL FIXED-DEPTH OPTIMIZER WITH ACTUAL WITNESS
# ================================================================

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


# ================================================================
# STEP-05 ORACLE FUNCTION
# ================================================================

def _oracle_optimize_fixed_depth_d4max(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
depth = 1
threshold = 0.90
""",
            "call": "optimize_fixed_depth_d4max(a,b,c,d,e,depth,threshold)",
            "gold_call": "_oracle_optimize_fixed_depth_d4max(a,b,c,d,e,depth,threshold)",
        },
        {
            "setup": """import numpy as np

a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
depth = 2
threshold = 0.90
""",
            "call": "optimize_fixed_depth_d4max(a,b,c,d,e,depth,threshold)",
            "gold_call": "_oracle_optimize_fixed_depth_d4max(a,b,c,d,e,depth,threshold)",
        },
        {
            "setup": """import numpy as np

a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
depth = 0
threshold = 0.90

def run_model():
    try:
        optimize_fixed_depth_d4max(
            a,b,c,d,e,depth,threshold
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_optimize_fixed_depth_d4max(
            a,b,c,d,e,depth,threshold
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
