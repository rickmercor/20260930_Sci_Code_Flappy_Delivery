"""
Validate a semantic three-qubit synthesis schedule against the source

computational-basis parametric-layer constraints.



The public input is a gate schedule expressed in terms of gate types and

qubit operands rather than private Boolean-variable numbers. This makes the

subproblem invariant to CNF variable allocation and clause ordering.



Each layer must cover each of q0, q1 and q2 exactly once. A one-qubit gate

covers one qubit, while a directed CX covers both its control and target.

Therefore a valid three-qubit layer consists either of three independent

one-qubit gates or one directed CX together with one one-qubit gate on the

remaining qubit.



The function also enforces the source cross-layer symmetry-breaking rules

that are active for depths through 4, including adjacent-H pruning,

T/T† cancellation pruning, identity-padding canonicalization, repeated-CX

pruning, CX-after-two-identities pruning, and the three-layer T/CX/T† rules.



The ordering of operation slots within a layer has no scientific meaning.

Determine whether the supplied semantic gate schedule satisfies the source

parametric-layer selector conditions and the source symmetry-breaking rules

that apply within the benchmark depth range.



Validity must depend only on gate identities and qubit operands, not on

private Boolean-variable numbering.

Returns
-------
float: 1.0 if the semantic schedule obeys the source selector-occupancy and depth<=4 symmetry-breaking constraints, otherwise 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def validate_source_schedule(
    schedule: np.ndarray,
) -> float:
    """
    Validate a semantic source synthesis schedule.

    Parameters
    ----------
    schedule : np.ndarray
        Shape (depth,3,3), where each operation slot is
        [gate_code, q0, q1].

        gate_code:
            -1 = unused padding slot
             0 = I
             1 = H
             2 = T
             3 = Tdg
             4 = directed CX

        For I/H/T/Tdg, q0 is the acted-on qubit and q1=-1.
        For CX, q0 is the control and q1 is the target.
        An unused slot is exactly [-1,-1,-1].

    Returns
    -------
    float
        1.0 if the schedule satisfies the source layer and symmetry
        constraints, otherwise 0.0.

    Raises
    ------
    ValueError
        If the array shape, depth, entries, gate codes, or operand syntax
        are malformed.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_validate_source_schedule(
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
            "call": "validate_source_schedule(schedule)",
            "gold_call": "_oracle_validate_source_schedule(schedule)",
        },
        {
            "setup": """import numpy as np

# H0 immediately followed by H0: source-invalid.
schedule = np.array([
    [
        [1,0,-1],
        [0,1,-1],
        [0,2,-1],
    ],
    [
        [1,0,-1],
        [0,1,-1],
        [0,2,-1],
    ],
], dtype=float)
""",
            "call": "validate_source_schedule(schedule)",
            "gold_call": "_oracle_validate_source_schedule(schedule)",
        },
        {
            "setup": """import numpy as np

schedule = np.array([
    [
        [4,0,0],
        [0,2,-1],
        [-1,-1,-1],
    ],
], dtype=float)

def run_model():
    try:
        validate_source_schedule(schedule)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_source_schedule(schedule)
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
