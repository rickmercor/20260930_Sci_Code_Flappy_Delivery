"""
Run the complete three-qubit source synthesis pipeline.



The orchestrator uses the numerical results of the preceding subproblems,

including value-sensitive checks of the source gate-semantic tensor and the

target-versus-identity cyclic fidelity. It then runs the increasing-depth

fixed-depth optimization, selects the first feasible depth, recovers and

validates a source-generated witness, and performs an independent unitary

cross-check.



The final public result is the minimum feasible synthesis depth.

This is the final pipeline orchestrator.



epsilon must be finite and satisfy



0 <= epsilon < 1.



max_depth must be an integer in



{1,2,3,4}.



All five rotation angles must be finite.



The Step-01 gate-semantic tensor must be checked numerically against

independently constructed gate matrices; checking only its array shape is not

sufficient.



The Step-02 target-versus-identity fidelity must likewise agree with an

independently constructed physical target unitary; checking only that the

value is finite and lies in [0,1] is not sufficient.



After those value-sensitive checks, run the source depth search, select the

first feasible depth, validate an actual source-generated witness, and use

unitary multiplication only as a final independent physical cross-check.

Returns
-------
float: the minimum feasible synthesis depth as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_paper_faithful_quokka_synthesis(
    epsilon: float = 0.10,
    max_depth: int = 4,
    theta_rx0: float = 0.8453,
    theta_rx1: float = 0.1729,
    theta_rx2: float = 0.6331,
    theta_rz0: float = 0.4217,
    theta_rz2: float = 0.2876,
) -> float:
    """
    Run the complete three-qubit source synthesis benchmark.

    Parameters
    ----------
    epsilon : float
        Approximation tolerance. Must be finite and satisfy
        0 <= epsilon < 1.
    max_depth : int
        Largest synthesis depth to search. Must be exactly one of
        {1, 2, 3, 4}.
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
    minimum_depth : float
        Minimum feasible synthesis depth as a native Python float.

    Raises
    ------
    ValueError
        If epsilon is non-finite or does not satisfy
        0 <= epsilon < 1; if max_depth is not an integer in
        {1,2,3,4}; if any rotation angle is non-finite; if an
        upstream source-semantic result fails its independent
        numerical consistency check; if no searched depth is
        feasible; or if a recovered witness fails its semantic,
        Boolean-fidelity, or physical cross-check.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import cmath
import math
import numpy as np


def _oracle_run_paper_faithful_quokka_synthesis(
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

    # ============================================================
    # STEP 01: VALUE-SENSITIVE GATE-SEMANTIC CHECK
    # ============================================================

    gate_check = (
        _oracle_reconstruct_source_gate_encodings(
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

    # ============================================================
    # INDEPENDENT PHYSICAL TARGET
    # ============================================================

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

    # ============================================================
    # STEP 02: VALUE-SENSITIVE IDENTITY-FIDELITY CHECK
    # ============================================================

    identity_fidelity = (
        _oracle_compute_target_identity_cyclic_fidelity(
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

    # ============================================================
    # STEP 06: INCREASING-DEPTH SEARCH
    # ============================================================

    depth_results = (
        _oracle_search_source_depths_d4max(
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
            epsilon,
            int(max_depth),
        )
    )

    # ============================================================
    # STEP 07: MINIMUM-DEPTH SELECTION
    # ============================================================

    selected = (
        _oracle_select_first_feasible_depth(
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

    # ============================================================
    # STEP 05: RECOVER AN ACTUAL SOLVER WITNESS
    # ============================================================

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

    # ============================================================
    # STEP 03: SEMANTIC SCHEDULE VALIDATION
    # ============================================================

    if (
        _oracle_validate_source_schedule(
            schedule
        )
        != 1.0
    ):
        raise ValueError(
            "solver witness failed source schedule validation"
        )

    # ============================================================
    # STEP 04: FIXED-SCHEDULE BOOLEAN FIDELITY
    # ============================================================

    boolean_fidelity = (
        _oracle_compute_fixed_schedule_cyclic_fidelity(
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

    # ============================================================
    # FINAL INDEPENDENT UNITARY CROSS-CHECK
    # ============================================================

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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

epsilon = 0.30
max_depth = 4
a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
""",
            "call": "run_paper_faithful_quokka_synthesis(epsilon,max_depth,a,b,c,d,e)",
            "gold_call": "_oracle_run_paper_faithful_quokka_synthesis(epsilon,max_depth,a,b,c,d,e)",
        },
        {
            "setup": """import numpy as np

epsilon = 0.81
max_depth = 4
a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
""",
            "call": "run_paper_faithful_quokka_synthesis(epsilon,max_depth,a,b,c,d,e)",
            "gold_call": "_oracle_run_paper_faithful_quokka_synthesis(epsilon,max_depth,a,b,c,d,e)",
        },
        {
            "setup": """import numpy as np

epsilon = 0.10
max_depth = 0
a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876

def run_model():
    try:
        run_paper_faithful_quokka_synthesis(
            epsilon,max_depth,a,b,c,d,e
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_paper_faithful_quokka_synthesis(
            epsilon,max_depth,a,b,c,d,e
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
