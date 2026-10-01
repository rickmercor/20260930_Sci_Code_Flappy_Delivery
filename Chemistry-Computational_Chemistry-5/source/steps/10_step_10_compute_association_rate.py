"""
Step name: 10_compute_association_rate
Step description: Compose every earlier step to obtain the association rate constant of the tagged adsorbed hydrogen atom in inverse picoseconds.

Orchestrator: yes - simulates the trajectory (simulate_surface_trajectory), featurises it (build_local_descriptors), labels its endpoints (label_reaction_endpoints), reduces it to slow coordinates (project_dynamical_components), discretises the transition region (assign_intermediate_microstates), tabulates the endpoint entry times (compute_stopping_times), assembles and solves the projected committor problem (build_galerkin_system, solve_committor) and integrates the reactive current (estimate_reactive_flux), consuming each output rather than reimplementing any step.
Step scientific background: A reaction rate constant is the reactive current normalised by the population of the state the reaction starts from, measured through the backward commitment probability rather than through a geometric state count. Reported per unit time, it is the quantity that can be compared with transition state theory estimates for the same surface.

Returns
-------
float, the association rate constant in inverse picoseconds as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_association_rate(n_frames: int = 200000, seed: int = 20260212,
                             dt_ps: float = 0.01, lag_frames: int = 20,
                             n_components: int = 5, n_microstates: int = 64,
                             cluster_seed: int = 0) -> float:
    """Return the rate constant for atomic hydrogen combining into a molecule.

    Simulate the tagged hydrogen atom for ``n_frames`` frames from ``seed``,
    build its adsorbate-centred descriptors with the default radial basis,
    label its endpoint states with the default smoothing window and radii,
    project the descriptors onto ``n_components`` slow coordinates at a lag of
    ``lag_frames``, cluster the intermediate frames into ``n_microstates``
    microstates from ``cluster_seed``, tabulate the endpoint entry times,
    solve the projected committor problem at a lag of ``lag_frames`` and
    integrate the reactive current at the same lag.

    Conventions fixed by this step. The rate is the net reactive flux per
    frame divided by the trajectory average of the backward committor over
    every frame of the trajectory and by ``dt_ps``, so it is reported in
    inverse picoseconds. The defaults reproduce the problem statement.

    Parameters
    ----------
    n_frames : int
        Number of recorded frames, at least 2.
    seed : int
        Seed of the trajectory increments.
    dt_ps : float
        Frame spacing in picoseconds, positive.
    lag_frames : int
        Lag in frames used for the projection, the committor problem and the
        reactive current, at least 1 and smaller than ``n_frames``.
    n_components : int
        Number of slow coordinates retained, at least 1.
    n_microstates : int
        Number of intermediate microstates, at least 1.
    cluster_seed : int
        Seed of the microstate seeding generator.

    Returns
    -------
    float
        Association rate constant in inverse picoseconds.

    Raises
    ------
    ValueError
        If any argument violates the contract of the step that consumes it,
        or if the trajectory average of the backward committor is not
        positive.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_association_rate(n_frames: int = 200000, seed: int = 20260212,
                                     dt_ps: float = 0.01, lag_frames: int = 20,
                                     n_components: int = 5, n_microstates: int = 64,
                                     cluster_seed: int = 0) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    if not _is_positive_float(dt_ps):
        raise ValueError("dt_ps must be a positive finite float")

    trajectory = _oracle_simulate_surface_trajectory(n_frames, seed, dt_ps)
    descriptors = _oracle_build_local_descriptors(trajectory)
    labels = _oracle_label_reaction_endpoints(trajectory)
    components = _oracle_project_dynamical_components(descriptors, lag_frames, n_components)
    microstates = _oracle_assign_intermediate_microstates(components, labels,
                                                          n_microstates, cluster_seed)
    stopping = _oracle_compute_stopping_times(labels)
    system = _oracle_build_galerkin_system(labels, microstates, stopping,
                                           lag_frames, n_microstates)
    committor = _oracle_solve_committor(labels, microstates, system)
    flux = _oracle_estimate_reactive_flux(labels, committor, stopping, lag_frames)

    backward_mean = float(np.mean(1.0 - committor))
    if not backward_mean > 0.0:
        raise ValueError("the mean backward committor must be positive")
    return float(flux / (backward_mean * float(dt_ps)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    plain = "import numpy as np\n"
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": plain,
            "call": "compute_association_rate(40000, 5, n_microstates=8)",
            "gold_call": "_oracle_compute_association_rate(40000, 5, n_microstates=8)",
        },
        {
            "setup": plain,
            "call": "compute_association_rate(30000, 20260212, n_microstates=12, cluster_seed=4)",
            "gold_call": "_oracle_compute_association_rate(30000, 20260212, n_microstates=12, cluster_seed=4)",
        },
        {
            "setup": plain,
            "call": "compute_association_rate(20000, 9, lag_frames=8, n_components=3, n_microstates=8)",
            "gold_call": "_oracle_compute_association_rate(20000, 9, lag_frames=8, n_components=3, n_microstates=8)",
        },
        {
            "setup": plain,
            "call": "compute_association_rate(60000, 5, dt_ps=0.02, lag_frames=40, n_microstates=12)",
            "gold_call": "_oracle_compute_association_rate(60000, 5, dt_ps=0.02, lag_frames=40, n_microstates=12)",
        },
        {
            "setup": plain,
            "call": "float(compute_association_rate(80000, 7, n_components=2, n_microstates=16) * 100.0)",
            "gold_call": "float(_oracle_compute_association_rate(80000, 7, n_components=2, n_microstates=16) * 100.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_association_rate(20000, 9, dt_ps=0.0))",
            "gold_call": "_status(lambda: _oracle_compute_association_rate(20000, 9, dt_ps=0.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_association_rate(20000, 9, n_microstates=1000000))",
            "gold_call": "_status(lambda: _oracle_compute_association_rate(20000, 9, n_microstates=1000000))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_association_rate(20000, 9, n_components=500))",
            "gold_call": "_status(lambda: _oracle_compute_association_rate(20000, 9, n_components=500))",
        },
    ]
