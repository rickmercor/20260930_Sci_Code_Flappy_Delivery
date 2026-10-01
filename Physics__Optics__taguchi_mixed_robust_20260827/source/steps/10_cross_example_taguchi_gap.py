"""
Resolve the joint design and its runner-up score gap by composing every earlier public subproblem.

The submitted orchestrator must call and combine all nine preceding public functions; its private gold path uses the corresponding oracle twins.

Returns
-------
float: runner-up joint score minus winning joint score.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cross_example_taguchi_gap(reduction_candidates: np.ndarray=None, surrogate_tensor: np.ndarray=None, closure_tolerances: np.ndarray=None, ddf_iterations: int=4, guiding_iterations: int=5) -> float:
    """Return the runner-up-minus-winner normalized closure score.

Parameters
----------
reduction_candidates : numpy.ndarray
    Three distinct rates in `(0,1)`; `None` selects the task instance.
surrogate_tensor : numpy.ndarray
    Shape `(2,6,6)`; span, polynomial-basis row, then four order
    residuals and two terminal residuals. `None` selects the task archive.
closure_tolerances : numpy.ndarray
    Five positive normalization scales. `None` selects the task scales.
ddf_iterations, guiding_iterations : int
    Positive modified-Taguchi iteration counts.

Returns
-------
gap : float
    Finite nonnegative difference between the two smallest joint scores.

Notes
-----
The submitted orchestrator must call and combine every earlier public subproblem
function; local reimplementations do not satisfy this integration contract.

Conventions
-----------
Use initialization-only intervals: no subsequent clipping. Use Step 7's exact-zero SNR and lowest-index tie rules. A finite surrogate tensor, including the zero tensor, is accepted for evaluation but does not guarantee a physically admissible final state. Raise ValueError if there is no positive dense-grid DDF order, if the selected DDF profile is nonpositive on a validation fold, if either final theoretical-target gain or launch factor is nonpositive or nonfinite, or if required diagnostics cannot be represented finitely. Inferred loss slopes may be signed. The default all-zero tensor with the default rates and iteration counts raises ValueError because deterministic recentering produces an inadmissible final theoretical state. No fallback clipping or gain floor is applied.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _default_surrogate_tensor() -> np.ndarray:
    b0 = np.array([
        [0.026,-0.021,0.017,-0.014,0.010,-0.008],
        [0.470,0.390,0.540,0.440,0.260,-0.200],
        [0.310,0.370,0.270,0.340,0.320,0.250],
        [0.130,-0.070,0.090,-0.050,-0.100,0.080],
        [0.070,0.060,0.080,0.050,0.040,0.010],
        [-0.030,-0.050,-0.020,-0.040,0.010,0.030],
    ], dtype=float)
    b1 = np.array([
        [0.034,-0.028,0.023,-0.017,0.014,-0.011],
        [0.560,0.460,0.630,0.500,0.310,-0.240],
        [0.360,0.440,0.320,0.400,0.380,0.300],
        [0.100,-0.090,0.120,-0.070,-0.120,0.100],
        [0.090,0.040,0.060,0.070,0.060,-0.010],
        [-0.050,-0.070,-0.040,-0.060,-0.010,0.050],
    ], dtype=float)
    return np.stack((b0, b1))

def _oracle_cross_example_taguchi_gap(reduction_candidates: np.ndarray = None,
                                      surrogate_tensor: np.ndarray = None,
                                      closure_tolerances: np.ndarray = None,
                                      ddf_iterations: int = 4,
                                      guiding_iterations: int = 5) -> float:
    if reduction_candidates is None:
        reduction_candidates = np.array([0.50, 0.65, 0.75], dtype=float)
    if surrogate_tensor is None:
        surrogate_tensor = _default_surrogate_tensor()
    if closure_tolerances is None:
        closure_tolerances = np.array([0.05, 0.24, 0.06, 0.002, 0.10], dtype=float)
    rates = np.asarray(reduction_candidates, dtype=float)
    tensor = np.asarray(surrogate_tensor, dtype=float)
    tau = np.asarray(closure_tolerances, dtype=float)
    if rates.shape != (3,) or tensor.shape != (2,6,6) or tau.shape != (5,):
        raise ValueError("Require three rates, a 2x6x6 archive, and five tolerances")
    if not np.isfinite(rates).all() or not np.isfinite(tensor).all() or not np.isfinite(tau).all():
        raise ValueError("All task inputs must be finite")
    if np.any(rates <= 0.0) or np.any(rates >= 1.0) or np.unique(rates).size != 3 or np.any(tau <= 0.0):
        raise ValueError("Rates must be distinct in (0,1) and tolerances positive")
    ni = int(ddf_iterations)
    ng = int(guiding_iterations)
    if ni != ddf_iterations or ng != guiding_iterations or ni < 1 or ng < 1:
        raise ValueError("Iteration counts must be positive integers")

    # Candidate rows are deliberately numerical: source selection fixes these indices.
    profile_mode = 2
    response_mode = 3
    protocol_mode = 1
    theory_mode = 2

    oa4 = _oracle_prime_strength2_oa(3, 4)
    ai0 = _oracle_taguchi_initial_levels(-1.9, -0.8, 3)
    c00 = _oracle_taguchi_initial_levels(-1.25, -0.75, 3)
    initial_ddf = np.vstack((ai0[:3], ai0[:3], ai0[:3], c00[:3]))
    initial_ddf_spacing = np.array([ai0[3], ai0[3], ai0[3], c00[3]], dtype=float)
    train_z = np.array([0.75,2.25,3.75,5.25,6.75,8.25,9.75], dtype=float)
    folds = np.array([[1.5,4.5,7.5],[3.0,6.0,9.0]], dtype=float)
    dense_z = np.linspace(0.0, 10.0, 201)
    alpha_refs = np.array([0.244,0.281], dtype=float) * np.log(10.0) / 10.0
    references = np.exp(-alpha_refs[None,:,None] * folds[:,None,:])

    p0_mw = _oracle_sech_soliton_peak_power_mw(-7650.0, 1.3, 50.0)
    spans = np.array([0.17,0.24], dtype=float) * 105.5
    theory_states = np.stack([
        _oracle_guiding_center_theory_candidates(0.2611, float(span), p0_mw)[theory_mode]
        for span in spans
    ])
    oa2 = _oracle_prime_strength2_oa(3, 2)
    start = _oracle_taguchi_initial_levels(1.0, 10.0, 3)

    records = []
    for rate in rates:
        current = initial_ddf.copy()
        spacings = initial_ddf_spacing.copy()
        ddf_center = np.zeros(4, dtype=float)
        for _ in range(ni):
            responses = np.empty(oa4.shape[0], dtype=float)
            for row_index, row in enumerate(oa4):
                settings = current[np.arange(4), row]
                q = _oracle_dispersion_profile_candidates(settings[:3], settings[3], train_z)[profile_mode]
                if np.any(q <= 0.0):
                    responses[row_index] = 1.0e6
                else:
                    response_by_loss = [
                        np.sqrt(np.mean((1.0 - np.sqrt(np.exp(-alpha*train_z)/q))**2))
                        for alpha in alpha_refs
                    ]
                    responses[row_index] = float(max(response_by_loss))
            update = _oracle_taguchi_protocol_candidates(current, oa4, responses, spacings, float(rate))[protocol_mode]
            ddf_center = update[:,1].copy()
            spacings = update[:,2].copy()
            current = update[:,3:].copy()

        order_diag = _oracle_ddf_truncation_selector(
            ddf_center[:3], ddf_center[3], folds, references, dense_z, 0.03
        )
        selected_order = int(order_diag[6])
        terms = ddf_center[3] * np.power(10.0, ddf_center[:3])
        validation_profiles = []
        for fold in folds:
            first = terms[0] * fold
            second = (terms[1] * fold)**2 / 2.0
            third = (terms[2] * fold)**3 / 6.0
            pieces = (first, second, third)
            validation_profiles.append(1.0 + sum(pieces[:selected_order]))
        q_holdout = np.stack(validation_profiles)
        if np.any(q_holdout <= 0.0):
            raise ValueError("Selected DDF profile is nonpositive on a validation fold")
        flat_z = folds.ravel()
        alpha_ddf = float(-np.dot(flat_z, np.log(q_holdout.ravel())) / np.dot(flat_z, flat_z))

        branch_centers = np.empty((2, 2, 2), dtype=float)
        branch_responses = np.empty((2, 2), dtype=float)
        for branch in range(2):
            for span_index in range(2):
                current_gc = np.vstack((start[:3], start[:3]))
                gc_spacings = np.array([start[3], start[3]], dtype=float)
                center = np.zeros(2, dtype=float)
                state = theory_states[span_index]
                for _ in range(ng):
                    responses = np.empty(oa2.shape[0], dtype=float)
                    for row_index, row in enumerate(oa2):
                        gain, p_fac = current_gc[np.arange(2), row]
                        x = gain / state[1] - 1.0
                        y = p_fac / (state[2] / p0_mw) - 1.0
                        basis = np.array([1.0,x,y,x*y,x*x,y*y], dtype=float)
                        residual = basis @ tensor[span_index]
                        order_residual = residual[:4].copy()
                        if branch == 0:
                            order_residual += state[3] - 1.0
                        responses[row_index] = _oracle_guiding_center_response_candidates(
                            order_residual, residual[4], residual[5]
                        )[response_mode]
                    update = _oracle_taguchi_protocol_candidates(
                        current_gc, oa2, responses, gc_spacings, float(rate)
                    )[protocol_mode]
                    center = update[:,1].copy()
                    gc_spacings = update[:,2].copy()
                    current_gc = update[:,3:].copy()
                branch_centers[branch, span_index] = center
                x = center[0] / state[1] - 1.0
                y = center[1] / (state[2] / p0_mw) - 1.0
                basis = np.array([1.0,x,y,x*y,x*x,y*y], dtype=float)
                residual = basis @ tensor[span_index]
                order_residual = residual[:4].copy()
                if branch == 0:
                    order_residual += state[3] - 1.0
                branch_responses[branch, span_index] = _oracle_guiding_center_response_candidates(
                    order_residual, residual[4], residual[5]
                )[response_mode]

        theory_centers = branch_centers[1]
        if not np.isfinite(theory_centers).all() or np.any(theory_centers <= 0):
            raise ValueError("Final theoretical gain and launch factor must be finite and positive")
        alpha_guiding = float(np.dot(spans, np.log(theory_centers[:,0])) / np.dot(spans, spans))
        p0_relative_errors = np.empty(2, dtype=float)
        for span_index, (gain, p_fac) in enumerate(theory_centers):
            x = float(np.log(gain))
            inverse_multiplier = 1.0 if x == 0.0 else float(np.expm1(x) / (np.exp(x) * x))
            p0_relative_errors[span_index] = p_fac * inverse_multiplier - 1.0
        closure = _oracle_cross_example_closure_score(
            float(order_diag[7]), float(np.max(branch_responses[0])),
            float(np.max(branch_responses[1])), alpha_ddf, alpha_guiding,
            p0_relative_errors, tau
        )
        records.append({
            "rate": float(rate),
            "order": selected_order,
            "ddf_response": float(order_diag[7]),
            "alpha_ddf": alpha_ddf,
            "alpha_guiding": alpha_guiding,
            "unit_response": float(np.max(branch_responses[0])),
            "theory_response": float(np.max(branch_responses[1])),
            "p0_error": float(np.max(np.abs(p0_relative_errors))),
            "loss_mismatch": float(abs(alpha_ddf-alpha_guiding)),
            "active_channel": int(np.argmax(closure[6:11])),
            "joint_score": float(closure[11]),
        })

    ranked = sorted(records, key=lambda item: (item["joint_score"], item["rate"]))
    value = float(ranked[1]["joint_score"] - ranked[0]["joint_score"])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Joint-score gap must be finite and nonnegative")
    return value

def _archive_tensor():
    return _default_surrogate_tensor()

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight noncanonical end-to-end instances alter rates, archives, scales, and iteration counts."""
    base = 'b=_archive_tensor()'
    return [{'setup': base, 'call': 'cross_example_taguchi_gap(np.array([.46,.62,.79]),b,np.array([.052,.235,.064,.0022,.11]),3,4)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.46,.62,.79]),b,np.array([.052,.235,.064,.0022,.11]),3,4)'},
        {'setup': base + ';b[0,1,:]*=1.08', 'call': 'cross_example_taguchi_gap(np.array([.54,.69,.83]),b,np.array([.047,.252,.057,.0018,.095]),4,4)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.54,.69,.83]),b,np.array([.047,.252,.057,.0018,.095]),4,4)'},
        {'setup': base + ';b[1,2,:]*=.91', 'call': 'cross_example_taguchi_gap(np.array([.43,.58,.72]),b,np.array([.061,.221,.071,.0025,.13]),3,5)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.43,.58,.72]),b,np.array([.061,.221,.071,.0025,.13]),3,5)'},
        {'setup': base + ';b[:,0,:]+=np.array([.004,-.003,.002,-.001,.003,-.002])', 'call': 'cross_example_taguchi_gap(np.array([.57,.71,.86]),b,np.array([.044,.265,.053,.0017,.087]),5,4)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.57,.71,.86]),b,np.array([.044,.265,.053,.0017,.087]),5,4)'},
        {'setup': base + ';b[0,4,:]*=1.17', 'call': 'cross_example_taguchi_gap(np.array([.39,.63,.81]),b,np.array([.056,.229,.068,.0028,.118]),4,6)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.39,.63,.81]),b,np.array([.056,.229,.068,.0028,.118]),4,6)'},
        {'setup': base + ';b[1,5,:]*=1.14', 'call': 'cross_example_taguchi_gap(np.array([.51,.67,.77]),b,np.array([.049,.244,.062,.0021,.104]),5,5)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.51,.67,.77]),b,np.array([.049,.244,.062,.0021,.104]),5,5)'},
        {'setup': base + ';b[:,3,:]*=.88', 'call': 'cross_example_taguchi_gap(np.array([.48,.73,.89]),b,np.array([.058,.216,.075,.0031,.142]),3,6)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.48,.73,.89]),b,np.array([.058,.216,.075,.0031,.142]),3,6)'},
        {'setup': base + ';b[0,2,:]*=1.09;b[1,1,:]*=.93', 'call': 'cross_example_taguchi_gap(np.array([.42,.66,.84]),b,np.array([.046,.273,.051,.0016,.082]),5,6)', 'gold_call': '_oracle_cross_example_taguchi_gap(np.array([.42,.66,.84]),b,np.array([.046,.273,.051,.0016,.082]),5,6)'},
        {'setup': 'def _raises_value_error(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises_value_error(cross_example_taguchi_gap,surrogate_tensor=np.zeros((2,6,6)))', 'gold_call': '_raises_value_error(_oracle_cross_example_taguchi_gap,surrogate_tensor=np.zeros((2,6,6)))'}]
