"""
Final end-to-end orchestrator for the deterministic PCKAN two-step calibration benchmark.



This step composes the public APIs of Steps 01–06 in dependency order and returns the final differential-evolution calibration objective as one Python float. It does not reimplement the pricing model, training-data construction, PCKAN forward evaluation, physics residual, network training, or differential-evolution algorithm.



First, Step 01 generates the five fixed synthetic market prices using the Heston-limit FVSJ Fourier-cosine European-put construction at the ground-truth parameters. Step 02 constructs the deterministic 80-row PCKAN training and collocation sets.



Step 03 executes the initialized PCKAN forward map, and Step 04 evaluates the corresponding Black-Scholes physics residual and physical derivatives. The Step-04 residual output is explicitly validated by the orchestrator before training proceeds, making successful completion of the initialized physics-residual stage a required dependency of the end-to-end pipeline. The initialized Step-03 parameters themselves are not used as the final calibrated surrogate parameters.



Only after the initialized Step-04 physics residual has been successfully evaluated and validated does Step 05 train the PCKAN for exactly 500 full-batch Adam updates using the supervised COS-price loss plus the weighted physics-residual loss. Its trained CKAN coefficients, readout weights, and output bias supersede the Step-03 initialization and are frozen for calibration.



Finally, Step 06 performs the deterministic DE/rand/1/bin calibration of (kappa_1, theta_1) against the five fixed Step-01 synthetic market prices using only the frozen Step-05 PCKAN surrogate. The calibration uses the prescribed asynchronous in-place replacement procedure for exactly 60 evolution generations.



The orchestrator returns the Step-06 minimum five-contract sum of squared frozen-PCKAN pricing errors after generation 60. No calibrated parameter vector, training diagnostic, intermediate network output, physics-residual array, or COS price is returned by this final step.

The complete benchmark implements the paper-inspired two-step PCKAN calibration workflow by separating expensive offline option pricing and surrogate construction from inexpensive online parameter calibration.



In the offline pricing stage, the FVSJ model is specialized to its single-factor Heston limit by setting lambda = 0, kappa_2 = theta_2 = sigma_2 = 0, and H_1 = H_2 = 1/2. Step 01 evaluates European puts with the Fourier-cosine construction, using the specialized characteristic function, cumulant-based truncation interval, and half-weighted zeroth COS term. At the benchmark ground truth (kappa_1, theta_1) = (1.5, 0.04), this produces the five fixed synthetic market prices used throughout calibration.



Step 02 constructs the supervised and collocation data over the prescribed Cartesian product of kappa_1, theta_1, and the five benchmark contracts. Each PCKAN input has the scaled form



(S0/100, K/100, T, r, kappa_1/4, theta_1/0.12).



The resulting training and collocation sets each contain 80 rows.



Steps 03 and 04 expose the two principal components of the physics-informed surrogate. The PCKAN first applies tanh coordinatewise and then expands each transformed input in the second-kind Chebyshev basis



U_0(z) = 1,

U_1(z) = 2z,

U_2(z) = 4z^2 - 1,

U_3(z) = 8z^3 - 4z.



The CKAN coefficients combine these expansions into eight hidden coordinates, followed by a scalar linear readout. The physics component uses the Black-Scholes PDE only as a training constraint. With T interpreted as time to maturity and sigma_phys = sqrt(v0) = 0.2, the residual is



R = rV - V_T - r S V_S

    - (1/2) sigma_phys^2 S^2 V_SS.



Because the network receives S0/100 rather than physical spot directly, the spot derivatives include the required chain-rule factors



V_S = (1/100) V_{x_0},



V_SS = (1/10000) V_{x_0 x_0}.



Step 05 combines the supervised and physics terms through



L = MSE_data + 0.1 MSE_physics



and performs exactly 500 full-batch Adam updates. After update 500, the trained CKAN coefficients, scalar readout weights, and output bias are frozen.



The online calibration stage in Step 06 searches only over



kappa_1 in [0.5, 4.0],

theta_1 in [0.01, 0.12].



For a candidate parameter vector (kappa_1, theta_1), the frozen PCKAN predicts prices for the five benchmark contracts and the calibration objective is



J(kappa_1, theta_1)

= sum_{i=1}^{5}

  (P_i^mkt - P_i^PCKAN(kappa_1, theta_1))^2.



The synthetic market prices are fixed before differential evolution begins. Consequently, the expensive Fourier-cosine pricer is not called inside the calibration objective.



Step 06 minimizes J using the deterministic DE/rand/1/bin procedure with population size 20, mutation factor F = 0.8, crossover probability CR = 0.9, numpy.random.default_rng(0), and exactly 60 evolution generations. Replacement is asynchronous and in-place, so an accepted trial immediately becomes available when later population members are processed within the same generation.



Step 07 is the final orchestrator. It invokes the public APIs of Steps 01–06 in sequence. Steps 03 and 04 explicitly exercise the initialized forward and physics-residual components, while the trained parameters returned by Step 05 are the parameters passed to Step 06. The final quantity returned by the orchestrator is the minimum five-contract sum of squared pricing errors after the prescribed generation-60 differential-evolution calibration.

Returns
-------
return float(best_objective)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_pckan_calibration_orchestrator() -> float:
    """
    Run the complete deterministic PCKAN calibration pipeline
    through Steps 01–06.

    The orchestrator generates the five fixed Heston-limit FVSJ COS
    market prices, constructs the PCKAN training and collocation data,
    executes the initialized PCKAN forward and physics-residual
    diagnostic stages, trains the PCKAN for 500 full-batch Adam
    updates, and finally performs the deterministic 60-generation
    frozen-surrogate differential-evolution calibration.

    Steps 03 and 04 are diagnostic stages only. Their initialized
    parameters are not used for final calibration. The trained
    parameters returned by Step 05 are frozen and passed to Step 06.

    Returns
    -------
    final_objective : float
        Minimum five-contract sum of squared pricing errors returned
        by the Step-06 differential-evolution calibration.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_pckan_calibration_orchestrator():
    """Independent composition of Steps 01–06 oracle APIs."""

    import numpy as np

    # Step 01 — fixed synthetic Heston-limit FVSJ COS market prices
    market_prices = _oracle_compute_heston_cos_put_prices(
        100.0,
        [90.0, 95.0, 100.0, 105.0, 110.0],
        [0.25, 0.50, 0.75, 1.00, 1.50],
        0.05,
        1.5,
        0.04,
        0.30,
        -0.5,
        0.04,
        128,
        10.0,
    )

    # Step 02 — supervised training and collocation data
    train_inputs, train_prices, collocation_inputs, _physical_rows = (
        _oracle_build_pckan_training_data()
    )

    # Step 03 — deterministic initialized PCKAN forward stage
    _init_preds, W_hat_init, W_out_init, b_out_init = (
        _oracle_initialize_and_evaluate_pckan(
            train_inputs
        )
    )

    # Step 04 — required initialized-model physics residual stage
    _pred, _dV_dS, _d2V_dS2, _dV_dT, _residuals = (
        _oracle_compute_pckan_physics_residual(
            train_inputs,
            W_hat_init,
            W_out_init,
            b_out_init,
        )
    )

    # Step 04 is a required dependency of the end-to-end pipeline.
    # Validate its returned residual before allowing training to proceed.
    _residuals_arr = np.asarray(_residuals, dtype=float)

    if _residuals_arr.shape != (80,):
        raise ValueError(
            "Step 04 must return exactly 80 physics residuals."
        )

    if not np.all(np.isfinite(_residuals_arr)):
        raise ValueError(
            "Step 04 physics residuals must all be finite."
        )

    # Step 05 — train the physics-informed PCKAN only after the
    # required Step-04 physics stage has completed successfully.
    (
        W_hat,
        W_out,
        b_out,
        _train_preds,
        _data_mse,
        _physics_mse,
        _total_loss,
    ) = _oracle_train_pckan(
        train_inputs,
        train_prices,
        collocation_inputs,
    )

    # Step 06 — calibrate using only the frozen Step-05 surrogate
    _best_parameters, best_objective = (
        _oracle_calibrate_with_differential_evolution(
            W_hat,
            W_out,
            b_out,
            market_prices,
        )
    )

    return float(best_objective)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # CASE 1 — full end-to-end frozen objective
        {
            "setup": (
                "obj = run_pckan_calibration_orchestrator()\n"
                "result = (isinstance(obj, float), float(obj))"
            ),
            "call": "result",
            "gold_call": (
                "(True, "
                "_oracle_run_pckan_calibration_orchestrator())"
            ),
        },

        # CASE 2 — determinism across repeated full pipelines
        {
            "setup": (
                "a = run_pckan_calibration_orchestrator()\n"
                "b = run_pckan_calibration_orchestrator()\n"
                "result = (float(a) == float(b), float(a))"
            ),
            "call": "result",
            "gold_call": (
                "(True, "
                "_oracle_run_pckan_calibration_orchestrator())"
            ),
        },

        # CASE 3 — Step 04 is a required end-to-end dependency
        {
            "setup": (
                "import numpy as np\n"
                "full = float(run_pckan_calibration_orchestrator())\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "ip, Wh0, Wo0, b0 = initialize_and_evaluate_pckan(ti)\n"
                "pred, d1, d2, dt, residuals = "
                "compute_pckan_physics_residual(\n"
                "    ti, Wh0, Wo0, b0\n"
                ")\n"
                "residuals = np.asarray(residuals, dtype=float)\n"
                "result = (\n"
                "    residuals.shape == (80,),\n"
                "    bool(np.all(np.isfinite(residuals))),\n"
                "    full,\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, float(_oracle_run_pckan_calibration_orchestrator()))",
        },

        # CASE 4 — Step-05 trained weights, not Step-03 initialization
        {
            "setup": (
                "import numpy as np\n"
                "full = float(run_pckan_calibration_orchestrator())\n"
                "markets = compute_heston_cos_put_prices(\n"
                "    100.0, [90, 95, 100, 105, 110],\n"
                "    [0.25, 0.5, 0.75, 1.0, 1.5],\n"
                "    0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0,\n"
                ")\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "ip, Wh0, Wo0, b0 = initialize_and_evaluate_pckan(ti)\n"
                "best0, obj0 = calibrate_with_differential_evolution(\n"
                "    Wh0, Wo0, b0, markets\n"
                ")\n"
                "result = (\n"
                "    full,\n"
                "    float(obj0) != full,\n"
                "    float(b0) == 0.0,\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(float(_oracle_run_pckan_calibration_orchestrator()), True, True)",
        },

        # CASE 5 — Step-01 market panel contract and final composition
        {
            "setup": (
                "import numpy as np\n"
                "markets = compute_heston_cos_put_prices(\n"
                "    100.0, [90, 95, 100, 105, 110],\n"
                "    [0.25, 0.5, 0.75, 1.0, 1.5],\n"
                "    0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0,\n"
                ")\n"
                "obj = float(run_pckan_calibration_orchestrator())\n"
                "result = (\n"
                "    bool(\n"
                "        markets.shape == (5,)\n"
                "        and np.all(np.isfinite(markets))\n"
                "    ),\n"
                "    obj,\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, float(_oracle_run_pckan_calibration_orchestrator()))",
        },

        # CASE 6 — Step-05 output contract and final composition
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "obj = float(run_pckan_calibration_orchestrator())\n"
                "result = (\n"
                "    Wh.shape == (8, 6, 4),\n"
                "    Wo.shape == (8,),\n"
                "    preds.shape == (80,),\n"
                "    bool(np.all(np.isfinite(Wh))),\n"
                "    bool(np.all(np.isfinite(Wo))),\n"
                "    bool(np.all(np.isfinite(preds))),\n"
                "    bool(np.isfinite(float(b))),\n"
                "    bool(np.isfinite(float(dm))),\n"
                "    bool(np.isfinite(float(pm))),\n"
                "    bool(np.isfinite(float(tl))),\n"
                "    obj,\n"
                ")"
            ),
            "call": "result",
            "gold_call": (
                "(True, True, True, True, True, True, True, True, True, True, "
                "float(_oracle_run_pckan_calibration_orchestrator()))"
            ),
        },

        # CASE 7 — finite positive objective and type contract
        {
            "setup": (
                "import math\n"
                "obj = run_pckan_calibration_orchestrator()\n"
                "result = (\n"
                "    isinstance(obj, float),\n"
                "    math.isfinite(obj),\n"
                "    obj > 0.0,\n"
                "    float(obj),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, True, float(_oracle_run_pckan_calibration_orchestrator()))",
        },
    ]
