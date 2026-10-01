"""
The calibration stage follows the surrogate-assisted second stage of the PCKAN framework. The computationally expensive Heston-limit FVSJ COS pricing model is used offline to generate the fixed synthetic market prices and supervised training targets. During calibration, candidate parameter vectors are evaluated only through the trained and frozen PCKAN surrogate.



The two calibrated parameters are kappa_1 and theta_1. For a candidate parameter vector (kappa_1, theta_1), the five benchmark contracts are



(K, T) =

(90, 0.25),

(95, 0.50),

(100, 0.75),

(105, 1.00),

(110, 1.50).



With S0 = 100 and r = 0.05, each candidate produces the five PCKAN feature rows



(S0/100, K/100, T, r, kappa_1/4, theta_1/0.12).



Thus the first coordinate is 1.0 and the candidate-dependent coordinates are kappa_1/4 and theta_1/0.12.



The frozen PCKAN applies tanh coordinatewise to these six scaled inputs and evaluates the second-kind Chebyshev basis through order 3:



U_0(z) = 1,

U_1(z) = 2z,

U_2(z) = 2z U_1(z) - U_0(z),

U_3(z) = 2z U_2(z) - U_1(z).



For the frozen coefficient tensor W_hat with shape (8, 6, 4), the eight hidden coordinates are formed by contracting W_hat with the Chebyshev-expanded inputs. The frozen readout weights W_out with shape (8,) and frozen scalar bias b_out then produce the five surrogate prices.



For fixed synthetic market prices P_i^mkt and frozen-surrogate predictions P_i^PCKAN, the calibration objective is



J(kappa_1, theta_1)

=

sum_{i=1}^{5}

(P_i^mkt - P_i^PCKAN)^2.



The optimization criterion is therefore the five-contract sum of squared errors, not the mean squared error. The FVSJ/COS pricing model is not reevaluated for candidate parameter vectors during differential evolution.



The bounded calibration domain is



kappa_1 in [0.5, 4.0],



theta_1 in [0.01, 0.12].



The benchmark uses DE/rand/1/bin differential evolution with population size 20, mutation factor F = 0.8, crossover probability CR = 0.9, and numpy.random.default_rng(0).



Generation 0 is initialized uniformly according to



population =

lo + (hi - lo) * rng.random((20, 2)),



where



lo = [0.5, 0.01]



and



hi = [4.0, 0.12].



Initial objective values are evaluated for all 20 generation-0 population members.



For each evolution generation, targets are processed sequentially in population-index order i = 0,...,19.



For target i, the eligible donor indices consist of every population index except i. Three distinct donor indices are sampled without replacement in one RNG operation. In the sampled order, their current population members define a, b, and c.



The DE/rand/1 mutant is



v = a + F(b - c),



with F = 0.8.



Because population replacement is asynchronous, a, b, and c are taken from the current population at the instant target i is processed. Consequently, accepted trials from earlier target indices in the same generation can participate as donors for later target indices.



Each mutant coordinate is then checked against its corresponding parameter bounds in coordinate order. If a coordinate is outside its allowed interval, it is repaired using an independent uniform redraw over that full interval. A coordinate already inside its interval consumes no repair random draw.



After bound repair, one forced crossover coordinate jrand is sampled uniformly from {0, 1} using



jrand = int(rng.integers(0, 2)).



Binomial crossover is then processed in coordinate order j = 0, 1.



The deterministic benchmark requires one crossover random draw to be consumed for every coordinate, including the forced coordinate. Therefore the crossover operation is conceptually



for j in range(2):

    crossover_draw = rng.random()

    if crossover_draw < CR or j == jrand:

        trial[j] = mutant[j]



with CR = 0.9.



The random draw corresponding to jrand is intentionally consumed even though that coordinate is guaranteed to come from the mutant. This RNG-consumption rule is part of the benchmark specification because numpy.random.Generator is stateful. Omitting the random draw for jrand changes every subsequent RNG state and therefore changes donor selection, bound repair, crossover, and ultimately the deterministic generation-60 result.



Accordingly, an implementation that first checks whether j == jrand and only calls rng.random() for non-forced coordinates is not equivalent for this benchmark. Likewise, short-circuit logic that avoids the random draw on the forced coordinate does not reproduce the specified deterministic trajectory.



After crossover, the trial objective is evaluated using the frozen PCKAN surrogate. Greedy selection uses



J(trial) <= J(target).



If this condition holds, the target population member and its stored objective are replaced immediately by the trial and its objective.



This replacement is asynchronous and in-place. No frozen generation-level population copy is used. Therefore the population can change while a generation is being processed.



Exactly 60 evolution generations are executed after generation 0. There is no early stopping.



After generation 60, the best population index is selected using



best_idx = np.argmin(objectives).



The corresponding two-dimensional population member is returned as the calibrated [kappa_1, theta_1] vector, together with its five-contract sum-of-squared-errors objective.



Throughout calibration, W_hat, W_out, and b_out remain frozen. No gradient-based optimization, PCKAN retraining, Black-Scholes physics-residual evaluation, or candidate-wise COS pricing is performed.



The deterministic differential-evolution trajectory depends not only on the mathematical DE/rand/1/bin rules but also on the exact ordering and consumption of random-number-generator calls. In particular, the benchmark consumes exactly one crossover rng.random() draw for each of the two coordinates of every target, including the forced crossover coordinate.

The calibration stage follows the surrogate-assisted second step of the PCKAN framework. The computationally expensive Heston-limit FVSJ COS pricing model is used offline to generate the fixed synthetic market prices, while calibration itself evaluates candidate model parameters only through the trained and frozen PCKAN surrogate.



The two calibrated parameters are kappa_1 and theta_1. For a candidate parameter vector (kappa_1, theta_1), the five benchmark contracts are represented by the PCKAN inputs



(S0/100, K/100, T, r, kappa_1/4, theta_1/0.12),



with S0 = 100 and r = 0.05. The frozen PCKAN applies tanh to these scaled coordinates, expands them in the second-kind Chebyshev basis through order 3, forms the eight CKAN hidden coordinates, and applies the trained scalar linear readout.



For the five fixed synthetic market prices P_i^mkt and corresponding frozen-surrogate predictions P_i^PCKAN, the calibration objective is



J(kappa_1, theta_1)

= sum_{i=1}^{5} (P_i^mkt - P_i^PCKAN)^2.



Thus the optimization criterion is the five-contract sum of squared errors, not the mean squared error. The COS pricing model is not reevaluated for candidate parameter vectors during calibration.



The parameter search uses DE/rand/1/bin differential evolution. For a target population member, three distinct donor members a, b, and c are selected and the mutant is constructed as



v = a + F(b - c),



with mutation factor F = 0.8. Each mutant coordinate that leaves its allowed parameter interval is repaired by an independent uniform redraw over that coordinate's full admissible interval.



Binomial crossover combines the target and mutant coordinates using crossover probability CR = 0.9, with one randomly selected coordinate forced to come from the mutant. The resulting trial vector is accepted whenever its objective is less than or equal to the current target objective.



The benchmark uses population size 20 and numpy.random.default_rng(0). Replacement is asynchronous and in-place: once a trial is accepted, it immediately becomes part of the population and can therefore participate in donor selection for later targets in the same generation. This differs from synchronous differential evolution, in which all trials would be constructed from an unchanged copy of the previous generation, and the distinction changes the deterministic optimization trajectory.



Generation 0 is the uniformly initialized population. Exactly 60 evolution generations are subsequently completed. After generation 60, the population member having the smallest objective defines the calibrated parameter vector and the reported minimum calibration objective.



Throughout this stage, the Step-05 PCKAN coefficients and readout parameters remain frozen. No gradient-based optimization, PCKAN retraining, Black-Scholes physics-residual evaluation, or candidate-wise COS pricing is performed. The role of differential evolution is solely to search the bounded (kappa_1, theta_1) parameter space using the frozen surrogate as the pricing evaluator.

Returns
-------
return best_parameters, best_objective
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_with_differential_evolution(
    W_hat: np.ndarray,
    W_out: np.ndarray,
    b_out: float,
    market_prices: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate (kappa_1, theta_1) against the five fixed synthetic
    market prices using the frozen PCKAN surrogate and deterministic
    asynchronous DE/rand/1/bin.

    The calibration uses numpy.random.default_rng(0), population size
    20, mutation factor F=0.8, crossover probability CR=0.9, and
    exactly 60 evolution generations.

    Targets are processed in population-index order with asynchronous
    in-place replacement. Three distinct donor indices are sampled
    without replacement from all population members except the current
    target. Out-of-bounds mutant coordinates are repaired in coordinate
    order by independent uniform redraws over their corresponding
    admissible intervals.

    Binomial crossover selects one forced coordinate jrand. One
    rng.random() crossover draw is consumed for every coordinate,
    including jrand. A coordinate is copied from the mutant whenever
    its crossover draw is below CR or it is the forced coordinate.

    Parameters
    ----------
    W_hat : np.ndarray
        Frozen trained CKAN coefficient tensor with shape (8, 6, 4).
    W_out : np.ndarray
        Frozen trained scalar-readout weights with shape (8,).
    b_out : float
        Frozen trained scalar output bias.
    market_prices : np.ndarray
        Five fixed synthetic Heston-limit FVSJ COS European-put prices
        in benchmark contract order, with shape (5,).

    Returns
    -------
    best_parameters : np.ndarray
        Float64 array with shape (2,) containing the generation-60
        best [kappa_1, theta_1] parameter vector.
    best_objective : float
        Minimum generation-60 sum of squared pricing errors over the
        five benchmark contracts.

    Raises
    ------
    ValueError
        If W_hat does not have shape (8, 6, 4), W_out does not have
        shape (8,), market_prices does not have shape (5,), b_out is
        non-finite, or any supplied array contains a non-finite value.
    """
    return (
        np.empty((2,), dtype=np.float64),
        0.0,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_calibrate_with_differential_evolution(
    W_hat,
    W_out,
    b_out,
    market_prices,
):
    """Run the deterministic frozen-PCKAN DE/rand/1/bin calibration."""
    import numpy as np

    W_hat = np.asarray(W_hat, dtype=np.float64)
    W_out = np.asarray(W_out, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)

    if W_hat.shape != (8, 6, 4):
        raise ValueError("W_hat must have shape (8, 6, 4)")

    if W_out.shape != (8,):
        raise ValueError("W_out must have shape (8,)")

    if market_prices.shape != (5,):
        raise ValueError("market_prices must have shape (5,)")

    if not np.all(np.isfinite(W_hat)):
        raise ValueError("W_hat must be finite")

    if not np.all(np.isfinite(W_out)):
        raise ValueError("W_out must be finite")

    if not np.all(np.isfinite(market_prices)):
        raise ValueError("market_prices must be finite")

    try:
        b_out = float(b_out)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("b_out must be a finite scalar") from exc

    if not np.isfinite(b_out):
        raise ValueError("b_out must be a finite scalar")

    contracts = (
        (90.0, 0.25),
        (95.0, 0.50),
        (100.0, 0.75),
        (105.0, 1.00),
        (110.0, 1.50),
    )

    lo = np.array(
        [0.5, 0.01],
        dtype=np.float64,
    )

    hi = np.array(
        [4.0, 0.12],
        dtype=np.float64,
    )

    def _predict(candidate):
        kappa_1 = float(candidate[0])
        theta_1 = float(candidate[1])

        inputs = np.empty(
            (5, 6),
            dtype=np.float64,
        )

        for i, (strike, maturity) in enumerate(contracts):
            inputs[i] = (
                1.0,
                strike / 100.0,
                maturity,
                0.05,
                kappa_1 / 4.0,
                theta_1 / 0.12,
            )

        z = np.tanh(inputs)

        u0 = np.ones_like(z)
        u1 = 2.0 * z
        u2 = 2.0 * z * u1 - u0
        u3 = 2.0 * z * u2 - u1

        U = np.stack(
            (u0, u1, u2, u3),
            axis=-1,
        )

        hidden = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            U,
        )

        predictions = (
            np.einsum(
                "bq,q->b",
                hidden,
                W_out,
            )
            + b_out
        )

        return predictions

    def _objective(candidate):
        predictions = _predict(candidate)
        errors = market_prices - predictions

        return float(
            np.sum(errors * errors)
        )

    # Deterministic local RNG.
    rng = np.random.default_rng(0)

    population_size = 20
    mutation_factor = 0.8
    crossover_probability = 0.9

    # Generation 0.
    population = (
        lo
        + (hi - lo)
        * rng.random((population_size, 2))
    )

    objectives = np.array(
        [
            _objective(population[i])
            for i in range(population_size)
        ],
        dtype=np.float64,
    )

    all_indices = np.arange(population_size)

    # Exactly 60 evolution generations.
    for _ in range(60):
        for i in range(population_size):

            # Current target cannot be selected as a donor.
            eligible = all_indices[
                all_indices != i
            ]

            donor_indices = rng.choice(
                eligible,
                size=3,
                replace=False,
            )

            a = population[donor_indices[0]]
            bb = population[donor_indices[1]]
            c = population[donor_indices[2]]

            # DE/rand/1 mutation.
            mutant = (
                a
                + mutation_factor * (bb - c)
            ).copy()

            # Coordinate-order bound repair.
            for j in range(2):
                if (
                    mutant[j] < lo[j]
                    or mutant[j] > hi[j]
                ):
                    mutant[j] = (
                        lo[j]
                        + (hi[j] - lo[j])
                        * rng.random()
                    )

            # One forced crossover coordinate.
            jrand = int(
                rng.integers(0, 2)
            )

            trial = population[i].copy()

            # Binomial crossover.
            #
            # IMPORTANT:
            # Consume one rng.random() for EVERY coordinate,
            # including the forced jrand coordinate.
            for j in range(2):
                crossover_draw = rng.random()

                if (
                    crossover_draw < crossover_probability
                    or j == jrand
                ):
                    trial[j] = mutant[j]

            trial_objective = _objective(trial)

            # Asynchronous / in-place replacement.
            if trial_objective <= objectives[i]:
                population[i] = trial
                objectives[i] = trial_objective

    best_idx = int(
        np.argmin(objectives)
    )

    best_parameters = (
        population[best_idx]
        .astype(
            np.float64,
            copy=True,
        )
    )

    best_objective = float(
        objectives[best_idx]
    )

    return (
        best_parameters,
        best_objective,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # CASE 1 — full deterministic benchmark result
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "best, obj = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "result = (\n"
                "    best.shape,\n"
                "    best.dtype == np.float64,\n"
                "    isinstance(obj, float),\n"
                "    float(best[0]),\n"
                "    float(best[1]),\n"
                "    float(obj),\n"
                ")"
            ),
            "call": "result",
            "gold_call": (
                "(lambda out: ("
                "out[0].shape, "
                "out[0].dtype == np.float64, "
                "isinstance(out[1], float), "
                "float(out[0][0]), "
                "float(out[0][1]), "
                "float(out[1])"
                "))("
                "_oracle_calibrate_with_differential_evolution("
                "Wh, Wo, b, market_prices"
                "))"
            ),
        },

        # CASE 2 — deterministic despite unrelated global RNG activity
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "np.random.seed(12345)\n"
                "_ = np.random.random(1000)\n"
                "a = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "np.random.seed(98765)\n"
                "_ = np.random.random(500)\n"
                "c = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "result = (\n"
                "    bool(np.array_equal(a[0], c[0])),\n"
                "    float(a[1]) == float(c[1]),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True)",
        },

        # CASE 3 — objective is five-contract SSE, not MSE
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "best, obj = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "contracts = [\n"
                "    (90.0, 0.25),\n"
                "    (95.0, 0.50),\n"
                "    (100.0, 0.75),\n"
                "    (105.0, 1.00),\n"
                "    (110.0, 1.50),\n"
                "]\n"
                "X = np.array([\n"
                "    [1.0, K/100.0, T, 0.05,\n"
                "     best[0]/4.0, best[1]/0.12]\n"
                "    for K, T in contracts\n"
                "], dtype=np.float64)\n"
                "z = np.tanh(X)\n"
                "u0 = np.ones_like(z)\n"
                "u1 = 2.0 * z\n"
                "u2 = 2.0 * z * u1 - u0\n"
                "u3 = 2.0 * z * u2 - u1\n"
                "U = np.stack((u0, u1, u2, u3), axis=-1)\n"
                "h = np.einsum('qpn,bpn->bq', Wh, U)\n"
                "p = np.einsum('bq,q->b', h, Wo) + b\n"
                "sse = float(np.sum((market_prices - p) ** 2))\n"
                "mse = float(np.mean((market_prices - p) ** 2))\n"
                "result = (\n"
                "    bool(np.isclose(\n"
                "        obj, sse, rtol=0.0, atol=1e-12\n"
                "    )),\n"
                "    bool(not np.isclose(\n"
                "        obj, mse, rtol=0.0, atol=1e-12\n"
                "    )),\n"
                "    bool(np.isclose(\n"
                "        sse, 5.0*mse, rtol=0.0, atol=1e-12\n"
                "    )),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, True)",
        },

        # CASE 4 — frozen surrogate inputs are not modified
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "Wh0 = Wh.copy()\n"
                "Wo0 = Wo.copy()\n"
                "b0 = float(b)\n"
                "mp0 = market_prices.copy()\n"
                "_ = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "result = (\n"
                "    bool(np.array_equal(Wh, Wh0)),\n"
                "    bool(np.array_equal(Wo, Wo0)),\n"
                "    float(b) == b0,\n"
                "    bool(np.array_equal(market_prices, mp0)),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, True, True)",
        },

        # CASE 5 — returned best individual remains inside bounds
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "best, obj = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "result = (\n"
                "    bool(0.5 <= best[0] <= 4.0),\n"
                "    bool(0.01 <= best[1] <= 0.12),\n"
                "    bool(np.isfinite(obj)),\n"
                "    bool(obj >= 0.0),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, True, True)",
        },

        # CASE 6 — exact deterministic DE trajectory matches oracle
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "best, obj = calibrate_with_differential_evolution(\n"
                "    Wh, Wo, b, market_prices\n"
                ")\n"
                "result = tuple(float(x) for x in np.ravel(best)) + (float(obj),)"
            ),
            "call": "result",
            "gold_call": (
                "(lambda g: tuple(float(x) for x in np.ravel(g[0])) + (float(g[1]),))("
                "_oracle_calibrate_with_differential_evolution(Wh, Wo, b, market_prices))"
            ),
        },

        # CASE 7 — malformed and nonfinite inputs raise ValueError
        {
            "setup": (
                "import numpy as np\n"
                "ti, tp, ci, pr = build_pckan_training_data()\n"
                "Wh, Wo, b, preds, dm, pm, tl = train_pckan(ti, tp, ci)\n"
                "market_prices = np.array([\n"
                "    0.7180726206185243,\n"
                "    2.6485698239608793,\n"
                "    4.98268141598846,\n"
                "    7.536523841443863,\n"
                "    10.32593069625738,\n"
                "], dtype=np.float64)\n"
                "cases = [\n"
                "    (Wh[:, :, :3], Wo, b, market_prices),\n"
                "    (Wh, Wo[:-1], b, market_prices),\n"
                "    (Wh, Wo, b, market_prices[:-1]),\n"
                "    (Wh, Wo, float('nan'), market_prices),\n"
                "]\n"
                "Wh_bad = Wh.copy()\n"
                "Wh_bad[0, 0, 0] = np.nan\n"
                "cases.append((Wh_bad, Wo, b, market_prices))\n"
                "statuses = []\n"
                "for args in cases:\n"
                "    try:\n"
                "        calibrate_with_differential_evolution(*args)\n"
                "        statuses.append(0)\n"
                "    except ValueError:\n"
                "        statuses.append(1)\n"
                "    except Exception:\n"
                "        statuses.append(2)"
            ),
            "call": "tuple(statuses)",
            "gold_call": "(1, 1, 1, 1, 1)",
        },
    ]
