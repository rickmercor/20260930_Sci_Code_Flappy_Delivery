"""
Step 1: estimate the age-specific rates from the monitoring records.

The records come from a marked female population followed by capture and recapture. One table holds the release groups of an April census programme: at each census every female present is caught with one common probability and released at once, and the group's first recaptures and its females later found dead are recorded. The other table holds the reproduction records of the May birth pulse, which the census follows by eleven months, so a female's age in completed years differs between the two tables.

The census paused in 1997 and 1998. A row's recapture columns therefore follow the census sequence rather than the calendar, and survival between two censuses several years apart is the product of the yearly survival over each year in between.

Survival is the age-specific probability of living from one April census to the next, and fecundity the number of female offspring per female at the pulse. Both are estimated from the records, survival through the capture model and fecundity by pooling.

Returns
-------
out : tuple of (np.ndarray, np.ndarray)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_rates(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    terminal_age: int = 13,
) -> tuple:
    """Estimate the age-specific survival and fecundity from the monitoring records.

    Parameters
    ----------
    release_records : array-like of shape (n_records, 4 + K)
        Integer records of the April census programme, one row per release group:
        ``[release year, birth year, females released, females later found dead,
        first recaptures at the 1st, 2nd, ..., K-th following census]``. A female
        released at an April census is counted in the release year. A recapture at the
        j-th following census means the female was seen again for the first time at
        that census, and a female not listed there counts as not yet recaptured. ``K``
        is the same for every row. The census paused in 1997 and 1998, so the rows'
        release years skip those two years. The j-th following census is the j-th
        census that was run after the release, not the j-th calendar year. The census
        years are the years present in the table plus the consecutive years after its
        last release, and a missing year inside the table's span is a paused census.
        At each census every female present is captured with the same probability,
        independently of the other females and of the year, and is released at once.
        Three rules turn these printed counts into the data the fit uses, in this
        order. Confirmation: the most recent census at which a group was captured is
        not confirmed by a later sighting, so the last recapture column of the group
        holding a positive count is emptied and those females move into the group's
        not-recaptured total. Validity: a group left with no confirmed capture does
        not enter the fit. Recovery removal: a female found dead is not at risk after
        her death, so a group's at-risk total is its released count minus its females
        found dead.
    reproduction_records : array-like of shape (n_records, 5)
        Integer records, one row per cohort and May birth pulse: ``[pulse year, birth
        year, females present at the pulse, females that gave birth, fawns born]``.
        The records cover maternal ages of two years and older (females first
        reproduce at age two), and the female share of the fawns is one half.
    terminal_age : int, optional
        Age class at and above which classes are pooled (the records pool ages 13 and
        older). A female's class is her age in completed years: at the April census
        for the release records, and at the May pulse for the reproduction records.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``survival`` (length ``terminal_age + 1``): for each class, the survival
        probability given by the better fitting of two stated curves in the class age
        ``a`` in years,

            late:     s(a) = (s0 + (sp - s0) * (1 - exp(-k * a))) * exp(-d * max(0, a - 8))
            logistic: s(a) = (s0 + (sp - s0) * (1 - exp(-k * a))) / (1 + exp((a - 11) / w))

        where ``s0`` is the newborn survival, ``sp`` the adult plateau, ``k`` the
        juvenile approach rate, and ``d`` or ``w`` the decline parameter. Both curves
        take their value at age 13 for the pooled terminal class, which keeps the same
        females in place. The four parameters of each curve and the common capture
        probability are estimated jointly by maximum likelihood, once per curve, and
        the reported rates are those of the curve whose maximized likelihood is larger.
        Each fit must be converged, not merely started, with every estimate stable to
        at least four decimal places. The rates are rounded to three decimal places.
        ``fecundity`` (length ``terminal_age + 1``): for each class, the pooled ratio
        of female fawns (fawns born times one half) to females present over that
        class's records, keyed by the mother's age at the pulse, rounded to three
        decimal places. Classes zero and one contribute zero.

    Raises
    ------
    ValueError
        If the release table is not a non-empty 2-D integer array. The table must have
        non-negative entries, at least six columns, whole-integer years with the
        release year after the birth year, and no more first recaptures in a row than
        females released. ``terminal_age`` must be at least 2. The rules must not
        leave a group with fewer females at risk than its confirmed captures. Every
        survival class from zero through ``terminal_age`` must appear among the usable
        release groups. The reproduction records must be a 2-D integer array of shape
        ``(n, 5)`` with non-negative entries and no more females giving birth than
        were present. Every fecundity class from two through ``terminal_age`` must
        have records.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_estimate_rates(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    terminal_age: int = 13,
) -> tuple:
    """Reference implementation for estimate_rates."""
    import numpy as np

    def _curve(age, s0, sp, k, extra, terminal_age, model="late"):
        """The stated survival curves. The pooled class takes the terminal-age value."""
        import numpy as np

        a = np.minimum(np.asarray(age, dtype=float), float(terminal_age))
        base = s0 + (sp - s0) * (1.0 - np.exp(-k * a))
        if model == "late":
            return base * np.exp(-extra * np.maximum(0.0, a - 8.0))
        return base / (1.0 + np.exp((a - 11.0) / extra))


    def _unpack(theta):
        import numpy as np

        s0, sp, p = 1.0 / (1.0 + np.exp(-theta[:3]))
        k, d = np.exp(theta[3]), np.exp(theta[4])
        return s0, sp, k, d, p


    def _census_schedule(all_years, k_cells):
        """The census years implied by a printed table: the years present, plus the
        consecutive years after the last one. Missing years inside the span are the
        paused censuses, so a row's columns follow the census sequence, not the years."""
        import numpy as np

        present = sorted({int(y) for y in all_years})
        pset = set(present)
        schedule = [y for y in range(present[0], present[-1] + 1) if y in pset]
        schedule += list(range(present[-1] + 1, present[-1] + 1 + k_cells))
        return schedule


    def _intervals(all_years, k_cells):
        """Per row, the years from its release census to each of the following K."""
        import numpy as np

        schedule = _census_schedule(all_years, k_cells)
        out = np.empty((len(all_years), k_cells))
        for i, y in enumerate(all_years):
            following = [c for c in schedule if c > y][:k_cells]
            out[i] = [c - y for c in following]
        return out


    def _apply_rules(m):
        """Confirmation, then validity, then recovery removal. Returns effective data."""
        import numpy as np

        years = m[:, 0].astype(float)
        births = m[:, 1].astype(float)
        released = m[:, 2].astype(float)
        dead = m[:, 3].astype(float)
        cols = m[:, 4:].astype(float).copy()
        deltas = _intervals(years, cols.shape[1])
        for i in range(len(cols)):
            nz = np.nonzero(cols[i])[0]
            if len(nz):
                cols[i, nz[-1]] = 0.0
        totals = released - dead
        keep = cols.sum(axis=1) > 0
        if np.any(totals[keep] - cols[keep].sum(axis=1) < -1e-9):
            raise ValueError("a group has fewer females at risk than confirmed captures")
        return years[keep], births[keep], totals[keep], cols[keep], deltas[keep]


    def _likelihood(theta, eff, terminal_age, model="late"):
        """Negative log-likelihood of the usable records under a candidate model."""
        import numpy as np

        years, births, totals, cols, deltas = eff
        s0, sp, k, extra, p = _unpack(theta)
        if not (0.0 < p < 1.0) or k <= 0.0 or extra <= 0.0:
            return 1e12
        k_cells = cols.shape[1]
        ages = np.minimum(years - births - 1, terminal_age).astype(float)
        surv = np.ones(len(years))
        probs = np.empty((len(years), k_cells))
        max_delta = int(deltas.max())
        for t in range(k_cells):
            lo = deltas[:, t - 1] if t else np.zeros(len(years))
            hi = deltas[:, t]
            for u in range(max_delta):
                surv = np.where(
                    (lo <= u) & (u < hi),
                    surv * _curve(ages + u, s0, sp, k, extra, terminal_age, model),
                    surv,
                )
            probs[:, t] = surv * p * (1.0 - p) ** t
        if np.any(probs < 0.0) or np.any(probs.sum(axis=1) > 1.0 + 1e-9):
            return 1e12
        never = 1.0 - probs.sum(axis=1)
        counts = np.concatenate([cols, (totals - cols.sum(axis=1))[:, None]], axis=1)
        vals = np.clip(np.concatenate([probs, never[:, None]], axis=1), 1e-12, 1.0)
        return -float((counts * np.log(vals)).sum())


    def _nelder_mead(fun, x0, step=0.25, max_iter=4000, tol=1e-10):
        """Deterministic derivative-free minimization (numpy only)."""
        import numpy as np

        n = len(x0)
        simplex = [np.asarray(x0, dtype=float).copy()]
        for i in range(n):
            x = np.asarray(x0, dtype=float).copy()
            x[i] += step
            simplex.append(x)
        simplex = np.array(simplex)
        fvals = np.array([fun(x) for x in simplex])
        for _ in range(max_iter):
            order = np.argsort(fvals)
            simplex, fvals = simplex[order], fvals[order]
            if np.max(np.abs(simplex[1:] - simplex[0])) < tol:
                break
            centroid = simplex[:-1].mean(axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = fun(xr)
            if fr < fvals[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = fun(xe)
                simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = fun(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                    fvals[1:] = np.array([fun(x) for x in simplex[1:]])
        order = np.argsort(fvals)
        return simplex[order[0]]


    m = np.asarray(release_records)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] < 4 + 2:
        raise ValueError("release_records must be a non-empty 2-D array with at "
                         "least six columns")
    if not np.all(np.isfinite(m)):
        raise ValueError("release_records entries must be finite")
    if np.any(m != np.round(m)):
        raise ValueError("release_records entries must be integer-valued")
    if np.any(m < 0):
        raise ValueError("release_records entries must be non-negative")
    if np.any(m[:, 0] <= m[:, 1]):
        raise ValueError("a release year must follow the birth year")
    if np.any(m[:, 4:].sum(axis=1) > m[:, 2]):
        raise ValueError("a release group cannot be recaptured more often than released")
    if not isinstance(terminal_age, (int, np.integer)) or isinstance(terminal_age, bool) \
            or terminal_age < 2:
        raise ValueError("terminal_age must be an integer of at least 2")

    eff = _apply_rules(m)
    ages = np.minimum(eff[0] - eff[1] - 1, int(terminal_age))
    if any(a not in ages for a in range(int(terminal_age) + 1)):
        raise ValueError("every survival class must appear among the usable release groups")

    r = np.asarray(reproduction_records)
    if r.ndim != 2 or r.shape[1] != 5 or r.size == 0:
        raise ValueError("reproduction_records must be a non-empty 2-D array with 5 columns")
    if not np.all(np.isfinite(r)):
        raise ValueError("reproduction_records entries must be finite")
    if np.any(r != np.round(r)):
        raise ValueError("reproduction_records entries must be integer-valued")
    if np.any(r < 0):
        raise ValueError("reproduction_records entries must be non-negative")
    if np.any(r[:, 3] > r[:, 2]):
        raise ValueError("more females cannot give birth than were present")

    T = int(terminal_age)
    candidates = {
        "late": [(0.60, 0.90, 0.8, 0.10, 0.80),
                 (0.50, 0.85, 0.4, 0.20, 0.70),
                 (0.70, 0.95, 1.5, 0.05, 0.85)],
        "logistic": [(0.60, 0.90, 0.8, 2.0, 0.80),
                     (0.50, 0.85, 1.2, 3.0, 0.70),
                     (0.70, 0.95, 0.5, 1.0, 0.85)],
    }
    selected = None
    for model, starts in candidates.items():
        for s0, sp, k, extra, p0 in starts:
            x0 = np.array([np.log(s0 / (1 - s0)), np.log(sp / (1 - sp)),
                           np.log(p0 / (1 - p0)), np.log(k), np.log(extra)])
            xb = _nelder_mead(
                lambda th: _likelihood(th, eff, T, model), x0, max_iter=8000)
            f = _likelihood(xb, eff, T, model)
            if selected is None or f < selected[0]:
                selected = (f, xb, model)
    s0, sp, k, extra, p = _unpack(selected[1])
    phi = [float(_curve(a, s0, sp, k, extra, T, selected[2])) for a in range(T + 1)]

    fec_obs = {a: [] for a in range(T + 1)}
    for pulse_year, birth_year, present, _gave, fawns in r:
        age = pulse_year - birth_year
        if age < 2:
            continue
        fec_obs[min(int(age), T)].append((present, fawns * 0.5))
    fecundity = [0.0, 0.0]
    for a in range(2, T + 1):
        pairs = fec_obs[a]
        if not pairs:
            raise ValueError(f"no reproduction records for age class {a}")
        fecundity.append(round(sum(n for _p, n in pairs) / sum(p for p, _n in pairs), 3))

    survival = [round(float(v), 3) for v in phi]
    return np.array(survival, dtype=float), np.array(fecundity, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        estimate_rates(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_estimate_rates(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    common = ("import numpy as np\n"
              "REC = np.array([[2002, 2000, 50, 30, 45],\n"
              "                [2003, 2000, 40, 24, 36],\n"
              "                [2003, 2001, 40, 24, 36],\n"
              "                [2004, 2002, 60, 40, 60]], dtype=int)\n")
    return [
        # Normal: the three rules bite. One group's last capture is unconfirmed, one
        # group has recoveries, and one group is left with no confirmed capture.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 100, 10, 40, 5, 1],\n"
                  "                [1991, 1988, 80, 8, 38, 5, 0],\n"
                  "                [1992, 1988, 60, 6, 34, 0, 0],\n"
                  "                [1991, 1990, 120, 12, 48, 6, 1],\n"
                  "                [1992, 1990, 90, 9, 43, 5, 0],\n"
                  "                [1991, 1989, 70, 7, 0, 0, 0]], dtype=int)\n" + common,
         "call": "estimate_rates(REL, REC, terminal_age=2)",
         "gold_call": "_oracle_estimate_rates(REL, REC, terminal_age=2)"},
        # Normal: the data select the logistic-curve model, so a solver that always
        # uses the late-decline model fails this case.
        {"setup": "import numpy as np\n"
                  "REL = np.array([\n"
                  "    [2000, 1999, 200, 21, 80, 13, 2, 0],\n"
                  "    [2001, 1999, 200, 10, 125, 22, 4, 1],\n"
                  "    [2002, 1999, 200, 5, 142, 26, 5, 1],\n"
                  "    [2003, 1999, 200, 4, 148, 28, 5, 1],\n"
                  "    [2004, 1999, 200, 3, 149, 28, 5, 1],\n"
                  "    [2005, 1999, 200, 4, 149, 27, 5, 1],\n"
                  "    [2006, 1999, 200, 4, 147, 26, 4, 1],\n"
                  "    [2007, 1999, 200, 6, 142, 24, 4, 0],\n"
                  "    [2008, 1999, 200, 9, 134, 20, 3, 0],\n"
                  "    [2009, 1999, 200, 13, 120, 15, 1, 0],\n"
                  "    [2010, 1999, 200, 18, 100, 10, 1, 0],\n"
                  "    [2011, 1999, 200, 24, 76, 5, 0, 0],\n"
                  "    [2012, 1999, 200, 29, 52, 2, 0, 0],\n"
                  "    [2013, 1999, 200, 33, 32, 1, 0, 0],\n"
                  "    ], dtype=int)\n"
                  "REC = np.array([[2000 + age, 2000, 50, 35, 60] for age in range(2, 14)], dtype=int)\n",
         "call": "estimate_rates(REL, REC)",
         "gold_call": "_oracle_estimate_rates(REL, REC)"},
        # Boundary: the terminal class pools every older age.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1989, 100, 12, 40, 6, 1, 0],\n"
                  "                [1990, 1988, 100, 10, 42, 5, 1, 0],\n"
                  "                [1990, 1987, 90, 9, 45, 6, 1, 0],\n"
                  "                [1991, 1987, 80, 8, 44, 6, 1, 0],\n"
                  "                [1992, 1986, 70, 7, 38, 6, 1, 0]], dtype=int)\n" + common,
         "call": "estimate_rates(REL, REC, terminal_age=3)",
         "gold_call": "_oracle_estimate_rates(REL, REC, terminal_age=3)"},
        # Invalid: a negative count.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 100, 10, -40, 5]], dtype=int)\n" + common + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, terminal_age=2)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, terminal_age=2)"},
        # Invalid: more first recaptures than females released.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 10, 0, 40, 5]], dtype=int)\n" + common + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, terminal_age=2)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, terminal_age=2)"},
        # Invalid: the rules leave fewer females at risk than confirmed captures.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 10, 9, 4, 2, 1]], dtype=int)\n" + common + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, terminal_age=2)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, terminal_age=2)"},
        # Invalid: a survival class never appears among the usable groups.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 100, 10, 40, 5, 1]], dtype=int)\n" + common + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, terminal_age=3)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, terminal_age=3)"},
        # Invalid: a non-integer count.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 100, 10, 40.5, 5]], dtype=float)\n" + common + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, terminal_age=2)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, terminal_age=2)"},
        # Invalid: more females give birth than were present.
        {"setup": "import numpy as np\n"
                  "REL = np.array([[1990, 1988, 100, 10, 40, 5],\n"
                  "                [1991, 1988, 100, 10, 40, 5],\n"
                  "                [1991, 1990, 100, 10, 40, 5]], dtype=int)\n"
                  "REC = np.array([[2002, 2000, 30, 40, 45]], dtype=int)\n" + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, terminal_age=2)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, terminal_age=2)"},
    ]
