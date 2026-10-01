"""
Compute the first-order Asimov median discovery significance of the on/off experiment from the background uncertainty.

The median discovery significance expected for a nominal signal strength $s$ is approximated by evaluating the significance on the Asimov data set, in which the observed counts are replaced by their expectation values, $n \to s + b$ and $m \to \tau b$. Substituting these into the first-order significance $Z = \sqrt{q_0}$ of the profile likelihood ratio gives

$$Z_A = \left\{ -2 \left[ (s + b) \ln \frac{s + (1 + \tau) b}{(1 + \tau)(s + b)} + \tau b \ln\!\left( 1 + \frac{s}{(1 + \tau) b} \right) \right] \right\}^{1/2} .$$

The background estimate from the control measurement, $\hat{b} = m/\tau$, has variance $\sigma_b^2 = b/\tau$ because the variance of a Poisson count equals its mean. Eliminating $\tau = b/\sigma_b^2$ expresses the same result through the standard deviation of the background estimate,

$$Z_A = \left\{ 2 \left[ (s + b) \ln \frac{(s + b)(b + \sigma_b^2)}{b^2 + (s + b)\sigma_b^2} - \frac{b^2}{\sigma_b^2} \ln\!\left( 1 + \frac{\sigma_b^2 s}{b (b + \sigma_b^2)} \right) \right] \right\}^{1/2} ,$$

which expands to $s / \sqrt{b + \sigma_b^2}$ when $s \ll b$ and $\sigma_b^2 \ll b$ and reverts to the known-background result $\sqrt{2[(s + b)\ln(1 + s/b) - s]}$ as $\sigma_b \to 0$. This is the signed likelihood-ratio root evaluated at the Asimov point, which is positive for any positive signal strength and zero for $s = 0$. The two closed forms are algebraically identical but not numerically equivalent. As $\sigma_b$ becomes small the second term of the $\sigma_b$ form is a large prefactor times the logarithm of a number close to one, and the result has to be evaluated in a form that keeps its accuracy all the way to the known-background limit.

Returns
-------
float, the first-order Asimov significance Z_A, zero at s = 0 and accurate to relative error 1e-9 for the stated input domain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_order_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    r"""Return the first-order Asimov median significance in terms of the background uncertainty.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    sigma_b : float
        Standard deviation of the background estimate inferred from the
        control measurement, finite and above zero.

    Returns
    -------
    z_asimov : float
        The first-order Asimov significance Z_A as a native Python float,
        zero for s equal to zero, accurate to a relative error of 1e-9 for
        every valid sigma_b, including sigma_b far below b, where it tends
        to the known-background value. A round-off negative value of the
        bracket under the square root is treated as zero.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or sigma_b is not finite or
        not above zero.
    """
    return z_asimov

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _asimov_log_series(coeff):
    # coeff[0]=1. If A'=A L', solve order by order for L=log(A).
    result=[0.0]*len(coeff)
    for n in range(1,len(coeff)):
        result[n]=coeff[n]-sum(k*result[k]*coeff[n-k] for k in range(1,n))/n
    return result

def _asimov_poly(coeff, x):
    result=0.0
    for c in reversed(coeff):
        result=result*x+c
    return result

def _asimov_small_signal(s, b, sigma_b, degree=14):
    tau=b/(sigma_b*sigma_b)
    t=s/b
    a=1/(1+tau)
    scale=math.sqrt(b*tau/(1+tau))
    # r^2 = b*tau/(1+tau) * t^2 * F(t).
    # F_j=2 (-1)^j (1+a+...+a^j)/((j+1)(j+2)).
    f=[2*(-1)**j*sum(a**k for k in range(j+1))/((j+1)*(j+2)) for j in range(degree+1)]
    lf=_asimov_log_series(f)
    ll=_asimov_log_series([(-1)**j/(j+1) for j in range(degree+1)])
    # log(u/r)=C(t); evaluate C(t)/t directly, without forming u/r.
    c=[0.0]+[.5*(-1)**(j+1)*(1-a**j)/j+ll[j]-.5*lf[j] for j in range(1,degree+1)]
    c[1]=(tau-1)/(6*(1+tau))
    froot=math.sqrt(_asimov_poly(f,t))
    r=t*scale*froot
    rstar=r+_asimov_poly(c[1:],t)/(scale*froot)
    return r,rstar


def _oracle_first_order_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    s = float(s)
    b = float(b)
    sigma_b = float(sigma_b)
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError("s must be a finite signal strength of at least zero")
    if not (math.isfinite(b) and b > 0.0):
        raise ValueError("b must be a finite background above zero")
    if not (math.isfinite(sigma_b) and sigma_b > 0.0):
        raise ValueError("sigma_b must be a finite standard deviation above zero")

    # Evaluate through the scale factor tau = b / sigma_b^2. The second term is
    # tau b ln(1 + s / ((1 + tau) b)), whose logarithm of a number close to one
    # is taken with log1p so that the result stays accurate as sigma_b -> 0.
    if s / b <= 0.01:
        return float(_asimov_small_signal(s, b, sigma_b)[0])
    tau = b / (sigma_b * sigma_b)
    term_on = (s + b) * math.log((s + (1.0 + tau) * b) / ((1.0 + tau) * (s + b)))
    term_off = tau * b * math.log1p(s / ((1.0 + tau) * b))
    return float(math.sqrt(max(-2.0 * (term_on + term_off), 0.0)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "first_order_asimov_significance"),
                           ("run_gold", "_oracle_first_order_asimov_significance")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # the benchmark configuration, a relative background uncertainty of one hundred percent
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 0.8\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # a configuration in the regime where the correction to s / sqrt(b + sigma_b^2) is moderate
        {
            "setup": "s, b, sigma_b = 10.0, 20.0, 2.0\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # a well-constrained background, close to the known-background limit
        {
            "setup": "s, b, sigma_b = 3.0, 2.0, 0.02\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # boundary: no signal, where the significance is zero
        {
            "setup": "s, b, sigma_b = 0.0, 1.5, 0.5\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
            "tol": 1e-6,
        },
        # edge: a background uncertainty far larger than the background itself
        {
            "setup": "s, b, sigma_b = 4.0, 0.5, 3.0\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: a large signal over a small background, far from the limiting formula
        {
            "setup": "s, b, sigma_b = 30.0, 0.4, 0.4\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: a background uncertainty one millionth of the background, close to the known-background limit
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 8.0e-7\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: a background uncertainty one billionth of the background, indistinguishable from the known-background value
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 8.0e-10\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: large expected counts with a background known to half a percent
        {
            "setup": "s, b, sigma_b = 300.0, 2000.0, 10.0\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: large expected counts with a background uncertainty one millionth of the background
        {
            "setup": "s, b, sigma_b = 300.0, 2000.0, 2.0e-3\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: a large signal over a small background with a tiny background uncertainty
        {
            "setup": "s, b, sigma_b = 30.0, 0.4, 4.0e-7\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        # edge: a small signal over a large background with a tiny background uncertainty
        {
            "setup": "s, b, sigma_b = 3.0, 5000.0, 5.0e-4\n",
            "call": "first_order_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_first_order_asimov_significance(s, b, sigma_b)",
        },
        {
            "setup": "# invalid: a background uncertainty of zero, outside the uncertain-background model\n"
                     "args = (5.0, 0.8, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative signal strength\n"
                     "args = (-0.3, 0.8, 0.8)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative background uncertainty\n"
                     "args = (5.0, 0.8, -0.8)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a background of zero\n"
                     "args = (5.0, 0.0, 0.8)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ] + [{'setup': 's, b, sigma_b = 1e-6, 1.0, 1.0\n', 'call': 'first_order_asimov_significance(s, b, sigma_b) / s', 'gold_call': '_oracle_first_order_asimov_significance(s, b, sigma_b) / s', 'tol': 1e-09}, {'setup': 's, b, sigma_b = 1e-4, 0.8, 0.8\n', 'call': 'first_order_asimov_significance(s, b, sigma_b) / s', 'gold_call': '_oracle_first_order_asimov_significance(s, b, sigma_b) / s', 'tol': 1e-09}]
