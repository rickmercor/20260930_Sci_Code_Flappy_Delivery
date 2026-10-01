"""
Evaluate the paper's two-loop order-alpha^2 Z outer radiative correction ghat^(2)(beta, mubar) to the positron spectrum of a superallowed decay, with the Fermi function factored out, at a given velocity and renormalization scale.

At order $\alpha^2 Z$, one Coulomb exchange combined with an ultrasoft photon produces a two-loop outer correction. After factoring the Fermi function out of the rate, the positron-emission correction is $1+\alpha^2 Z\,\widehat g^{(2)}$. The coefficient depends on the lepton velocity and renormalization scale, but not on the neutrino energy.



For this coding step, the complete expression from the source's Eq. (3.3) is supplied here. Define

$$

b=\beta,\qquad q=\frac{1-b}{1+b},\qquad r=\sqrt q,\qquad

L=\log\frac{1+b}{1-b},\qquad L_\mu=\log\frac{\bar\mu^2}{m_e^2}.

$$

All logarithms are natural, $0<b<1$, and $\bar\mu$ and $m_e$ are positive and expressed in the same energy units. The real dilogarithm is

$$

\operatorname{Li}_2(x)=-\int_0^x\frac{\log(1-t)}{t}\,dt.

$$

For positron emission use the lower overall sign, namely

$$

\widehat g^{(2)}(b,\bar\mu)=-B(b,\bar\mu),

$$

where

$$

\begin{aligned}

B={}&-\left(\frac{1}{3b}+\frac12\right)L_\mu

+\frac{9-b^2}{2b}\operatorname{Li}_2(q)

+\frac{b^2-5}{b}\operatorname{Li}_2(r)

-\frac{1}{b}\operatorname{Li}_2(q^2)

+\frac{3-b^2}{2b}\frac{\pi^2}{6}\\

&-\frac{4-5b+b^3}{16b^2}L^2

+\frac{b^2-2}{b^2}\log(1+r)

-\frac{2b^2+2}{b^2}\log\frac{1+b}{2}\\

&+\frac{L}{12b^2}(-6+10b+3b^2+3b^3)-\frac4b\\

&+\frac{(1-r)^3(1+b)^2}{144b^4}

\left[r(430-220b-39b^2+48b^3)

+434-652b+327b^2-96b^3\right].

\end{aligned}

$$



The scale dependence of the positron coefficient is therefore **positive**: $(1/(3b)+1/2)L_\mu$. Its threshold expansion starts as

$$

\widehat g^{(2)}=

\left(\frac{1}{3b}+\frac12\right)L_\mu

+\frac{37}{8}-4\log2-\frac43 b+O(b^2).

$$

For $b\to1$, with $E_e=m_e/\sqrt{1-b^2}$, it approaches

$$

\frac56\log\frac{\bar\mu^2}{4E_e^2}

+\frac{131}{36}-\frac{\pi^2}{6}.

$$

Evaluate the full function throughout the stated domain. At small velocity, individually large terms cancel; a stable algebraic rearrangement or a sufficiently accurate series is appropriate. The leading threshold limit alone is not a replacement for the velocity-dependent function.

Returns
-------
float — the paper's two-loop alpha^2 Z outer correction function ghat^(2)(beta, mubar) for positron emission.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def g2_alpha2z(beta: float, mubar: float, m_e: float) -> float:
    '''Order-alpha^2 Z outer correction ghat^(2)(beta, mubar) for positron emission.

    Parameters
    ----------
    beta : float
        Positron velocity, 0 < beta < 1.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.

    Returns
    -------
    g2 : float
        The dimensionless function ghat^(2) for a positron emitter (the
        correction factor is 1 + alpha^2 Z ghat^(2)), as a native Python
        float.

    Raises
    ------
    ValueError
        If beta is outside (0, 1), or mubar or m_e is not strictly positive.
    '''
    return g2  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

from scipy.special import spence


def _oracle_g2_alpha2z(beta: float, mubar: float, m_e: float) -> float:
    beta = float(beta)
    mubar = float(mubar)
    m_e = float(m_e)
    if not 0.0 < beta < 1.0 or mubar <= 0.0 or m_e <= 0.0:
        raise ValueError("require 0 < beta < 1, mubar > 0, m_e > 0")

    b = beta
    log_two = math.log(2.0)
    if 0.5 * m_e <= mubar <= 1.5 * m_e:
        log_scale = 2.0 * math.log1p((mubar - m_e) / m_e)
    else:
        log_scale = 2.0 * (math.log(mubar) - math.log(m_e))
    scale_term = (log_scale / 3.0) / b + 0.5 * log_scale

    if b < 0.02:
        # Positron expansion of Eq. (3.3), excluding the exact scale term.
        # Terms through b**7 avoid the cancellation of the closed form.
        coefficients = (
            37.0 / 8.0 - 4.0 * log_two,
            -4.0 / 3.0,
            2.0 * log_two / 3.0 - 239.0 / 288.0,
            -4.0 / 15.0,
            2.0 * log_two / 15.0 - 169.0 / 1152.0,
            -106.0 / 315.0,
            2.0 * log_two / 35.0 + 371.0 / 15360.0,
            -104.0 / 315.0,
        )
        regular = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            regular = coefficient + b * regular
        result = scale_term + regular
    else:
        q = (1.0 - b) / (1.0 + b)
        r = math.sqrt(q)
        log_ratio = math.log1p(b) - math.log1p(-b)
        # 1 - r, rationalized to avoid loss of significance.
        one_minus_r = 2.0 * b / ((1.0 + b) * (1.0 + r))

        # scipy.special.spence(z) = Li_2(1-z).
        li_q = float(spence(1.0 - q))
        li_r = float(spence(1.0 - r))
        li_q2 = float(spence(1.0 - q * q))
        braces_without_scale = math.fsum(
            (
                (9.0 - b * b) / (2.0 * b) * li_q,
                (b * b - 5.0) / b * li_r,
                -li_q2 / b,
                (3.0 - b * b) / (2.0 * b) * math.pi**2 / 6.0,
                -(4.0 - 5.0 * b + b**3)
                / (16.0 * b * b) * log_ratio**2,
                (b * b - 2.0) / (b * b) * math.log1p(r),
                -(2.0 * b * b + 2.0) / (b * b)
                * (math.log1p(b) - log_two),
                log_ratio / (12.0 * b * b)
                * (-6.0 + 10.0 * b + 3.0 * b * b + 3.0 * b**3),
                -4.0 / b,
                one_minus_r**3 * (1.0 + b)**2 / (144.0 * b**4)
                * (
                    r * (430.0 - 220.0 * b - 39.0 * b * b + 48.0 * b**3)
                    + 434.0 - 652.0 * b + 327.0 * b * b - 96.0 * b**3
                ),
            )
        )
        result = scale_term - braces_without_scale
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent configurations; each reference call invokes this step once."""
    return [
        {
            # Case 1
            "setup": """
import math

m = 0.51099895
endpoint = 4.2327 - m
mu = 2.0 * endpoint * math.exp(-1.0)
beta = math.sqrt((2.0 - m) * (2.0 + m)) / 2.0
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 2
            "setup": """beta = 0.2
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 3
            "setup": """beta = 0.5
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 4
            "setup": """beta = 0.8
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 5
            "setup": """beta = 0.95
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 6
            "setup": """beta = 0.995
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 7
            "setup": """beta = 0.6
mu = 8.0
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 8
            "setup": """beta = 0.6
mu = 2.0
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 9
            "setup": """beta = 0.0001
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-08,
        },
        {
            # Case 10
            "setup": """beta = 0.999999
mu = 2.7
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-08,
        },
        {
            # Case 11
            "setup": """beta = 1e-08
mu = 0.51099895
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 12
            "setup": """beta = 1e-06
mu = 0.51099895
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 13
            "setup": """beta = 0.019999
mu = 0.51099895
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 14
            "setup": """beta = 0.020001
mu = 0.51099895
m = 0.51099895
""",
            'call': 'g2_alpha2z(beta, mu, m)',
            'gold_call': '_oracle_g2_alpha2z(beta, mu, m)',
            'tol': 1e-09,
        },
        {
            # Case 15
            "setup": """
def rejects_nonpositive_scale(function):
    try:
        function(0.5, 0.0, 0.51099895)
    except ValueError:
        return 1
    return 0
""",
            'call': 'rejects_nonpositive_scale(g2_alpha2z)',
            'gold_call': 'rejects_nonpositive_scale(_oracle_g2_alpha2z)',
            'tol': 1e-09,
        },
    ]
