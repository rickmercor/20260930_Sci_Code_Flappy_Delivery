"""
Construct the partial-fraction coefficients of the normalized [P/P] approximation to the forward-envelope exponential for dimensionless t=k0*delta_z. Return arrays b and d with shapes (P,) and (P+1,). The represented rational function, not a particular ordering of its matched terms, defines the answer. Include the identity limit at t=0; invalid inputs or a numerically unresolved finite representation raise ValueError as specified in the Signature.

1. For f_t(x)=exp(i*t*(sqrt(1+x)-1)), impose Q(0)=1 and Q*f_t-N=O(x**(2*P+1)). A simple root r_j of Q gives b_j=-1/r_j and d[j+1]=b_j*N(r_j)/Q'(r_j), with d[0] the leading-coefficient ratio. The residue includes b_j. Constructing coefficients and resolving poles may require internal higher precision, particularly at high P; ordinary finite complex arrays are returned. Taylor matching in x does not imply order 2P in propagation distance.

Returns
-------
return np.zeros(P, dtype=complex), np.zeros(P + 1, dtype=complex)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def PadeCoeff(P, t):
    """Construct normalized [P/P] Pade partial fractions at x=0.

    Approximate f(x)=exp(1j*t*(sqrt(1+x)-1)), with Q(0)=1 and
    Q(x)*f(x)-N(x)=O(x**(2*P+1)). The input t=k0*h is dimensionless.
    P must be a positive integer and t a finite real or complex scalar.
    Return finite complex arrays b and d of shapes (P,) and (P+1,), with
    R(x)=d[0]+sum(d[j+1]/(1+b[j]*x)). Simultaneously permuting b and d[1:]
    is equivalent. At t=0 return b=0, d[0]=1, and d[1:]=0.
    For nonzero t the tested regimes have simple denominator roots and a
    finite partial-fraction representation. Raise ValueError for invalid
    inputs or a numerically unresolved representation. Internal higher
    precision is permitted; return ordinary NumPy complex arrays.
    The reference uses NumPy and Python's standard library only.
    """
    return np.zeros(P, dtype=complex), np.zeros(P + 1, dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_PadeCoeff(P, t):
    """Return high-precision Pade partial fractions without extra packages.

    Decimal real/imaginary pairs retain internal precision. Taylor
    coefficients satisfy (1+x)*f'' + f'/2 + t**2*f/4 = 0, with
    f(0) = 1 and f'(0) = 1j*t/2. The normalized denominator is obtained
    by Taylor matching, then its simple poles are refined with Newton.
    """
    from decimal import Decimal, localcontext

    if (
        isinstance(P, (bool, np.bool_))
        or not isinstance(P, (int, np.integer))
        or P < 1
    ):
        raise ValueError("P must be a positive integer, not a boolean.")
    raw_parameter = np.asarray(t)
    if raw_parameter.ndim != 0 or raw_parameter.dtype.kind not in "iufc":
        raise ValueError("t must be a finite real or complex scalar.")
    try:
        parameter = complex(raw_parameter)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "t must be representable as a complex scalar."
        ) from error
    if not np.isfinite(parameter):
        raise ValueError("t must be finite.")

    order = int(P)
    if parameter == 0:
        poles = np.zeros(order, dtype=complex)
        residues = np.zeros(order + 1, dtype=complex)
        residues[0] = 1.0
        return poles, residues

    with localcontext() as context:
        context.prec = max(80, 12 * order)

        def number(real=0, imaginary=0):
            return np.array([Decimal(real), Decimal(imaginary)], dtype=object)

        def multiply(first, second):
            return np.array(
                [
                    first[0] * second[0] - first[1] * second[1],
                    first[0] * second[1] + first[1] * second[0],
                ],
                dtype=object,
            )

        def divide(first, second):
            squared_norm = second[0]**2 + second[1]**2
            conjugate = np.array([second[0], -second[1]], dtype=object)
            return multiply(first, conjugate) / squared_norm

        def evaluate(coefficients, point):
            value = coefficients[-1].copy()
            for coefficient in reversed(coefficients[:-1]):
                value = multiply(value, point) + coefficient
            return value

        try:
            scaled_parameter = number(parameter.real, parameter.imag)
            series = [number(1), multiply(number(0, 1), scaled_parameter) / 2]
            squared_parameter = (
                multiply(scaled_parameter, scaled_parameter) / 4
            )
            for degree in range(2, 2 * order + 1):
                factor = Decimal(degree - 1) * (
                    Decimal(degree) - Decimal("1.5")
                )
                series.append(
                    -(
                        factor * series[-1]
                        + multiply(squared_parameter, series[-2])
                    ) / Decimal(degree * (degree - 1))
                )

            coefficients = np.array(
                [
                    [series[order + row - column] for column in range(order)]
                    for row in range(order)
                ],
                dtype=object,
            )
            real = coefficients[:, :, 0]
            imaginary = coefficients[:, :, 1]
            system = np.block([[real, -imaginary], [imaginary, real]])
            right_hand_side = np.array(
                [-value[0] for value in series[order + 1:]]
                + [-value[1] for value in series[order + 1:]],
                dtype=object,
            )
            augmented = np.column_stack((system, right_hand_side))
            size = 2 * order
            for column in range(size):
                pivot = max(
                    range(column, size),
                    key=lambda row: abs(augmented[row, column]),
                )
                if augmented[pivot, column] == 0:
                    raise ValueError("The Pade matching system is singular.")
                augmented[[column, pivot]] = augmented[[pivot, column]]
                augmented[column, column:] /= augmented[column, column]
                for row in range(column + 1, size):
                    augmented[row, column:] -= (
                        augmented[row, column] * augmented[column, column:]
                    )

            solution = np.empty(size, dtype=object)
            for row in reversed(range(size)):
                solution[row] = augmented[row, -1] - sum(
                    augmented[row, row + 1:size] * solution[row + 1:]
                )
            denominator = np.vstack(
                (
                    number(1),
                    np.column_stack((solution[:order], solution[order:])),
                )
            )
            numerator = np.array(
                [
                    sum(
                        (
                            multiply(
                                denominator[index], series[degree - index]
                            )
                            for index in range(degree + 1)
                        ),
                        number(),
                    )
                    for degree in range(order + 1)
                ],
                dtype=object,
            )
            derivative = np.array(
                [
                    degree * denominator[degree]
                    for degree in range(1, order + 1)
                ],
                dtype=object,
            )
            guesses = np.roots(
                np.array([complex(*value) for value in denominator[::-1]])
            )
            if guesses.size != order or not np.all(np.isfinite(guesses)):
                raise ValueError("The Pade poles cannot be represented.")

            tolerance = Decimal(10)**(20 - context.prec)
            roots = []
            for guess in guesses:
                root = number(float(guess.real), float(guess.imag))
                for _iteration in range(150):
                    correction = divide(
                        evaluate(denominator, root), evaluate(derivative, root)
                    )
                    root -= correction
                    if max(abs(correction)) <= (
                        tolerance * max(1, max(abs(root)))
                    ):
                        break
                else:
                    raise ValueError("A Pade pole could not be refined.")
                if (
                    max(abs(root)) == 0
                    or max(abs(evaluate(derivative, root))) == 0
                ):
                    raise ValueError("Simple finite Pade poles are required.")
                if any(
                    max(abs(root - previous))
                    <= tolerance.sqrt() * max(1, max(abs(root)))
                    for previous in roots
                ):
                    raise ValueError(
                        "Distinct Pade poles could not be resolved."
                    )
                roots.append(root)

            decimal_poles = [-divide(number(1), root) for root in roots]
            decimal_residues = [divide(numerator[-1], denominator[-1])] + [
                multiply(
                    pole,
                    divide(
                        evaluate(numerator, root), evaluate(derivative, root)
                    ),
                )
                for pole, root in zip(decimal_poles, roots)
            ]
            poles = np.array([complex(*value) for value in decimal_poles])
            residues = np.array(
                [complex(*value) for value in decimal_residues]
            )
        except (
            ArithmeticError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            raise ValueError(
                "The Pade partial-fraction representation could not "
                "be computed."
            ) from error

    if not np.all(np.isfinite(poles)) or not np.all(np.isfinite(residues)):
        raise ValueError(
            "The Pade coefficients must be finite complex arrays."
        )
    return poles, residues

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "\nimport numpy as np\n\n\ndef represented_pade(P, t):\n    b, d = PadeCoeff(P, t)\n\n    # Evaluate the represented rational function at fixed test points.\n    # This is invariant under simultaneous permutation of b[j] and d[j+1].\n    z_values = np.array(\n        [\n            0.0,\n            0.01,\n            0.1,\n            0.5,\n            -0.5,\n            0.1 + 0.2j,\n            -0.25 + 0.15j\n        ],\n        dtype=complex\n    )\n\n    values = np.empty(len(z_values), dtype=complex)\n\n    for i, z in enumerate(z_values):\n        values[i] = d[0] + np.sum(\n            d[1:] / (1.0 + b * z)\n        )\n\n    return values\n\n\nP = 1\nt = 10.0\n\n\ndef run_model():\n    return represented_pade(P, t)\n\n\ndef run_gold():\n    b, d = _oracle_PadeCoeff(P, t)\n\n    z_values = np.array(\n        [\n            0.0,\n            0.01,\n            0.1,\n            0.5,\n            -0.5,\n            0.1 + 0.2j,\n            -0.25 + 0.15j\n        ],\n        dtype=complex\n    )\n\n    values = np.empty(len(z_values), dtype=complex)\n\n    for i, z in enumerate(z_values):\n        values[i] = d[0] + np.sum(\n            d[1:] / (1.0 + b * z)\n        )\n\n    return values\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "\nimport numpy as np\n\n\ndef represented_pade(P, t):\n    b, d = PadeCoeff(P, t)\n\n    z_values = np.array(\n        [\n            0.0,\n            0.01,\n            0.1,\n            0.5,\n            -0.5,\n            0.1 + 0.2j,\n            -0.25 + 0.15j\n        ],\n        dtype=complex\n    )\n\n    values = np.empty(len(z_values), dtype=complex)\n\n    for i, z in enumerate(z_values):\n        values[i] = d[0] + np.sum(\n            d[1:] / (1.0 + b * z)\n        )\n\n    return values\n\n\nP = 9\nt = 10.0\n\n\ndef run_model():\n    return represented_pade(P, t)\n\n\ndef run_gold():\n    b, d = _oracle_PadeCoeff(P, t)\n\n    z_values = np.array(\n        [\n            0.0,\n            0.01,\n            0.1,\n            0.5,\n            -0.5,\n            0.1 + 0.2j,\n            -0.25 + 0.15j\n        ],\n        dtype=complex\n    )\n\n    values = np.empty(len(z_values), dtype=complex)\n\n    for i, z in enumerate(z_values):\n        values[i] = d[0] + np.sum(\n            d[1:] / (1.0 + b * z)\n        )\n\n    return values\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "\nimport numpy as np\n\n\ndef represented_pade(P, t):\n    b, d = PadeCoeff(P, t)\n\n    # A small but nonzero t avoids the degenerate t=0\n    # root/residue representation. The test points remain bounded\n    # away from the branch point z = -1.\n    z_values = np.array(\n        [\n            0.0,\n            1.0e-4,\n            1.0e-3,\n            1.0e-2,\n            0.05,\n            0.01 + 0.01j\n        ],\n        dtype=complex\n    )\n\n    values = np.empty(len(z_values), dtype=complex)\n\n    for i, z in enumerate(z_values):\n        values[i] = d[0] + np.sum(\n            d[1:] / (1.0 + b * z)\n        )\n\n    return values\n\n\nP = 9\nt = 1.0e-6\n\n\ndef run_model():\n    return represented_pade(P, t)\n\n\ndef run_gold():\n    b, d = _oracle_PadeCoeff(P, t)\n\n    z_values = np.array(\n        [\n            0.0,\n            1.0e-4,\n            1.0e-3,\n            1.0e-2,\n            0.05,\n            0.01 + 0.01j\n        ],\n        dtype=complex\n    )\n\n    values = np.empty(len(z_values), dtype=complex)\n\n    for i, z in enumerate(z_values):\n        values[i] = d[0] + np.sum(\n            d[1:] / (1.0 + b * z)\n        )\n\n    return values\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "\nimport numpy as np\n\n\ndef represented_pade_benchmark(P, t_values):\n    z_values = np.array(\n        [\n            0.0,\n            0.01,\n            0.05,\n            0.1,\n            0.25,\n            0.5,\n            0.1 + 0.2j,\n            -0.25 + 0.15j\n        ],\n        dtype=complex\n    )\n\n    output = []\n\n    for t in t_values:\n        b, d = PadeCoeff(P, t)\n\n        values = np.empty(\n            len(z_values),\n            dtype=complex\n        )\n\n        for i, z in enumerate(z_values):\n            values[i] = d[0] + np.sum(\n                d[1:] / (1.0 + b * z)\n            )\n\n        output.append(values)\n\n    return np.concatenate(output)\n\n\nP = 4\n\n# Includes small, moderate, and propagation-relevant values.\n# For the stated physical parameters, k0*h is approximately pi/2.\nt_values = np.array(\n    [\n        0.25,\n        0.5,\n        np.pi / 4.0,\n        np.pi / 2.0,\n        2.0\n    ],\n    dtype=float\n)\n\n\ndef run_model():\n    return represented_pade_benchmark(P, t_values)\n\n\ndef run_gold():\n    z_values = np.array(\n        [\n            0.0,\n            0.01,\n            0.05,\n            0.1,\n            0.25,\n            0.5,\n            0.1 + 0.2j,\n            -0.25 + 0.15j\n        ],\n        dtype=complex\n    )\n\n    output = []\n\n    for t in t_values:\n        b, d = _oracle_PadeCoeff(P, t)\n\n        values = np.empty(\n            len(z_values),\n            dtype=complex\n        )\n\n        for i, z in enumerate(z_values):\n            values[i] = d[0] + np.sum(\n                d[1:] / (1.0 + b * z)\n            )\n\n        output.append(values)\n\n    return np.concatenate(output)\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "\nimport numpy as np\n\npoints = np.array([0j, (0.01+0j), (0.1+0j), (0.5+0j), (-0.5+0j), (0.1+0.2j), (-0.25+0.15j)], dtype=complex)\nexpected = np.array([(1+0j), (1+0j), (1+0j), (1+0j), (1+0j), (1+0j), (1+0j)], dtype=complex)\n\ndef represented():\n    poles, residues = PadeCoeff(4, 0.0)\n    poles = np.asarray(poles)\n    residues = np.asarray(residues)\n    if poles.shape != (4,) or residues.shape != (5,):\n        raise AssertionError(\"Incorrect coefficient shapes.\")\n    return residues[0] + np.sum(residues[1:, None] / (1 + poles[:, None] * points), axis=0)\n",
            "call": "represented()",
            "gold_call": "expected",
        },
        {
            "setup": "\nimport numpy as np\n\npoints = np.array([0j, (0.01+0j), (0.1+0j), (0.5+0j), (-0.5+0j), (0.1+0.2j), (-0.25+0.15j)], dtype=complex)\nexpected = np.array([(1+0j), (0.9999001293662778+0.0004987064414623825j), (0.9990123994697322+0.004876103173837455j), (0.9952637815077832+0.022371809599682095j), (1.0054436306421928-0.02945716754494769j), (0.9894920213832584+0.0033748920256202568j), (0.9938870668109002-0.014604519277649736j)], dtype=complex)\n\ndef represented():\n    poles, residues = PadeCoeff(4, (0.1+0.02j))\n    poles = np.asarray(poles)\n    residues = np.asarray(residues)\n    if poles.shape != (4,) or residues.shape != (5,):\n        raise AssertionError(\"Incorrect coefficient shapes.\")\n    return residues[0] + np.sum(residues[1:, None] / (1 + poles[:, None] * points), axis=0)\n",
            "call": "represented()",
            "gold_call": "expected",
        },
        {
            "setup": "\nimport numpy as np\n\npoints = np.array([0j, (0.01+0j), (0.1+0j), (0.5+0j), (-0.5+0j), (0.1+0.2j), (-0.25+0.15j)], dtype=complex)\nexpected = np.array([(1+0j), (0.999969268392919+0.00783978760747777j), (0.9970583233089035+0.07664659105555853j), (0.9382454485343792+0.34597034309969643j), (0.8958777816837442-0.4443005742573529j), (0.8583443040382758+0.07180859962684899j), (0.8552281082202535-0.17681121846706493j)], dtype=complex)\n\ndef represented():\n    poles, residues = PadeCoeff(4, 1.5718837664637613)\n    poles = np.asarray(poles)\n    residues = np.asarray(residues)\n    if poles.shape != (4,) or residues.shape != (5,):\n        raise AssertionError(\"Incorrect coefficient shapes.\")\n    return residues[0] + np.sum(residues[1:, None] / (1 + poles[:, None] * points), axis=0)\n",
            "call": "represented()",
            "gold_call": "expected",
        },
    ]
