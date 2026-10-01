"""
Step 1 - the reactive-bond site table from the ensemble extension records.

The KIMMDY method models reactions over an ensemble of reactive sites, each of which can undergo homolytic bond scission (force-accelerated) or hydrolysis (base-catalysed). A site is specified by its bond label, its dissociation energy E_dis (kJ/mol), its effective force constant k (kJ/mol/nm^2), and the local pulling force F it experiences (nN). The force is not quoted directly: the simulations recorded each bond extension frame by frame, and the force of a frame is the restoring force of that bond own Morse potential at the recorded extension. 

Two instruments produced the records: 

* instrument A reports extensions in nm
* instrument B reports them in Angstrom. 

A frame whose force evaluation did not converge is recorded as exactly 0 and marks the end of a replica equilibration only the production window after the last such reading is used. A bond may be locally compressed in an individual frame: negative extensions are physical, they enter the restoring force with their sign, and the mean force of a tensile network stays positive. A replica draws contamination when its mean force deviates from the median replica force by more than one tenth of that median, and such a replica is discarded before the bond force is formed. 

This step validates the records and returns the resulting site table.

Returns
-------
triples : np.ndarray, shape (n, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def site_table(records: "list[tuple]") -> "np.ndarray":
    """Validate the ensemble records and return the site table.

    Parameters
    ----------
    records : list of tuple
        Each entry is (label, E_dis, k, instrument, replicas): label str, dissociation
        energy E_dis in kJ/mol (positive), force constant k in kJ/mol/nm^2 (positive),
        instrument either "A" (extensions recorded in nm) or "B" (Angstrom), and
        `replicas`, a non-empty sequence of replica trajectories. Each trajectory is a
        non-empty 1-D sequence of extension readings, in the instrument length unit and
        in time order, finite and signed; a reading of exactly 0.0 marks a frame whose
        force evaluation did not converge.

    Returns
    -------
    np.ndarray
        Shape (n, 3), float array of [E_dis, k, F] rows, in input order: F is the bond's
        ensemble force in nN, the mean of its surviving replica production means,
        rounded to two decimals. Labels are validated but the numeric triple is returned.

    Raises
    ------
    ValueError
        If the record list is empty, if any entry is not a 5-tuple, if the label is not
        a string, if E_dis or k is not finite or not positive, if the instrument is
        neither "A" nor "B", if a replica is empty or not one-dimensional, if any
        reading is not finite, if a replica has no production reading after its last
        non-converged reading, or if every replica is discarded as contaminated.
    """
    return triples  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_site_table(records: "list[tuple]") -> "np.ndarray":
    """Reference implementation for site_table."""
    import numpy as np

    out = []
    for entry in records:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 5):
            raise ValueError("each site record must be a 5-tuple "
                             "(label, E_dis, k, instrument, replicas)")
        label, E, k, instrument, replicas = entry
        if not isinstance(label, str):
            raise ValueError("site label must be a string")
        E = float(E); k = float(k)
        if not (np.isfinite(E) and np.isfinite(k)):
            raise ValueError("E_dis and k must be finite")
        if E <= 0.0:
            raise ValueError("E_dis must be positive")
        if k <= 0.0:
            raise ValueError("force constant must be positive")
        if instrument not in ("A", "B"):
            raise ValueError("instrument must be 'A' or 'B'")
        beta = float(np.sqrt(k / (2.0 * E)))
        unit = 1.0 if instrument == "A" else 0.1

        trajectories = list(replicas)
        if len(trajectories) == 0:
            raise ValueError("each site needs at least one replica")
        replica_forces = []
        for trajectory in trajectories:
            readings = np.asarray(trajectory, dtype=float)
            if readings.ndim != 1 or readings.size == 0:
                raise ValueError("each replica must be a non-empty 1-D sequence")
            if not np.all(np.isfinite(readings)):
                raise ValueError("extension readings must be finite")
            failures = np.flatnonzero(readings == 0.0)
            start = int(failures[-1]) + 1 if failures.size else 0
            production = readings[start:]
            if production.size == 0:
                raise ValueError("a replica has no production reading after its last "
                                 "non-converged reading")
            x = production * unit
            u = np.exp(-beta * x)
            frame_force = 2.0 * E * beta * u * (1.0 - u) * 1.66054     # pN, signed
            replica_forces.append(float(np.mean(frame_force)))
        median = float(np.median(replica_forces))
        surviving = [f for f in replica_forces if abs(f - median) <= 0.10 * abs(median)]
        if not surviving:
            raise ValueError("every replica is contaminated")
        bond_force = float(np.mean(surviving))
        out.append([E, k, float(np.round(bond_force / 1000.0, 2))])
    if not out:
        raise ValueError("site table must not be empty")
    return np.asarray(out, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    graded = setup + (
        "records = [\n"
        "    ('disulfide S-S', 251.0, 15954.0, 'A',\n"
        "     [[0.005954, 0, 0.025799, 0.02825],\n"
        "      [0.005954, 0, 0.02952, 0.030823, 0.031486, 0.031486],\n"
        "      [0.005954, 0, 0.012594, 0, 0.050917, 0.055129]]),\n"
        "    ('ester C-O', 280.0, 16082.0, 'B',\n"
        "     [[0.05889, 0, 0.3388],\n"
        "      [0.05889, 0, 0.3388, 0.36639],\n"
        "      [0.05889, 0, 0.1241, 0, 0.34556, 0.37352, 0.38075]]),\n"
        "    ('glycosidic C-O', 230.0, 27566.0, 'A',\n"
        "     [[0.003409, 0, -0.003159, 0.017358, 0.018031],\n"
        "      [0.003409, 0, 0.010065, 0.011192]]),\n"
        "    ('peroxide O-O', 170.0, 73368.0, 'B',\n"
        "     [[0.01266, 0, 0.07328],\n"
        "      [0.01266, 0, 0.07328, 0.07559, 0.07792],\n"
        "      [0.01266, 0, 0.02608, 0, 0.07559, 0.08029]]),\n"
        "    ('peptide C-N', 350.0, 22016.0, 'A',\n"
        "     [[0.004252, 0, 0.028469],\n"
        "      [0.004252, 0, 0.029388, 0.030323, 0.031275]]),\n"
        "    ('thioester C-S', 260.0, 45653.0, 'B',\n"
        "     [[0.02036, 0, 0.1517, 0.16009],\n"
        "      [0.02036, 0, 0.15797, 0.16653],\n"
        "      [0.02036, 0, 0.04198, 0, 0.16437, 0.1687, 0.1731]]),\n"
        "]\n")
    invalid = setup + (
        "def run_model(r):\n"
        "    try:\n"
        "        site_table(r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(r):\n"
        "    try:\n"
        "        _oracle_site_table(r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the graded six-bond records, two or three replicas each, one compressive
        # frame, two instruments, and multi-marker windows.
        {"setup": graded,
         "call": "site_table(records)",
         "gold_call": "_oracle_site_table(records)"},
        # Normal: the same physical extension read on instrument A (nm) and instrument B
        # (Angstrom) gives the same force.
        {"setup": setup + "records = [('a', 216.0, 7155.0, 'A', [[0.005, 0, 0.03, 0.05]]), "
                          "('b', 216.0, 7155.0, 'B', [[0.05, 0, 0.3, 0.5]])]\n",
         "call": "site_table(records)",
         "gold_call": "_oracle_site_table(records)"},
        # Normal: a compressive frame is physical and keeps its sign.
        {"setup": setup + "records = [('a', 216.0, 7155.0, 'A', [[0.01, 0, -0.02, 0.04]])]\n",
         "call": "site_table(records)",
         "gold_call": "_oracle_site_table(records)"},
        # Boundary: the third replica is contaminated and must leave the bond mean
        # (500, 520, 700 pN -> the 700 pN replica is dropped -> 510 pN).
        {"setup": setup + "records = [('a', 216.0, 7155.0, 'A', "
                          "[[0, 0.060864, 0.060864], [0, 0.064826, 0.064826], "
                          "[0, 0.125059, 0.125059]])]\n",
         "call": "site_table(records)",
         "gold_call": "_oracle_site_table(records)"},
        # Boundary: the production window opens after the LAST non-converged reading.
        {"setup": setup + "records = [('a', 216.0, 7155.0, 'A', "
                          "[[0.01, 0, 0.02, 0, 0.060864, 0.060864]])]\n",
         "call": "site_table(records)",
         "gold_call": "_oracle_site_table(records)"},
        # Invalid: a non-positive dissociation energy.
        {"setup": invalid,
         "call": "run_model([('a', 0.0, 7155.0, 'A', [[0.0, 0.03]])])",
         "gold_call": "run_gold([('a', 0.0, 7155.0, 'A', [[0.0, 0.03]])])"},
        # Invalid: an unknown instrument.
        {"setup": invalid,
         "call": "run_model([('a', 216.0, 7155.0, 'C', [[0.0, 0.03]])])",
         "gold_call": "run_gold([('a', 216.0, 7155.0, 'C', [[0.0, 0.03]])])"},
        # Invalid: a replica whose last reading is the non-converged marker.
        {"setup": invalid,
         "call": "run_model([('a', 216.0, 7155.0, 'A', [[0.03, 0.0]])])",
         "gold_call": "run_gold([('a', 216.0, 7155.0, 'A', [[0.03, 0.0]])])"},
        # Invalid: an entry that is not a 5-tuple.
        {"setup": invalid,
         "call": "run_model([('a', 216.0, 7155.0, 'A')])",
         "gold_call": "run_gold([('a', 216.0, 7155.0, 'A')])"},
        # Invalid: two replicas whose means sit far on either side of their median, so
        # both are contaminated and no bond force exists.
        {"setup": invalid,
         "call": "run_model([('a', 216.0, 7155.0, 'A', "
                  "[[0.008885, 0.008885], [0.125059, 0.125059]])])",
         "gold_call": "run_gold([('a', 216.0, 7155.0, 'A', "
                       "[[0.008885, 0.008885], [0.125059, 0.125059]])])"},
        # Invalid: a non-finite extension reading.
        {"setup": invalid,
         "call": "run_model([('a', 216.0, 7155.0, 'A', [[float('inf'), 0.03]])])",
         "gold_call": "run_gold([('a', 216.0, 7155.0, 'A', [[float('inf'), 0.03]])])"},
        # Invalid: an empty record list.
        {"setup": invalid,
         "call": "run_model([])",
         "gold_call": "run_gold([])"},
    ]
