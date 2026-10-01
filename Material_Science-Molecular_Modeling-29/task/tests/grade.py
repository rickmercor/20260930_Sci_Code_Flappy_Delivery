#!/usr/bin/env python3
"""Grade /app/solution.py against gold targets. Writes /logs/verifier/reward.json.
Paths overridable via env vars (SOLUTION_PATH/TESTDATA_PATH/SPEC_PATH/REWARD_PATH)
so this file can be exercised locally, outside the container."""
import os, json, traceback
import numpy as np
import h5py

SOLUTION = os.environ.get("SOLUTION_PATH", "/app/solution.py")
TESTDATA = os.environ.get("TESTDATA_PATH", "/tests/test_data.h5")
SPEC     = os.environ.get("SPEC_PATH", "/tests/spec.json")
REWARD   = os.environ.get("REWARD_PATH", "/logs/verifier/reward.json")

# numpy defaults rtol=1e-05, atol=1e-08; tolerate tuple/list vs ndarray shapes
_orig_allclose = np.allclose
def _robust_allclose(a, b, rtol=1e-05, atol=1e-08):
    try:
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(
                _robust_allclose(x, y, rtol, atol) for x, y in zip(a, b))
        if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
            aa, bb = np.asarray(a), np.asarray(b)
            return aa.shape == bb.shape and bool(_orig_allclose(aa, bb, rtol=rtol, atol=atol))
        return bool(_orig_allclose(a, b, rtol=rtol, atol=atol))
    except Exception:
        return False
np.allclose = _robust_allclose

def _targets(h5, step_number, n_tests):
    out = []
    for i in range(1, n_tests + 1):
        g = h5[f"{step_number}/test{i}"]
        keys = sorted(g.keys())
        out.append(g[keys[0]][()] if len(keys) == 1 else tuple(g[k][()] for k in keys))
    return out

def main():
    result = {"reward": 0.0, "steps_total": 0, "steps_passed": 0, "steps": []}
    try:
        spec = json.load(open(SPEC))
        code = open(SOLUTION).read() if os.path.exists(SOLUTION) else ""
        steps = spec["sub_steps"]
        result["steps_total"] = len(steps)
        with h5py.File(TESTDATA, "r") as h5:
            all_pass = bool(steps)
            for s in steps:
                sn, tcs = s["step_number"], s["test_cases"]
                step_ok, err = True, None
                base = {}
                try:
                    exec(code, base)
                except Exception as e:  # solution failed to import/parse
                    step_ok, err, base = False, f"import: {e!r}", None
                if base is not None:
                    tgts = _targets(h5, sn, len(tcs))
                    for i, tc in enumerate(tcs):
                        ns = dict(base); ns["np"] = np; ns["target"] = tgts[i]
                        try:
                            exec(tc, ns)
                        except Exception as e:
                            step_ok, err = False, repr(e); break
                result["steps"].append({"step": sn, "passed": step_ok, "error": err})
                if step_ok:
                    result["steps_passed"] += 1
                else:
                    all_pass = False
            result["reward"] = 1.0 if all_pass else 0.0
    except Exception as e:
        result["error"] = f"{e!r}\n{traceback.format_exc()}"
        result["reward"] = 0.0
    outdir = os.path.dirname(REWARD) or "."
    os.makedirs(outdir, exist_ok=True)
    # reward.json / reward.txt must be schema-clean for Harbor's VerifierResult
    # (rewards = dict[str, float]); keep only numeric keys here.
    clean = {"reward": float(result["reward"]),
             "steps_passed": float(result["steps_passed"]),
             "steps_total": float(result["steps_total"])}
    json.dump(clean, open(REWARD, "w"))
    with open(os.path.join(outdir, "reward.txt"), "w") as f:
        f.write(str(float(result["reward"])))
    # per-step diagnostics (NOT read by the verifier schema)
    try:
        json.dump(result.get("steps", []),
                  open(os.path.join(outdir, "steps_detail.json"), "w"))
    except Exception:
        pass
    print(json.dumps(clean))

main()
