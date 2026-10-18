"""Run ONE isolated full-TNL2K inference-parameter trial.

Usage:
    python tracking/run_tnl2k_trial_one.py <label> <ALPHA_CORR> <TAU> <OMEGA> <LAMBDA> <PSI_OUT> "<rationale>"

Mirrors run_lasot_trial_one.py / run_otb_trial_one.py but for the tnl2k dataset
(700 seqs / 516,038 frames). Starts from the ORIGINAL published MVLM block, NOT
the tuned mvlm_TF.yaml or the repo's mvlm_TF_TNL2K.yaml.

WARNING: this is a TNL2K test-set parameter search. Results must never be
reported as an independent reproduction of the paper.
"""
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "tnl2k_parameter_trials"
BASELINE = OUT / "tnl2k_language_baseline.yaml"
WEIGHT = ROOT / "models" / "MVLM_TF.pth.tar"

KEYS = ("ALPHA_CORR", "TAU", "OMEGA", "LAMBDA", "PSI_OUT")
ORIGINAL = {"ALPHA_CORR": 1.0, "TAU": 0.1, "OMEGA": 0.4, "LAMBDA": 0.4, "PSI_OUT": 0.0}
DATASET = "tnl2k"


def timestamp():
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz).isoformat()


def patch_config(base_text, params):
    data = yaml.safe_load(base_text)
    assert data["TEST"]["MVLM"] == ORIGINAL, data["TEST"]["MVLM"]
    for key, value in params.items():
        base_text, count = re.subn(
            r"(?m)^(    " + re.escape(key) + r":)[^\n]*$",
            lambda m: m[1] + " " + str(value),
            base_text,
        )
        assert count == 1, key
    assert yaml.safe_load(base_text)["TEST"]["MVLM"] == params
    return base_text


def append_record(row):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "experiments.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) < 8:
        sys.exit(__doc__)
    label = sys.argv[1]
    values = [float(x) for x in sys.argv[2:7]]
    rationale = sys.argv[7]
    params = dict(zip(KEYS, values))

    base_text = BASELINE.read_text()

    batch = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = "tnl2k_tune_" + label
    artifact = OUT / exp
    result = ROOT / "output" / "test" / "tracking_results" / "mvlm" / exp / DATASET
    if artifact.exists() or result.exists():
        sys.exit("REFUSING: %s already exists (use a new label)" % exp)

    artifact.mkdir(parents=True)
    result.mkdir(parents=True)

    config_text = patch_config(base_text, params)
    (ROOT / "experiments" / "mvlm" / (exp + ".yaml")).write_text(config_text)
    (artifact / "config.yaml").write_text(config_text)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               OPENCV_FOR_THREADS_NUM="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    test = [sys.executable, "-u", "-m", "tracking.test", "mvlm", exp,
            "--dataset_name", DATASET, "--weight_path", str(WEIGHT),
            "--exp_id", exp, "--mode", "mp", "--threads", "16", "--num_gpus", "8"]
    analysis = [sys.executable, "-u", "-m", "tracking.analysis_results",
                "--tracker_name", "mvlm", "--tracker_param", exp,
                "--exp_id", exp, "--dataset", DATASET]

    row = {"label": label, "exp_id": exp, "batch": batch, "parameters": params,
           "rationale": rationale, "status": "testing", "started_at": timestamp(),
           "artifact_dir": str(artifact), "results_dir": str(result),
           "commands": [test, analysis],
           "weight_sha256": None,
           "warning": "TNL2K test-set parameter trial; NOT an independent reproduction result."}
    import hashlib
    h = hashlib.sha256()
    with WEIGHT.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    row["weight_sha256"] = h.hexdigest()
    (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    append_record(dict(row, event="running", at=timestamp()))

    try:
        with (artifact / "test.log").open("w") as log:
            subprocess.run(test, cwd=str(ROOT), env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

        from lib.test.evaluation import get_dataset
        dataset = get_dataset(DATASET)
        errors = []
        for seq in dataset:
            try:
                boxes = np.loadtxt(result / (seq.name + ".txt"))
                times = np.loadtxt(result / (seq.name + "_time.txt"))
                assert boxes.shape == (len(seq.frames), 4), boxes.shape
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
                assert times.shape == (len(seq.frames),), times.shape
                assert np.isfinite(times).all() and (times >= 0).all()
            except Exception as exc:
                errors.append({"sequence": seq.name, "error": repr(exc)})
        (artifact / "prediction_validation.json").write_text(
            json.dumps({"checked": len(dataset), "errors": errors}, indent=2) + "\n")
        assert not errors, "Prediction validation failed: %s" % errors[:3]

        with (artifact / "metrics.txt").open("w") as mf:
            subprocess.run(analysis, cwd=str(ROOT), env=env, stdout=mf,
                           stderr=subprocess.STDOUT, check=True)
        text = (artifact / "metrics.txt").read_text()
        match = re.search(re.escape(exp) + r"\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", text)
        assert match, "Metrics not found"
        metrics = dict(zip(("PRE", "NPR", "AUC"), map(float, match.groups())))
        row.update(status="completed", completed_at=timestamp(), metrics=metrics,
                   validated_sequences=len(dataset),
                   validated_frames=sum(len(s.frames) for s in dataset))
        pkl = ROOT / "output" / "test" / "result_plots" / DATASET / "eval_data.pkl"
        if pkl.exists():
            shutil.copy2(pkl, artifact / "eval_data.pkl")
    except BaseException as exc:
        row.update(status="failed", completed_at=timestamp(), error=repr(exc),
                   traceback=traceback.format_exc())
    finally:
        (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        append_record(dict(row, event="finished", at=timestamp()))

    if row["status"] == "completed":
        p, m = row["parameters"], row["metrics"]
        print("RESULT %s | a=%g tau=%g w=%g l=%g psi=%g | PRE=%.2f NPR=%.2f AUC=%.2f  (w*l=%g)"
              % (exp, p["ALPHA_CORR"], p["TAU"], p["OMEGA"], p["LAMBDA"], p["PSI_OUT"],
                 m["PRE"], m["NPR"], m["AUC"], p["OMEGA"] * p["LAMBDA"]))
    else:
        print("FAILED %s: %s" % (exp, row.get("error")))
        sys.exit(1)


if __name__ == "__main__":
    main()
