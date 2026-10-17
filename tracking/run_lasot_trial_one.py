"""Run ONE isolated full-LaSOT inference-parameter trial.

Usage:
    python tracking/run_lasot_trial_one.py <label> <ALPHA_CORR> <TAU> <OMEGA> <LAMBDA> <PSI_OUT> "<rationale>"

Writes artifacts under output/lasot_parameter_trials/<exp_id>/ and appends a
machine-readable record to output/lasot_parameter_trials/experiments.jsonl.
Designed for adaptive, one-trial-at-a-time search: the caller inspects the
result before choosing the next configuration.

WARNING: this is a LaSOT test-set parameter search. Results must never be
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
OUT = ROOT / "output" / "lasot_parameter_trials"
BASELINE = ROOT / "output" / "lasot_evaluations" / "lasot_language_original_mp16_20261017_103923"
RUNTIME = ROOT / "tracking" / "lasot_runtime.yaml"
SOURCE_CONFIG = ROOT / "experiments" / "mvlm" / "lasot_language_original_mp16_20261017_103923.yaml"
WEIGHT = ROOT / "models" / "MVLM_TF.pth.tar"

KEYS = ("ALPHA_CORR", "TAU", "OMEGA", "LAMBDA", "PSI_OUT")
ORIGINAL = {"ALPHA_CORR": 1.0, "TAU": 0.1, "OMEGA": 0.4, "LAMBDA": 0.4, "PSI_OUT": 0.0}


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


def fmt(v):
    return ("%g" % v)


def main():
    if len(sys.argv) < 8:
        sys.exit(__doc__)
    label = sys.argv[1]
    values = [float(x) for x in sys.argv[2:7]]
    rationale = sys.argv[7]
    params = dict(zip(KEYS, values))

    runtime = yaml.safe_load(RUNTIME.read_text())
    workers = int(runtime["num_gpus"]) * int(runtime["workers_per_gpu"])
    base_text = SOURCE_CONFIG.read_text()

    batch = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = "lasot_tune_" + label
    artifact = OUT / exp
    result = ROOT / "output" / "test" / "tracking_results" / "mvlm" / exp / "lasot"
    if artifact.exists() or result.exists():
        sys.exit("REFUSING: %s already exists (use a new label)" % exp)

    artifact.mkdir(parents=True)
    result.mkdir(parents=True)

    config_text = patch_config(base_text, params)
    (ROOT / "experiments" / "mvlm" / (exp + ".yaml")).write_text(config_text)
    (artifact / "config.yaml").write_text(config_text)
    (artifact / "runtime.yaml").write_text(RUNTIME.read_text())

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1",
        PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
    )
    test = [sys.executable, "-u", "-m", "tracking.test", "mvlm", exp,
            "--dataset_name", "lasot", "--weight_path", str(WEIGHT), "--exp_id", exp,
            "--mode", "mp", "--threads", str(workers), "--num_gpus", str(runtime["num_gpus"])]
    analysis = [sys.executable, "-u", "-m", "tracking.analysis_results", "--tracker_name", "mvlm",
                "--tracker_param", exp, "--exp_id", exp, "--dataset", "lasot"]

    row = {
        "label": label, "exp_id": exp, "batch": batch, "parameters": params,
        "rationale": rationale, "status": "testing", "started_at": timestamp(),
        "artifact_dir": str(artifact), "results_dir": str(result),
        "commands": [test, analysis], "runtime": runtime,
        "warning": "LaSOT test-set parameter trial; NOT an independent reproduction result.",
    }
    (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    append_record(dict(row, event="running", at=timestamp()))

    try:
        with (artifact / "test.log").open("w") as log:
            subprocess.run(test, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        # subprocess.run above does not affect our sys.path, but be explicit and
        # robust: make sure the repo root is importable regardless of cwd.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from lib.test.evaluation import get_dataset
        dataset = get_dataset("lasot")
        errors = []
        for seq in dataset:
            try:
                boxes = np.loadtxt(result / (seq.name + ".txt"))
                times = np.loadtxt(result / (seq.name + "_time.txt"))
                assert boxes.shape == (len(seq.frames), 4)
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
                assert times.shape == (len(seq.frames),) and np.isfinite(times).all() and (times >= 0).all()
            except Exception as exc:
                errors.append({"sequence": seq.name, "error": repr(exc)})
        (artifact / "prediction_validation.json").write_text(
            json.dumps({"checked": len(dataset), "errors": errors}, indent=2) + "\n")
        assert not errors, "Prediction validation failed"

        with (artifact / "metrics.txt").open("w") as metrics_file:
            subprocess.run(analysis, cwd=ROOT, env=env, stdout=metrics_file,
                           stderr=subprocess.STDOUT, check=True)
        metrics_text = (artifact / "metrics.txt").read_text()
        match = re.search(re.escape(exp) + r"\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", metrics_text)
        assert match, "Metrics not found in analysis output"
        metrics = dict(zip(("PRE", "NPR", "AUC"), map(float, match.groups())))
        row.update(status="completed", completed_at=timestamp(),
                   validated_sequences=len(dataset),
                   validated_frames=sum(len(seq.frames) for seq in dataset),
                   metrics=metrics)
        shutil.copy2(ROOT / "output" / "test" / "result_plots" / "lasot" / "eval_data.pkl",
                     artifact / "eval_data.pkl")
    except BaseException as exc:
        row.update(status="failed", completed_at=timestamp(),
                   error=repr(exc), traceback=traceback.format_exc())
    finally:
        (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        append_record(dict(row, event="finished", at=timestamp()))

    if row["status"] == "completed":
        m = row["metrics"]
        print("RESULT %s | a=%s tau=%s w=%s l=%s psi=%s | PRE=%.2f NPR=%.2f AUC=%.2f"
              % (exp, fmt(params["ALPHA_CORR"]), fmt(params["TAU"]), fmt(params["OMEGA"]),
                 fmt(params["LAMBDA"]), fmt(params["PSI_OUT"]),
                 m["PRE"], m["NPR"], m["AUC"]))
        sys.exit(0)
    else:
        print("FAILED %s: %s" % (exp, row.get("error")))
        sys.exit(1)


if __name__ == "__main__":
    main()
