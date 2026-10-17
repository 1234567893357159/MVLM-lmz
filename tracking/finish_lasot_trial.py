"""Finish a trial whose inference already completed: validate predictions and
compute metrics, without re-running inference.

Usage:
    python tracking/finish_lasot_trial.py <exp_id>

Writes prediction_validation.json, metrics.txt, eval_data.pkl and updates
status.json / experiments.jsonl for <exp_id>.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "lasot_parameter_trials"
WEIGHT = ROOT / "models" / "MVLM_TF.pth.tar"


def timestamp():
    import datetime
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz).isoformat()


def append_record(row):
    with (OUT / "experiments.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    exp = sys.argv[1]
    artifact = OUT / exp
    result = ROOT / "output" / "test" / "tracking_results" / "mvlm" / exp / "lasot"
    assert artifact.exists(), artifact
    assert result.exists(), result

    row = json.loads((artifact / "status.json").read_text())

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg")

    try:
        from lib.test.evaluation import get_dataset
        dataset = get_dataset("lasot")
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

        analysis = [sys.executable, "-u", "-m", "tracking.analysis_results",
                    "--tracker_name", "mvlm", "--tracker_param", exp,
                    "--exp_id", exp, "--dataset", "lasot"]
        with (artifact / "metrics.txt").open("w") as metrics_file:
            subprocess.run(analysis, cwd=str(ROOT), env=env, stdout=metrics_file,
                           stderr=subprocess.STDOUT, check=True)
        text = (artifact / "metrics.txt").read_text()
        match = re.search(re.escape(exp) + r"\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", text)
        assert match, "Metrics not found"
        metrics = dict(zip(("PRE", "NPR", "AUC"), map(float, match.groups())))

        row.update(status="completed", completed_at=timestamp(), metrics=metrics,
                   validated_sequences=len(dataset),
                   validated_frames=sum(len(s.frames) for s in dataset),
                   recovered_by="finish_lasot_trial.py (inference had already completed)")
        row.pop("error", None)
        row.pop("traceback", None)
        shutil.copy2(ROOT / "output" / "test" / "result_plots" / "lasot" / "eval_data.pkl",
                     artifact / "eval_data.pkl")
        p = row["parameters"]
        print("RESULT %s | a=%g tau=%g w=%g l=%g psi=%g | PRE=%.2f NPR=%.2f AUC=%.2f"
              % (exp, p["ALPHA_CORR"], p["TAU"], p["OMEGA"], p["LAMBDA"], p["PSI_OUT"],
                 metrics["PRE"], metrics["NPR"], metrics["AUC"]))
    except BaseException as exc:
        row.update(status="failed", completed_at=timestamp(), error=repr(exc),
                   traceback=traceback.format_exc())
        print("FAILED %s: %s" % (exp, repr(exc)))
    finally:
        (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        append_record(dict(row, event="finished", at=timestamp()))


if __name__ == "__main__":
    main()
