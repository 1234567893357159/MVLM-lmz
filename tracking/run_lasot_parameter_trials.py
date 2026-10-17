"""Run isolated, full-LaSOT inference-parameter trials and retain all artifacts.

This is deliberately a test-set parameter search.  It must not be presented as
an independent reproduction result.
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


def timestamp():
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz).isoformat()


def load_trials():
    baseline = json.loads((BASELINE / "status.json").read_text())
    return [
        ("00_original", baseline["config"]["TEST"]["MVLM"],
         "已完成的原始发布参数基线；为避免重复消耗计算资源，只归档其可复核的已有结果。"),
        ("01_otb_best", {"ALPHA_CORR": 0.7, "TAU": 0.03, "OMEGA": 0.2,
                         "LAMBDA": 0.4, "PSI_OUT": 0.1},
         "OTB99-Lang 上最高 AUC 的配置，先检验跨数据集迁移。"),
        ("02_more_memory", {"ALPHA_CORR": 0.7, "TAU": 0.03, "OMEGA": 0.4,
                             "LAMBDA": 0.4, "PSI_OUT": 0.1},
         "只将 OMEGA 从 0.2 调到 0.4，隔离当前决策对时间记忆的依赖。"),
        ("03_no_overlap_distractor", {"ALPHA_CORR": 0.7, "TAU": 0.03, "OMEGA": 0.2,
                                       "LAMBDA": 0.4, "PSI_OUT": 0.0},
         "只移除重叠干扰候选框，隔离 PSI_OUT=0.1 的作用。"),
        ("04_tau_zero", {"ALPHA_CORR": 0.7, "TAU": 0.0, "OMEGA": 0.2,
                          "LAMBDA": 0.4, "PSI_OUT": 0.1},
         "只把阈值降为零，检验更少触发全局重定位的影响。"),
        ("05_tau_six", {"ALPHA_CORR": 0.7, "TAU": 0.06, "OMEGA": 0.2,
                         "LAMBDA": 0.4, "PSI_OUT": 0.1},
         "只提高阈值，检验更强候选筛选和全局重定位的影响。"),
    ]


def patch_config(base_text, params):
    data = yaml.safe_load(base_text)
    assert data["TEST"]["MVLM"] == {"ALPHA_CORR": 1.0, "TAU": 0.1,
                                       "OMEGA": 0.4, "LAMBDA": 0.4, "PSI_OUT": 0.0}
    for key, value in params.items():
        base_text, count = re.subn(r"(?m)^(    " + re.escape(key) + r":)[^\n]*$",
                                    lambda match: match[1] + " " + str(value), base_text)
        assert count == 1, key
    assert yaml.safe_load(base_text)["TEST"]["MVLM"] == params
    return base_text


def write_summary(records):
    lines = ["# LaSOT 推理参数试验记录", "",
             "本目录的非基线结果是在 LaSOT 测试集上调参获得，不能作为独立复现结果。",
             "所有新试验使用相同官方权重、280 条序列、685,360 帧、8 张 RTX 3090 和 16 个测试进程。",
             "", "| 试验 | 状态 | αcorr | τ | ω | λ | ψout | PRE (%) | NPR (%) | AUC (%) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        params = row["parameters"]
        metrics = row.get("metrics", {})
        values = [row["label"], row["status"], *(str(params[key]) for key in
                  ("ALPHA_CORR", "TAU", "OMEGA", "LAMBDA", "PSI_OUT")),
                  *(str(metrics.get(key, "—")) for key in ("PRE", "NPR", "AUC"))]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "按 AUC 最高、PRE 作为同分判别选择本轮最佳参数。每组的 YAML、日志、指标、预测校验和命令均保存在对应子目录。"])
    (OUT / "调参总结.md").write_text("\n".join(lines) + "\n")


def run_trial(batch, label, params, rationale, base_text, runtime):
    exp = "lasot_" + batch + "_" + label
    artifact = OUT / exp
    artifact.mkdir(parents=True, exist_ok=False)
    result = ROOT / "output" / "test" / "tracking_results" / "mvlm" / exp / "lasot"
    result.mkdir(parents=True, exist_ok=False)
    config_text = patch_config(base_text, params)
    config_path = ROOT / "experiments" / "mvlm" / (exp + ".yaml")
    config_path.write_text(config_text)
    (artifact / "config.yaml").write_text(config_text)
    (artifact / "runtime.yaml").write_text(RUNTIME.read_text())
    workers = int(runtime["num_gpus"]) * int(runtime["workers_per_gpu"])
    env = dict(os.environ, OMP_NUM_THREADS=str(runtime["cpu_threads_per_worker"]),
               MKL_NUM_THREADS=str(runtime["cpu_threads_per_worker"]),
               OPENBLAS_NUM_THREADS=str(runtime["cpu_threads_per_worker"]),
               NUMEXPR_NUM_THREADS=str(runtime["cpu_threads_per_worker"]),
               OPENCV_FOR_THREADS_NUM=str(runtime["cpu_threads_per_worker"]),
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    test = [sys.executable, "-u", "-m", "tracking.test", "mvlm", exp,
            "--dataset_name", "lasot", "--weight_path", str(WEIGHT), "--exp_id", exp,
            "--mode", "mp", "--threads", str(workers), "--num_gpus", str(runtime["num_gpus"])]
    analysis = [sys.executable, "-u", "-m", "tracking.analysis_results", "--tracker_name", "mvlm",
                "--tracker_param", exp, "--exp_id", exp, "--dataset", "lasot"]
    row = {"batch": batch, "label": label, "exp_id": exp, "parameters": params,
           "rationale": rationale, "status": "testing", "started_at": timestamp(),
           "artifact_dir": str(artifact), "results_dir": str(result),
           "commands": [test, analysis], "runtime": runtime,
           "warning": "LaSOT test-set parameter trial; not an independent reproduction result."}
    (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    try:
        with (artifact / "test.log").open("w") as log:
            subprocess.run(test, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
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
        (artifact / "prediction_validation.json").write_text(json.dumps({"checked": len(dataset), "errors": errors}, indent=2) + "\n")
        assert not errors, "Prediction validation failed"
        with (artifact / "metrics.txt").open("w") as metrics_file:
            subprocess.run(analysis, cwd=ROOT, env=env, stdout=metrics_file,
                           stderr=subprocess.STDOUT, check=True)
        metrics_text = (artifact / "metrics.txt").read_text()
        match = re.search(re.escape(exp) + r"\s*\|\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", metrics_text)
        assert match, "Metrics not found"
        row.update(status="completed", completed_at=timestamp(), validated_sequences=len(dataset),
                   validated_frames=sum(len(seq.frames) for seq in dataset),
                   metrics=dict(zip(("PRE", "NPR", "AUC"), map(float, match.groups()))))
        shutil.copy2(ROOT / "output" / "test" / "result_plots" / "lasot" / "eval_data.pkl",
                     artifact / "eval_data.pkl")
    except BaseException as exc:
        row.update(status="failed", completed_at=timestamp(), error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        (artifact / "status.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batch = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime = yaml.safe_load(RUNTIME.read_text())
    assert runtime["num_gpus"] == 8 and runtime["workers_per_gpu"] == 2
    base_text = SOURCE_CONFIG.read_text()
    records = []
    for label, params, rationale in load_trials():
        if label == "00_original":
            baseline = json.loads((BASELINE / "status.json").read_text())
            records.append({"label": label, "parameters": params, "rationale": rationale,
                            "status": "completed", "metrics": baseline["metrics_percent"],
                            "artifact_dir": str(BASELINE), "exp_id": baseline["exp_id"]})
            write_summary(records)
            continue
        row = run_trial(batch, label, params, rationale, base_text, runtime)
        records.append(row)
        write_summary(records)
    winner = max((row for row in records if row["label"] != "00_original" and row["status"] == "completed"),
                 key=lambda row: (row["metrics"]["AUC"], row["metrics"]["PRE"]))
    summary = {"batch": batch, "completed_at": timestamp(), "criterion": "highest AUC; PRE breaks ties",
               "best_new_trial": winner, "records": records,
               "warning": "Test-set tuning only; do not report as independent reproduction."}
    (OUT / (batch + "_summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_summary(records)


if __name__ == "__main__":
    main()
