"""Explicit opt-in reuse of a previous full audit for memory-world C training."""
import json
import os
from pathlib import Path
import sys
import torch

import train_memory_world as trainer


def cached_manifest(path, clip_length):
    cache_path = Path(os.environ["MVLM_AUDIT_CACHE"]).resolve()
    manifest_hash = trainer.sha256(path)
    payload = torch.load(str(cache_path), map_location="cpu")
    if payload.get("manifest_hash") != manifest_hash:
        raise ValueError("Audit cache manifest hash differs")
    baseline = sys.argv[sys.argv.index("--baseline") + 1]
    if payload.get("baseline_hash") != trainer.sha256(baseline):
        raise ValueError("Audit cache baseline hash differs")
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    provenance = {"mode": "reuse_previous_full_audit", "cache_path": str(cache_path),
                  "cache_sha256": trainer.sha256(cache_path), "manifest_sha256": manifest_hash,
                  "cached_data_signature": payload.get("data_signature"),
                  "clip_length": clip_length,
                  "limitation": "Current video paths, annotations and language were not re-audited."}
    (output / "cache_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Reusing previously audited dataset cache; current dataset files are not re-audited.", flush=True)
    return payload["partitions"], manifest_hash


trainer.read_manifest = cached_manifest
if __name__ == "__main__":
    trainer.main()
