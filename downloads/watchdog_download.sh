#!/bin/bash
# Read current dataset stages and bytes; never unlink a live lock file.
exec /root/miniconda3/envs/mvlm/bin/python /root/MVLM/downloads/watchdog_training.py "$@"
