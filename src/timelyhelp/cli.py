"""Explicit research stages. Isaac pilots partition simulation across isolated GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config, validate_config, write_json


def doctor(output=None):
    import platform
    import sys
    import numpy as np
    import torch
    from importlib.metadata import version

    root = Path(__file__).resolve().parents[2]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "core_training_device": "cpu",
        "isaac_python": str(root / ".local/isaacsim/python.sh"),
        "isaac_installed": (root / ".local/isaacsim/python.sh").is_file(),
        "gazebo_installed": (
            root
            / ".local/gazebo/usr/lib/python3/dist-packages/gz/sim8.cpython-310-x86_64-linux-gnu.so"
        ).is_file(),
        "packages": {name: version(name) for name in ("scipy", "matplotlib", "pytest", "PyYAML")},
    }
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    report["gpus"] = process.stdout.splitlines() if process.returncode == 0 else []
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report["pip_check"] = {"passed": check.returncode == 0, "output": check.stdout + check.stderr}
    if output:
        write_json(output, report)
    return report


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="TimelyHelp reproducible research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Read-only environment compatibility report")
    check.add_argument("--output")
    for name in ("pipeline", "generate", "train", "evaluate", "report", "transfer"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/smoke.yaml")
        command.add_argument("--output", required=True)
        command.add_argument("--backend", choices=("reference", "isaac", "gazebo"))
        if name in ("train", "evaluate", "report"):
            command.add_argument("--dataset", required=name == "train")
        if name in ("evaluate", "transfer", "report"):
            command.add_argument("--models", required=name in ("evaluate", "transfer"))
        if name == "report":
            command.add_argument("--records", required=True)
    audit = commands.add_parser("native-smoke", help="Short native physics/sensor/replay audit")
    audit.add_argument("--backend", choices=("isaac", "gazebo"), required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--steps", type=int, default=3)
    live = commands.add_parser("live", help="SSH browser viewer: bounded Isaac demos, one per GPU")
    from .live import parse_gpus

    live.add_argument("--gpus", type=parse_gpus, default=(0, 1, 2, 3))
    live.add_argument("--port", type=int, default=8765)
    live.add_argument("--output", default="artifacts/live_demo")
    live.add_argument("--duration", type=float, default=18)
    live.add_argument("--session-minutes", type=float, default=60)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(args.output), indent=2))
        return 0
    if args.command == "live":
        from .live import serve_live

        return serve_live(args.gpus, args.port, args.output, args.duration, args.session_minutes)
    if args.command == "native-smoke":
        from .backends.native import native_smoke

        report = native_smoke(args.backend, args.output, steps=args.steps)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    from .experiments import (
        generate_data,
        train_pipeline,
        evaluate_pipeline,
        report_pipeline,
        run_pipeline,
    )

    config = load_config(args.config)
    if args.backend:
        config["backend"] = args.backend
        validate_config(config)
    if args.command == "pipeline":
        result = run_pipeline(config, args.output)
    elif args.command == "generate":
        result = generate_data(config, args.output)
    elif args.command == "train":
        result = train_pipeline(config, args.dataset, args.output)
    elif args.command in ("evaluate", "transfer"):
        if args.command == "transfer":
            config["transfer"] = True
            result = evaluate_pipeline(config, args.models, args.output, transfer=True)
        else:
            result = evaluate_pipeline(config, args.models, args.output, dataset=args.dataset)
    else:
        result = report_pipeline(
            config, args.records, args.output, counterfactual_rows=args.dataset, models=args.models
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
