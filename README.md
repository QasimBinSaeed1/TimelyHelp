# TimelyHelp

Reproducible research software for **controller-conditioned scheduling of delayed
assistance** during differential-drive material transport around workers and carts.
The specification is [TimelyHelp_Research_Brief.md](TimelyHelp_Research_Brief.md).
Implementation status, scientific changes, test evidence and resume steps are in
[Implementation.md](Implementation.md); component mapping is in
[docs/Methodology.md](docs/Methodology.md).

The pipeline implements controlled A/W/Q continuations, signed information and
motion labels, a GRU ensemble, calibrated conservative source/motion scheduling,
all baselines and ablations, independent split audits, closed-loop evaluation,
useful windows, hierarchical intervals and report figures. Smoke results verify
software operation. They do not establish the central research hypothesis.

Verified on this server: **171 unit/integration tests pass**, lint/format checks pass,
both native physics/RGB-D/contact/replay audits pass, and a one-epoch Isaac gate runs
frozen in Gazebo across three layouts. Final implementation evidence is in
[Implementation.md](Implementation.md#final-validation-evidence-and-reproducible-state).
The authorized pilot was **resumed on GPUs 0–3 on 15 October 2026** in tmux
session `timelyhelp-pilot`. It retained 159/200 completed generation episodes;
all 159 episodes / 477 conditioning snapshots reproduced exactly after the native
capture fix. The remaining 41 episodes are scheduled before model training.
See the recovery entry in `Implementation.md`. Confirmatory experiments remain off.
No performance or hypothesis result is claimed from the pilot while it is in progress.
Machine-readable evidence from the completed local checks is in
[artifacts/verification_summary.json](artifacts/verification_summary.json).
The later live-viewer tests and four-GPU checks are documented separately below
and in the implementation ledger; original experiment artifacts retain their hashes.

## Setup on this server

```bash
cd /home/binsaeeq@eng.umanitoba.ca/TimelyHelp
bash scripts/setup_environment.sh
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate "$PWD/.conda"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
timelyhelp doctor --output artifacts/environment.json
```

The isolated environment is `.conda`, Python 3.11 and PyTorch 2.8 CPU. Training the
small supervisor does not require CUDA. The server's NVIDIA driver and other GPU
jobs are left under the server's existing management. Exact Python and Conda package
locks are under `docs/locks/`. To recreate their installed versions instead of
resolving dependencies again:

```bash
conda create --prefix "$PWD/.conda" --file docs/locks/conda-linux-64.txt -y
.conda/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r docs/locks/python-linux-cpu.txt
.conda/bin/python -m pip install --no-deps -e .
.conda/bin/python -m pip check
```

Install native backends separately; do not install the learner's Torch/NumPy pins
into Isaac's shipped interpreter:

```bash
bash scripts/setup_upstream.sh
bash scripts/setup_isaac.sh
bash scripts/setup_gazebo_userland.sh
export OMNI_KIT_ACCEPT_EULA=YES
export TIMELYHELP_GPU=2
```

Isaac Sim uses the official 5.0.0 standalone archive (about 8.7 GB compressed) and
its own Python/CUDA libraries. Gazebo Harmonic is extracted from official signed
Ubuntu/OSRF packages into `.local/gazebo`, including DART, Ogre2 and a private Xvfb
display for the headless server; setup does not require sudo. Gazebo's native worker
uses system Python 3.10; both backends communicate with the learning environment
through local Unix sockets. Installed archives have SHA-256 records in `docs/locks/`.
Gazebo versions and architectures are also recorded. For exact recreation from
retained archives or repositories still serving those versions, use
`bash scripts/setup_gazebo_userland.sh --locked`; it verifies every archive checksum
before extraction. Ubuntu 22.04 and compatible host GPU/system libraries are the
declared native platform. Setup fails explicitly if a locked package is unavailable.

## Live simulation over SSH, using all four GPUs

Run on the server from this repository (no training or full pilot):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .conda/bin/python -m timelyhelp.cli live \
  --gpus 0,1,2,3 --port 8765 --duration 18 --session-minutes 60 \
  --output artifacts/live_demo
```

In a separate terminal **on your own computer**, open a tunnel using your usual
SSH login, including any normal jump-host/VPN options. This server reports the
hostname `ce-dlshm-s.eng.umanitoba.ca` and the current account `binsaeeq`:

```bash
ssh -N -L 8765:127.0.0.1:8765 binsaeeq@ce-dlshm-s.eng.umanitoba.ca
```

Keep that terminal open. Visit **http://localhost:8765** in your browser. When
all four cards show `ready` or `done`, click **Run 18s demos**. Each click starts
one finite mission per GPU. Pause/resume, replay and overview/top/follow camera
controls work across the four worlds. Stop the GPU processes with **Stop session**,
or Ctrl-C in the server terminal. The session also closes after 60 minutes; change
`--session-minutes` to adjust the lifetime. To keep it running after disconnecting
SSH, start a detached `tmux` session instead of the foreground command above:

```bash
mkdir -p artifacts/live_demo
tmux new-session -d -s timelyhelp-live -c "$PWD" \
  'env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .conda/bin/python -m timelyhelp.cli live --gpus 0,1,2,3 --port 8765 --duration 18 --session-minutes 60 --output artifacts/live_demo > artifacts/live_demo/server.log 2>&1'
tail -f artifacts/live_demo/server.log
```

Use one viewer at a time for a given output directory and port. The command rejects
an output directory owned by an active viewer. Stop the previous session using the
browser before restarting; its `tmux` session exits when the viewer closes.

This displays actual Isaac RTX scenes and onboard/infrastructure cameras plus
query/reply state. The main image is now a detailed 1024×576 3D diagnostic scene:
an industrial robot carrying cargo, skinned DynaNav humans, carts, warehouse racks,
pallets, safety rails and loading doors, or a sidewalk with road/buildings/trees.
Models follow actual native poses. Human gait is an authored visual animation tied
to actual velocity/time; it is not stock DynaNav navigation playback.

**Drag** the world image to orbit, **scroll** to zoom, select **Follow robot** for
a close moving camera, or use **Facility overview / Top**. Each card has **Focus**
and **Fullscreen** controls; the first GPU starts focused. The human-only world
camera uses nonphysical display models, not collision meshes, and never enters
policy features. The small onboard/infrastructure panels retain the research
marker rendering: green robot, orange worker, blue cart, gray obstacles. Each
diagnostic capture restores marker visibility/render profiles and refreshes their
annotators before a query or next physics tick. Actual native tests measured zero
RGB/pose/history change and depth differences below 1.2e-7 m in paired short runs.
Onboard depth is colored
yellow near → blue far at 7 m. The **always-query baseline** demonstrates delayed
assistance without relying on the tiny smoke-trained model. View raw telemetry in
each card's expandable panel. Rendering may be slower than wall clock; simulation
time and achieved rate are shown. Browser frames refresh up to 2 Hz.

All four GPUs are assigned independent rendered worlds. Ready/done worlds retain
warm contexts and do not advance physics. PhysX dynamics and the current small GRU
learner execute on CPU. These demos do not claim four-way GPU training or saturate
every card continuously. The live viewer was stopped so the pilot could use the
four GPUs. Restart it only after the pilot releases those cards.

Reproduce the four simultaneous native readiness audits from a server terminal:

```bash
mkdir -p artifacts/gpu_readiness
pids=()
for gpu in 0 1 2 3; do
  TIMELYHELP_GPU="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    .conda/bin/python -m timelyhelp.cli native-smoke --backend isaac --steps 3 \
    --output "artifacts/gpu_readiness/gpu_$gpu" \
    > "artifacts/gpu_readiness/gpu_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0
```

All four passed; evidence is under `artifacts/gpu_readiness/`. Viewer control/replay
checks, camera images and mission records are under `artifacts/live_validation/`.
The verified 18-second four-GPU demo and actual browser screenshots are under
`artifacts/live_demo/`; consolidated follow-up evidence is
[artifacts/live_verification_summary.json](artifacts/live_verification_summary.json).
Those short cautious runs stop before the goal; their purpose is visual and
integration inspection, and they are not evidence of research performance.
The later detailed-3D checks and screenshot evidence are under `artifacts/live_3d/`
and `artifacts/presentation_audit_textured/`; their consolidated summary is
[artifacts/presentation_verification_summary.json](artifacts/presentation_verification_summary.json).
To reproduce the native test of
sensor restoration and matching missions with/without the presentation:

```bash
TIMELYHELP_GPU=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .conda/bin/python scripts/check_presentation.py --output artifacts/presentation_audit
```

The presentation is authored locally, with human meshes and textures from the
pinned TIC-VLA checkout. Each GPU's `frames.json` records their SHA-256 hashes.
Imported human base-color maps are bound using portable USD PreviewSurface materials
because the original MDL bindings did not display reliably in this standalone
runtime. Without optional upstream assets, an authored animated PPE figure is used.
This visual enhancement does not replace the scientific marker detector, change
the collision model or establish photorealistic perception performance.
The browser viewer itself has no extra dependencies. An optional real-browser
regression check uses an isolated Playwright installation and executes one short
demo round against a running server (use port 8766 for an independent validation
session with `--duration 1.2 --output artifacts/live_validation`):

```bash
.conda/bin/python -m venv .local/viewer-check
.local/viewer-check/bin/python -m pip install -r docs/locks/python-live-browser-check.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/.local/viewer-browsers" \
  .local/viewer-check/bin/python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.local/viewer-browsers" \
  .local/viewer-check/bin/python scripts/check_live_browser.py \
  --url http://127.0.0.1:8766 --output artifacts/live_validation/browser
```

## Tests and short smoke runs

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .conda/bin/python -m pytest -q
.conda/bin/ruff check src tests
.conda/bin/ruff format --check src tests
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timelyhelp pipeline \
  --config configs/smoke.yaml --output artifacts/smoke
timelyhelp native-smoke --backend isaac --steps 3 --output artifacts/native_smoke/isaac
timelyhelp native-smoke --backend gazebo --steps 3 --output artifacts/native_smoke/gazebo

# Short native primary pipeline and frozen transfer; no research-scale training.
timelyhelp pipeline --config configs/native_smoke.yaml --output artifacts/native_pipeline
timelyhelp transfer --config configs/native_smoke.yaml \
  --models artifacts/native_pipeline/models/manifest.json --output artifacts/native_transfer
```

The reference smoke uses short episodes, one training seed, two ensemble members
and two epochs. It exercises both nominal controllers, every baseline, all seven
ablations, actual threshold calibration episodes, paired request audits, saved model
reloads, window predictions and figure generation. The configured smoke samples
a bounded disturbance subset; unit tests cover all families and disturbance mechanisms.
It also executes six separately privileged oracle missions, excluded from ordinary
policy results. Pilot/confirmatory configs budget the costly oracle on six/twelve
matched scenarios, using actual controlled continuations and 2.5-second replanning.
Native smoke checks actual wheel motion, rendered onboard/infrastructure RGB-D,
sensor tracks, physical contact labels, reset/replay and paired continuations with
recorded tolerances. It fails explicitly when a native audit does not pass.
The native pipeline uses four 0.8-second split rollouts, one ensemble member, one
epoch and three short primary comparator missions. Its transfer uses three physical
Gazebo layouts and the same fitted weights/selection checksums; it does not fit
anything on transfer data.

Artifacts include `pipeline_manifest.json`, stage `provenance.json`, paired feature
caches and raw branch outcomes, development-only privileged source audits, model
checkpoints and fit reports, deployed selection calibration reports, mission records,
oracle headroom diagnostics, pilot audits, resource profiles, and PNG/JSON/Markdown
reports. Every run records its resolved configuration, source/brief hashes, upstream
commit, seeds, dependencies and backend. Tiny calibration sets may have **no feasible
risk threshold**; the report preserves that result.

The separately executed oracle comparison is
`report/privileged_oracle_mission_report.json`; it includes matched mission frontiers
and paired intervals against no-query controls and delay-aware VOI. Its ideal chosen
arrival times are privileged diagnostic interventions, with imperfect/missing replies
retained. The verified frozen transfer audit is
`artifacts/native_transfer/frozen_gate_verification.json`, and the combined native
figures are in `artifacts/native_combined_report/`. Native 0.8-second plumbing missions
retain expected time-limit failures and cannot establish transport performance.

## Run research stages and reproduce the pipeline

Isaac simulation for `configs/pilot.yaml` is one schedule partitioned across
GPUs 0–3. Each GPU has its own process with `TIMELYHELP_GPU` set to that index.
An episode, calibration rollout, or matched mission runs on one GPU. The same
pilot is not started four times. Seeds, split membership, and matched A/W/Q
branches stay with the episode that produced them. Completed episode shards
are reused when the same command is started again. `TIMELYHELP_GPUS` overrides
the device list. `TIMELYHELP_SERIAL=1` keeps Isaac in the parent process.
Reference and Gazebo runs stay in-process. The small GRU still trains on CPU.

The pilot command below is the run that is in progress in tmux session
`timelyhelp-pilot`. Do not start `configs/confirmatory.yaml` from these results
until the pilot has been assessed.

```bash
# Initial 300 snapshots per nominal controller; native Isaac primary backend.
mkdir -p outputs/pilot
tmux new-session -d -s timelyhelp-pilot -c "$PWD" \
  'exec env TIMELYHELP_GPUS=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES .conda/bin/python -m timelyhelp.cli pipeline --config configs/pilot.yaml --output outputs/pilot >> outputs/pilot/pipeline.log 2>&1'

# Monitor the running pilot. progress.json is the live schedule.
watch -n 10 .conda/bin/python scripts/pilot_status.py
tmux ls
tail -f outputs/pilot/pipeline.log
cat outputs/pilot/workers/progress.json
tail -f outputs/pilot/workers/gpu_0.log
nvidia-smi

# After the pilot finishes, or to continue a stopped run, use the same tmux
# command. Shards under outputs/pilot/data/shards, calibration shards, and
# mission shards are kept. A finished stage manifest is not repeated.
# Interrupt with Ctrl-C in the tmux pane. In-flight episodes are redone;
# completed shards remain. If Isaac processes remain, stop the process groups
# recorded in outputs/pilot/workers/progress.json.

# Proposed 3000 snapshots per controller, five independent training seeds,
# 200 matched test missions per seed and method. Not started. Scale only after
# pilot evidence.
# timelyhelp pipeline --config configs/confirmatory.yaml --output outputs/confirmatory

# Limited Gazebo robustness check after confirmatory training. Not launched.
# 100 transfer episodes per seed; never retrain or recalibrate on transfer test data.
# timelyhelp transfer --config configs/confirmatory.yaml \
#   --models outputs/confirmatory/models/manifest.json --output outputs/transfer
```

The status command checks process liveness and shows the active snapshot/branch
alongside physics updates. **159/200 counts completed whole generation episodes**:
each episode has three snapshots, 60 query candidates/replications per snapshot,
and the matched A/W continuations. The counter can stay fixed while those branches
execute. This is data generation; GRU fitting starts after generation finishes.
Later stages have separate calibration/evaluation mission counters. There is no
estimated completion time until representative pilot throughput is measured.

The pipeline holds an exclusive `outputs/pilot/.pipeline.lock`; a second pipeline
launch against that directory fails before changing progress or provenance. A
stale lock-file PID does not prevent restarting an exited process. Run standalone
stages below only after the pipeline exits. Keep scientific source/configuration
fixed during a run. Completed fits require matching `.fit.stamp.json`, weights,
fit report and dataset/source-calibration identity before resuming calibration.

Recovery checks and original-data hashes are in
`artifacts/pilot_recovery_20261015/`, including `resume_preflight.json` and
`saved_shard_audit.json`. The full bounded reference/native pipelines and nine
frozen Gazebo transfer missions passed; these verify integration, not scientific
performance. Historic pilot shards retain their original work hashes and are
listed individually in the generated data manifest. New captures do not revalidate
historical post-arrival images that were not saved.

Reproduce the focused native W/Q check without full missions or training:

```bash
TIMELYHELP_GPU=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .conda/bin/python \
  scripts/check_native_pairs.py --config configs/pilot.yaml \
  --episode generate/reactive/development/0000 --snapshot-times .5,1.5,2.5 \
  --delays .25,3 --replications 0,2 --sources infrastructure --bridges cautious \
  --capture-interleave --output artifacts/new_native_pair_check
```

Use a fresh diagnostic output directory. To audit retained conditioning, run four
disjoint parts with `scripts/check_pilot_prefixes.py --gpu N --part N --parts 4
--output artifacts/new_prefix_check/part_N` for N=0,1,2,3 after the pilot releases
the devices. Neither diagnostic rewrites saved research data.

Stages can also be resumed independently:

```bash
timelyhelp generate --config configs/pilot.yaml --output outputs/pilot/data
timelyhelp train --config configs/pilot.yaml \
  --dataset outputs/pilot/data/manifest.json --output outputs/pilot/models
timelyhelp evaluate --config configs/pilot.yaml \
  --models outputs/pilot/models/manifest.json --output outputs/pilot/evaluation
timelyhelp report --config configs/pilot.yaml \
  --records outputs/pilot/evaluation/records.json \
  --dataset outputs/pilot/data/records.json \
  --models outputs/pilot/models/manifest.json --output outputs/pilot/report
```

Output directories are explicit. Use a fresh directory for an independent run;
stage manifests and incremental progress files preserve completed artifacts. To
change physics, costs, budgets, delays, split size, source priors, model settings or
seeds, copy a YAML configuration and pass its path. Configuration and dataset readers
reject unknown simulation settings, split leakage and inconsistent paired labels.
The 5-second supervisor probe is configurable; it remains a proposed sensitivity
choice, not a calibrated human response model.

For a combined primary/transfer figure, concatenate the two **mission records** and
run the report with the primary counterfactual records:

```bash
.conda/bin/python - <<'PY'
import json
from pathlib import Path
from timelyhelp.config import load_config
from timelyhelp.experiments import report_pipeline
primary = json.loads(Path('outputs/confirmatory/evaluation/records.json').read_text())
transfer = json.loads(Path('outputs/transfer/records.json').read_text())
report_pipeline(load_config('configs/confirmatory.yaml'), primary + transfer,
                'outputs/combined_report',
                counterfactual_rows='outputs/confirmatory/data/records.json',
                models='outputs/confirmatory/models/manifest.json')
PY
```

## Code and scientific boundaries

```text
src/timelyhelp/
  simulation/       split layouts, tasks, reactive reference backend
  backends/         genuine Isaac/Gazebo physics and rendered camera workers
  perception.py     RGB-D detector, tracker and calibrated backprojection
  sources.py        bounded questions, capture/arrival/error/loss models
  control.py        frozen planners, common evidence admission and safety
  counterfactual.py A/W/Q branches, branch reuse and restoration audits
  learning/         audited caches, GRU ensemble, calibration, gate and windows
  baselines.py      matched adaptations and observation-belief VOI
  evaluation.py     finite commitment execution and paired hierarchical inference
  reporting.py      Pareto metrics, six figures and machine-readable reports
  audits.py         no-leakage, label accounting, oracle/pilot and resource audits
  experiments.py    generation, fitting, calibration, evaluation and reporting
```

All autonomous policies receive public observations and calibrated source metadata.
Rendered source tracks pass through the detector/tracker; direct simulator tracks
are restricted to explicitly privileged diagnostics. Known static floorplans and
crossing zones are public mission metadata. Questions obtain information and never
command workers to yield. A/W/Q use common random draws while workers respond to
the actual branch's robot motion. Failed/time-limited missions remain in the cost,
classification and raw metrics. Time heads estimate restricted completion time;
failure targets use the mission budget, while actual elapsed time is reported separately.

The CPU backend is labeled `reference` and cannot substitute for Isaac/Gazebo
evidence. Native tasks use a frozen detector for visible class markers and authored
reactive actors, keeping the assistance estimator auditable. DynaNav assets can be
referenced with `TIMELYHELP_DYNANAV_SCENE=/absolute/path/to/scene.usd`; the stock
DynaNav evaluation remains in `third_party/TIC-VLA/DynaNav/run_benchmark.sh` and is
separate from the assistance-enabled task suite. Pretrained TIC-VLA training includes
DynaNav maps. No original-map unseen claim, adapted TIC-VLA controller result,
real-robot validation or construction-field safety claim is made. The second
implemented nominal controller is `ReactivePlanner`.

The authored actor/controller implementation is a documented change from reusing
stock DynaNav human/navigation playback: every relevant state and exogenous draw
must be reconstructible for controlled A/W/Q labels. Its extension-managed playback
is not assumed restorable. The original benchmark, assets and optional USD scene
reference remain separate; stock scene/animation integration has not been validated
in these assistance runs. The verified task suite uses authored reactive actors and
frozen visible class markers, without claiming photorealistic human perception.

The frozen encoder has 61 public features, including received intent and observed
track geometry/velocity. Both native cameras use their actual intrinsic calibration
and captured pose for depth backprojection. Residual calibration and deployed
threshold calibration use distinct rollout seed namespaces; test layouts remain
independent. Selected-policy collision bounds account for the threshold grid and
may remain infeasible on small samples. Engine-specific metrics and intervals remain
separate in combined primary/transfer reports.

Confidence, failure/lead-time and planning-aware baselines are stated task adaptations.
The failure comparator uses frozen calibrated nominal collision/completion forecasts
and an explicit public conflict lead estimate; it does not select on information value.
The strongest VOI baseline samples an uncertain observation-conditioned belief and
runs the common sensor/planner/evidence/safety stack; it receives no world-state
oracle. Prediction intervals and deployed calibration are empirical and do not
guarantee collision avoidance under shift. Full-scale results must demonstrate a
benefit beyond caution and delay-aware VOI before supporting the research hypothesis.
