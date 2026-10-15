#!/usr/bin/env bash
# Official standalone distribution; Isaac's own Python/CUDA stays outside Conda.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local/downloads .local/isaacsim docs/locks artifacts
archive=.local/downloads/isaac-sim-5.0.0.zip
if [[ ! -x .local/isaacsim/python.sh ]]; then
  curl -fL --retry 2 -C - -o "$archive" \
    https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip
  if [[ -f docs/locks/isaac-download.sha256 ]]; then
    sha256sum -c docs/locks/isaac-download.sha256
  fi
  unzip -q "$archive" -d .local/isaacsim
fi
export OMNI_KIT_ACCEPT_EULA=YES
# The standalone worker imports only its adjacent bridge modules. Installing the
# learner package here would declare incompatible Torch/NumPy requirements even
# with --no-deps, so preserve the shipped native environment as provided.
.local/isaacsim/python.sh -c 'import numpy, scipy, yaml; print(numpy.__version__,scipy.__version__,yaml.__version__)'
sha256sum "$archive" > docs/locks/isaac-download.sha256
