#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
upstream_commit=9fa6f8b66b9e121d5df5df071297bba8e5353ebb
mkdir -p third_party
if [[ ! -d third_party/TIC-VLA/.git ]]; then
  git clone https://github.com/ucla-mobility/TIC-VLA.git third_party/TIC-VLA
fi
git -C third_party/TIC-VLA fetch origin "$upstream_commit"
git -C third_party/TIC-VLA checkout --detach "$upstream_commit"
git -C third_party/TIC-VLA rev-parse HEAD
