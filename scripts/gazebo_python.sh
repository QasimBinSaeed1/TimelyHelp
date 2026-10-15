#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
prefix="$root/.local/gazebo/usr"
export LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu:$prefix/lib/x86_64-linux-gnu/OGRE-2.3:$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$prefix/lib/python3/dist-packages:$root/src${PYTHONPATH:+:$PYTHONPATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$prefix/lib/x86_64-linux-gnu/gz-sim-8/plugins"
export GZ_SIM_RESOURCE_PATH="$prefix/share/gz/gz-sim8/worlds:$root/assets/gazebo"
export GZ_RENDERING_PLUGIN_PATH="$prefix/lib/x86_64-linux-gnu/gz-rendering-8/engine-plugins"
export OGRE2_RESOURCE_PATH="$prefix/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
export GZ_RENDERING_RESOURCE_PATH="$prefix/share/gz/gz-rendering8"
export GZ_SIM_PHYSICS_ENGINE_PATH="$prefix/lib/x86_64-linux-gnu/gz-physics-7/engine-plugins"
export GZ_PARTITION="timelyhelp-${USER}-${TIMELYHELP_RUN_ID:-$PPID}"
export PATH="$prefix/bin:$PATH"
# Gazebo's Python TestFixture lacks ServerConfig's headless-rendering setter.
# A private Xvfb display permits offscreen Ogre2 rendering without a desktop.
if [[ -z "${DISPLAY:-}" && -x "$prefix/bin/Xvfb" ]]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  exec "$prefix/bin/xvfb-run" -a -s '-screen 0 800x600x24 -nolisten tcp' /usr/bin/python3 "$@"
fi
exec /usr/bin/python3 "$@"
