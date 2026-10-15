#!/usr/bin/env bash
# Install Gazebo Harmonic into the workspace without sudo or system mutation.
set -euo pipefail
cd "$(dirname "$0")/.."
prefix="$PWD/.local/gazebo"
aptroot="$PWD/.local/apt"
mkdir -p "$prefix" "$aptroot/lists/partial" "$aptroot/archives/partial" docs/locks
curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o "$aptroot/gazebo.gpg"
cat > "$aptroot/sources.list" <<EOF
deb http://archive.ubuntu.com/ubuntu jammy main universe
deb http://archive.ubuntu.com/ubuntu jammy-updates main universe
deb http://security.ubuntu.com/ubuntu jammy-security main universe
deb [signed-by=$aptroot/gazebo.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main
EOF
options=(-o "Dir::Etc::sourcelist=$aptroot/sources.list" -o 'Dir::Etc::sourceparts=-'
  -o "Dir::State::lists=$aptroot/lists" -o "Dir::Cache::archives=$aptroot/archives"
  -o "APT::Sandbox::User=$(id -un)" -o 'Debug::NoLocking=1')
apt-get "${options[@]}" update
if [[ "${1:-}" == "--locked" ]]; then
  mapfile -t pins < docs/locks/gazebo-package-pins.txt
  apt-get "${options[@]}" --download-only --reinstall --no-install-recommends -y install "${pins[@]}"
  sha256sum -c docs/locks/gazebo-debs.sha256
elif [[ $# -eq 0 ]]; then
  apt-get "${options[@]}" --download-only --no-install-recommends -y install \
    gz-sim8-cli libgz-sim8-plugins python3-gz-sim8 python3-gz-transport13 python3-gz-msgs10 \
    libgz-physics7-dartsim libgz-rendering8-ogre2 libgz-rendering8-ogre2-dev \
    xvfb xauth python3-numpy python3-protobuf
else
  echo 'Usage: setup_gazebo_userland.sh [--locked]' >&2
  exit 2
fi
for package in "$aptroot"/archives/*.deb; do
  dpkg-deb -x "$package" "$prefix"
done
# Runtime-only packages ship versioned engines; the loader's conventional names
# are normally provided by development packages. Create local aliases explicitly.
ln -sfn libgz-physics7-dartsim-plugin.so.7 \
  "$prefix/usr/lib/x86_64-linux-gnu/gz-physics-7/engine-plugins/libgz-physics-dartsim-plugin.so"
ln -sfn libgz-rendering8-ogre2.so.8 \
  "$prefix/usr/lib/x86_64-linux-gnu/gz-rendering-8/engine-plugins/libgz-rendering-ogre2.so"
if [[ $# -eq 0 ]]; then
  find .local/apt/archives -name '*.deb' -print0 | sort -z | xargs -0 sha256sum > docs/locks/gazebo-debs.sha256
  printf 'package\tversion\tarchitecture\n' > docs/locks/gazebo-packages.tsv
  : > docs/locks/gazebo-package-pins.txt
  for package in .local/apt/archives/*.deb; do
    dpkg-deb --show --showformat='${Package}\t${Version}\t${Architecture}\n' "$package" >> docs/locks/gazebo-packages.tsv
    dpkg-deb --show --showformat='${Package}=${Version}\n' "$package" >> docs/locks/gazebo-package-pins.txt
  done
fi
