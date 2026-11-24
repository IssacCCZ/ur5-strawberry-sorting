#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ISAAC_SIM_PATH:-}" ]]; then
    isaac_python="${ISAAC_SIM_PATH}/python.sh"
    if [[ ! -x "$isaac_python" ]]; then
        printf 'Cannot execute Isaac Sim Python: %s\n' "$isaac_python" >&2
        exit 2
    fi
elif [[ -n "${ISAAC_SIM_PYTHON:-}" ]]; then
    isaac_python="$ISAAC_SIM_PYTHON"
elif [[ -x "$HOME/isaacsim-6.1.0/python.sh" ]]; then
    isaac_python="$HOME/isaacsim-6.1.0/python.sh"
elif [[ -x "$HOME/isaacsim-5.1.0/python.sh" ]]; then
    isaac_python="$HOME/isaacsim-5.1.0/python.sh"
else
    printf '%s\n' 'Set ISAAC_SIM_PATH to the Isaac Sim directory, or ISAAC_SIM_PYTHON to a Python environment with isaacsim installed.' >&2
    exit 2
fi
# A sourced ROS Jazzy shell injects Python 3.12 modules into Isaac Python 3.11.
# This standalone application has no ROS dependencies.
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
exec "$isaac_python" "$project_dir/run.py" "$@"
