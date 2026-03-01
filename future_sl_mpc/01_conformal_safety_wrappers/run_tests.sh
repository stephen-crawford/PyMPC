#!/usr/bin/env bash
# Run conformal safety tests (build from cpp_mpc first: cd cpp_mpc/build && cmake .. && make test_conformal_safety)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD="${BUILD:-$ROOT/cpp_mpc/build}"
if [[ -x "$BUILD/test_conformal_safety" ]]; then
    "$BUILD/test_conformal_safety"
else
    echo "Build test_conformal_safety first: cd cpp_mpc/build && cmake .. && make test_conformal_safety"
    exit 1
fi
