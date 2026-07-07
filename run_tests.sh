#!/usr/bin/env bash
set -e

# ----------------------------------------------------------
# Always run from the repo root
# ----------------------------------------------------------

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT"

# ----------------------------------------------------------
# Ensure imports like `from lib...` work
# ----------------------------------------------------------

export PYTHONPATH="$ROOT:$PYTHONPATH"

# ----------------------------------------------------------
# Test discovery is *explicitly restricted* to tests/
# ----------------------------------------------------------
TESTDIR="tests"
VISDIR="tmp/visual_tests"

# ----------------------------------------------------------
# Argument handling
#   ./run_tests.sh             -> run normal tests
#   ./run_tests.sh --visual    -> only visual tests
#   ./run_tests.sh --all       -> all tests including visuals
#   ./run_tests.sh --long      -> include long-running tests
# ----------------------------------------------------------

MODE="standard"
PYTEST_LONG_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --visual)
            MODE="visual"
            ;;
        --all)
            MODE="all"
            ;;
        --long)
            PYTEST_LONG_FLAG="--run-long"
            ;;
        *)
            echo "Usage: $0 [--visual|--all] [--long]"
            exit 1
            ;;
    esac
    shift
done

if [[ "$MODE" == "visual" ]]; then
    echo "[run_tests] Clearing old visual outputs..."
    rm -rf "$VISDIR"
    mkdir -p "$VISDIR"

    echo "[run_tests] Running visual tests only (tests/)..."
    pytest "$TESTDIR" -m visual $PYTEST_LONG_FLAG
    exit $?
fi

if [[ "$MODE" == "all" ]]; then
    echo "[run_tests] Clearing old visual outputs..."
    rm -rf "$VISDIR"
    mkdir -p "$VISDIR"

    echo "[run_tests] Running ALL tests in tests/ (including visual)..."
    pytest "$TESTDIR" $PYTEST_LONG_FLAG
    exit $?
fi

# default = run all non-visual tests from tests/
echo "[run_tests] Running standard test suite from tests/..."
pytest "$TESTDIR" -m "not visual" --disable-warnings --maxfail=1 $PYTEST_LONG_FLAG
