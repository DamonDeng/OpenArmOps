#!/bin/bash
# Launch the prebuilt PXREARobotSDK ConsoleDemo as the SDK-consumer side
# of the XRoboToolkit broker. The broker (RoboticsServiceProcess) must
# already be running on :60061 for the SDK port and :63901 for the
# device port. This script attaches a consumer so the broker has
# somewhere to deliver pose frames pushed by the Pico.
#
# Logs land in /tmp/consoledemo_<timestamp>/ along with a tail of the
# broker's own log file so post-mortem only needs one directory.

set -u

SVC_DIR=/opt/apps/roboticsservice
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=/tmp/consoledemo_${TS}
mkdir -p "${OUT_DIR}"

if [ ! -x "${SVC_DIR}/ConsoleDemo" ]; then
    echo "ERROR: ${SVC_DIR}/ConsoleDemo not found or not executable" >&2
    exit 1
fi

if ! pgrep -x RoboticsService >/dev/null 2>&1 \
   && ! pgrep -f RoboticsServiceProcess >/dev/null 2>&1; then
    echo "WARNING: RoboticsServiceProcess does not appear to be running."
    echo "         Start it first: cd ${SVC_DIR} && ./runService.sh"
    echo "         Continuing anyway — ConsoleDemo will retry the connect."
fi

BROKER_LOG_DIR="${HOME}/.local/share/PICOBusinessSuitData/log"
BROKER_LOG=$(ls -t "${BROKER_LOG_DIR}"/*.txt 2>/dev/null | head -1 || true)

export LD_LIBRARY_PATH="${SVC_DIR}:${SVC_DIR}/lib:${SVC_DIR}/SDK/arm64:${LD_LIBRARY_PATH:-}"

cd "${SVC_DIR}"

echo "===== run_consoledemo.sh ====="                  | tee "${OUT_DIR}/consoledemo.log"
echo "started: $(date -Is)"                             | tee -a "${OUT_DIR}/consoledemo.log"
echo "out_dir: ${OUT_DIR}"                              | tee -a "${OUT_DIR}/consoledemo.log"
echo "broker_log: ${BROKER_LOG:-<none found>}"          | tee -a "${OUT_DIR}/consoledemo.log"
echo "ld_library_path: ${LD_LIBRARY_PATH}"              | tee -a "${OUT_DIR}/consoledemo.log"
echo "============================="                    | tee -a "${OUT_DIR}/consoledemo.log"

# Tail the broker log into our out_dir so all post-mortem evidence is
# collected in one place. -F so it follows rotation if the broker rolls
# the file. Run as background; killed on exit via trap below.
TAIL_PID=""
if [ -n "${BROKER_LOG}" ] && [ -r "${BROKER_LOG}" ]; then
    tail -n 0 -F "${BROKER_LOG}" > "${OUT_DIR}/broker.log" 2>&1 &
    TAIL_PID=$!
fi

cleanup() {
    if [ -n "${TAIL_PID}" ] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill "${TAIL_PID}" 2>/dev/null || true
    fi
    echo                                                   | tee -a "${OUT_DIR}/consoledemo.log"
    echo "===== stopped: $(date -Is) ====="                | tee -a "${OUT_DIR}/consoledemo.log"
    echo "log dir: ${OUT_DIR}"
}
trap cleanup EXIT INT TERM

# stdbuf -oL forces line-buffered stdout so we see callbacks live in
# the tee'd file (otherwise stdio block-buffers when redirected and
# we'd only see output on shutdown).
echo "Launching ConsoleDemo. Ctrl-C to stop."           | tee -a "${OUT_DIR}/consoledemo.log"
echo                                                    | tee -a "${OUT_DIR}/consoledemo.log"
stdbuf -oL -eL "${SVC_DIR}/ConsoleDemo" 2>&1 | tee -a "${OUT_DIR}/consoledemo.log"
