#!/usr/bin/env bash
# Usage: ./run.sh [run|probe|wifi|power-on|power-off|land|detector-check] [flags]
cd "$(dirname "$0")"
exec python3 -m tello_jev "$@"
