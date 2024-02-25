#!/bin/bash
until python3 DaShengScaler.py; do
    echo "DaShengScaler.py crashed with exit code $?. Respawning.." >&2
    sleep 1
done
