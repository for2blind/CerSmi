#!/bin/bash
until python3 DaShengSchedulerEP3.py $1; do
    echo "DaShengSchedulerEP3.py crashed with exit code $?. Respawning.." >&2
    sleep 1
done
