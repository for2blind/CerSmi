# !/bin/bash
until python3 DaShengScalerEP3.py; do
    echo "DaShengScalerEP3.py crashed with exit code $?. Respawning.." >&2
    sleep 1
done
# until python3 TetrisScaler.py; do
#     echo "TetrisScaler.py crashed with exit code $?. Respawning.." >&2
#     sleep 1
# done
