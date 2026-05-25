#!/bin/bash

echo "Starting Track-Fit Demo"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mediapipe

cd ./app
python main.py &
PID=$!

echo ""
echo "Service started"
echo "Open browser: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $PID" INT
wait
