#!/bin/bash

echo "Starting Track-Fit Demo Script"

echo "Starting MediatePipe Pose Estimation"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mediapipe
cd ../extractor
python api.py &
PID1=$!

sleep 3


echo "Starting ProtoGCN"
conda activate protogcn
cd ../inferencer
python api.py &
PID2=$!

sleep 5


echo "Starting Web Application"
conda activate mediapipe
cd ../app
python main.py &
PID3=$!

echo ""
echo "All services started"
echo "Open browser : http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $PID1 $PID2 $PID3" INT
wait