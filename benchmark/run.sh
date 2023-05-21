#!/bin/bash
# This script runs both the executable as well as capture_stats.sh to monitor & collect
# CPU,mem,GPU data used by the program over the its lifetime. All collected data will
# be saved as .csv files for post-processing by data-analysis.ipynb

BASE_DIR="$1"
OPTION="$2"
WORK_DIR="$BASE_DIR"/"$OPTION"
BUILD_PATH="$WORK_DIR"/build/
EXPR_ITER_FN="$WORK_DIR"/expr_iter
EXPR_ITER=0

cd "$BASE_DIR"

# Launch gpu_logger.py for GPU statistical sampling
python3 gpu_logger.py --file="$WORK_DIR"/gpu_log.csv &
GPU_LOGGER_PID=$!

# Run the target face_tracker
cd "$BUILD_PATH"
./face_tracker &

# Obtain the face_tracker process ID
FT_PID=$!

# Update the experiment number
if [ ! -f "$EXPR_ITER_FN" ];then
    EXPR_ITER=1
    echo "1" > "$EXPR_ITER_FN"
    mkdir -p "$WORK_DIR"/"top-1"
else
    iter=$(cat "$EXPR_ITER_FN")
    EXPR_ITER=$(( $iter + 1 ))
    echo ${EXPR_ITER} > "$EXPR_ITER_FN"
    mkdir -p "$WORK_DIR"/"top-""$EXPR_ITER"
fi 

cd "$BASE_DIR"

# Capture GPU & Memory data to .dat 
./capture_stats.sh "$FT_PID" "$WORK_DIR"/"top-""$EXPR_ITER"

# Kill gpu_logger.py
kill -9 "$GPU_LOGGER_PID"
