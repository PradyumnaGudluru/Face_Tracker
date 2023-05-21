#!/bin/bash
# This script extracts FPS data from the syslog file for both the left camera and the 
# right camera. The extracted files are saved as .csv files for post-processing by
# data-analysis.ipynb

BASE_DIR="$1"
DIRECTORY="$BASE_DIR"/fps_logs

if [ ! -d "$DIRECTORY" ];
then
    mkdir "$DIRECTORY"
fi 

cd "$DIRECTORY"

# loop through the directory
for d in */ ; do

    SOURCE="$d"/fps_log

    # skip this directory if fps_log is not found
    if [ ! -f "$SOURCE" ];
    then
        continue
    fi 

    LCAM_CSV="$d"/lcam_fps.csv
    RCAM_CSV="$d"/rcam_fps.csv

    #Extract FPS values for the left camera
    echo "fps" > "$LCAM_CSV"
    cat ${SOURCE} | grep "Left-Tracking" | awk '{print $7}' | cut -d "=" -f2 >> "$LCAM_CSV"

    #Extract FPS values for the right camera
    echo "fps" > "$RCAM_CSV"
    cat ${SOURCE} | grep "Right-Tracking" | awk '{print $7}' | cut -d "=" -f2 >> "$RCAM_CSV"

done

