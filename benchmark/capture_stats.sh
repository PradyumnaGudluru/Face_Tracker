#!/bin/sh
# Usage: ./monitor-usage.sh <PID of the process> <Output folder>
# Output: top.dat with lines such as `305m 2.0`, i.e. memory with m/g suffix - CPU load in %

PID="$1"
OUTPUT_DIR="$2"
OUTPUT="$OUTPUT_DIR"/top.csv
echo 'memory(Mb),cpu(%)' > "$OUTPUT"
while true; 
do 
    # check if the process is running
    t_pid=$( ps aux | awk '{print $2 }' | grep "$PID" )
    if [ -z "$t_pid" ];
    then
        echo "Process ""$PID"" has exited, stop capturing stats"
        break
    fi 
    top -p $PID -bn 1 | egrep "$PID" | awk '{print $6/1000,","$9}' >> "$OUTPUT"
done
