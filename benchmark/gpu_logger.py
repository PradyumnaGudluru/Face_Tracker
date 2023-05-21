#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This program logs the GPU usage data as well as the GPU temperaure data consistently to the log.csv file.
# The implementation is based on the example given from the jetson-stat repository:
# https://github.com/rbonghi/jetson_stats/blob/master/examples/jtop_logger.py

from jtop import jtop, JtopException
import csv
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    # Standard file to store the logs
    parser.add_argument('--file', action="store", dest="file", default="gpu_log.csv")
    args = parser.parse_args()

    print("Simple jtop logger")
    print("Saving log on {file}".format(file=args.file))

    try:
        with jtop() as jetson:
            with open(args.file, 'w') as csvfile:
                stats = jetson.stats
                print(type(stats))
                # Initialize cws writer
                writer = csv.DictWriter(csvfile, fieldnames=["GPU", "Temp_GPU"])
                # Write header
                writer.writeheader()
                # This is to ensure that the data is actually written to the file,
                # especially when running the script as a background process
                csvfile.flush()
                # Write first row
                gpu_stats = {"GPU": stats["GPU"], "Temp GPU" : stats["Temp GPU"]}
                writer.writerow(gpu_stats)
                csvfile.flush()
                # Start loop
                while jetson.ok():
                    stats = jetson.stats
                    gpu_stats = {"GPU": stats["GPU"], "Temp GPU" : stats["Temp GPU"]}
                    # Write row
                    writer.writerow(gpu_stats)
                    csvfile.flush()
                    print("Log at {time}".format(time=stats['time']))
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")
# EOF