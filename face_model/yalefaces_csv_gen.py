#!/usr/bin/env python

import sys
import os.path
import re 

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
# shuran@fun:~/ECEN5763-Exercise4/Q5$ tree -L 2 yalefaces/
# .
# |-- README
# |-- Subject13
# ├── subject13.centerlight
# │   ├── subject13.glasses
# │   ├── subject13.happy
# │   ├── subject13.leftlight
# │   ├── subject13.noglasses
# │   ├── subject13.normal
# │   ├── subject13.rightlight
# │   ├── subject13.sad
# │   ├── subject13.sleepy
# │   ├── subject13.surprised
# │   └── subject13.wink

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: csv_gen <base_path>")
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=","
    # print header
    print("file_path,label")

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            # using re.findall() to find the label 
            # getting numbers from string 
            temp = re.findall(r'\d+', subject_path)
            res = list(map(int, temp))
            label = res[-1]
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print("%s%s%d" % (abs_path, SEPARATOR, label))