# Jetson stat tutorial page can be found as follows:
# https://rnext.it/jetson_stats/jtop.html

from jtop import jtop
import pprint
import time
import os 

def read_stats(jetson):
    # clear the Screen
    os.system('clear')
    print("\n" * 3)
    print("JTOP Statistic Summary:")
    stats = jetson.stats
    print_stats = pprint.pformat(stats, indent=4)
    print(print_stats)
    # delay 1 second prior to collect data again
    time.sleep(1)
    


jetson = jtop()
jetson.attach(read_stats)
jetson.loop_for_ever()

