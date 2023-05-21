import sys
import re 
import os 


def searchFirstMatch(regex, text):
	return regex.search(text)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: csv_gen <base_path>")
        sys.exit(1)

    directory=sys.argv[1]
    SEPARATOR=","
    regex = re.compile(r'\d+')
    # print header
    print("file_path,label")

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        label = int(searchFirstMatch(regex, path).group())
        print("%s%s%d" % (path, SEPARATOR, label))
        
