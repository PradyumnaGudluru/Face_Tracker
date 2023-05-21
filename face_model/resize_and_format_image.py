import sys
import os.path
import argparse
import cv2
import os 



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=str,
                        help="The base path of the directory to be parsed")
    parser.add_argument("output_path", type=str,
                        help="The output path of resized images")
    parser.add_argument("row", type=int,
                        help="The target row resolution")
    parser.add_argument("col", type=int,
                        help="The target column resolution")


    args = parser.parse_args()
    BASE_PATH = args.base_path
    OUTPUT_PATH = args.output_path
    ROW_SIZE = args.row 
    COL_SIZE = args.col


    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img_path = os.path.join(subject_path, filename)
                output_dir = os.path.join(OUTPUT_PATH, subdirname)

                if os.path.isdir(output_dir) == False:
                    try: 
                        os.mkdir(output_dir) 
                    except OSError as error: 
                        print(error) 
                
                output_path = os.path.join(output_dir,filename)
                print("Resizing image %s ..." % (subject_path))
                image = cv2.imread(img_path)
                down_points = (ROW_SIZE, COL_SIZE)
                new_image = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
                # Convert to grey
                gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                # Write to disk
                cv2.imwrite(output_path, gray)

        # resize images in the root working directory
        for filename in filenames:
            # create input file path
            file_path = os.path.join(BASE_PATH, filename)
            print("Resizing image %s ..." % (file_path))
            image = cv2.imread(file_path)
            down_points = (ROW_SIZE, COL_SIZE)
            new_image = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
            # Convert to grey
            gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            # create output file path
            output_path = os.path.join(OUTPUT_PATH, filename)
            # Write to disk
            cv2.imwrite(output_path, gray)
  
       


    print("Resizing completed")