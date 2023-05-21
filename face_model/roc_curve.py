'''
This Python scripts builds the ROC curve for the fisherface model. It trains
the model using the prepared training data and tests the model. The test
result is used to draw the ROC curve as well as calculate the AUC value.
'''

import csv
import cv2
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt



def draw_ROC(y_test, y_pred):
    #define metrics
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    # plt.show()
    plt.savefig("ROC.png")



def prepare_data(filename):

    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    #list to hold all image paths
    image_paths = []

    df = pd.read_csv(filename)
    image_paths = df["file_path"].tolist()
    image_labels = df["label"].tolist()

    for i in range(len(image_paths)):
        #read image
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        #add face to list of faces
        faces.append(image)
        #add label for this face
        labels.append(image_labels[i])
    
    return faces, labels



# Prepare the training data
print("Preparing training data...")
faces, labels = prepare_data("yalefaces.csv")

#create our Fisherface recognizer 
face_recognizer = cv2.face.FisherFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

#save the model
face_recognizer.save("model.yml")

# Prepare the testing data
print("Preparing testing data...")
test_faces, test_labels = prepare_data("testfaces.csv")

# Create binarized test labels
bin_test_labels = list()
for label in test_labels:
    if label > 21:
        bin_test_labels.append(0)
    else:
        bin_test_labels.append(1)

# Test each face in the test face set
pred_labels = list()
bin_predict_labels = list()

# print("Test labels = {}".format(test_labels))

# Fill out the binarized prediction labels
for i in range(len(test_faces)):
    #predict the image using our face recognizer 
    pred_label, confidence = face_recognizer.predict(test_faces[i])
    pred_labels.append(pred_label)
    if pred_label == test_labels[i]:
        bin_predict_labels.append(1)
    else:
        bin_predict_labels.append(0)

# print("Pred labels = {}".format(pred_labels))

# print("bin_test_labels = {}".format(bin_test_labels))
# print("bin_pred_labels = {}".format(bin_predict_labels))

# Draw ROC with AUC calculation
draw_ROC(bin_test_labels, bin_predict_labels)




