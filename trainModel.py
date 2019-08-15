# Anel Salas 08/14/19
########### scikit-learn for the Nano on Linux###########
# so I went the anaconda route: downloaaded the jetson dockerfile and grabed the scrip
# that installed anaconda for the nano: archiconda3.sh and ran it:
# sudo sh archiconda.sh -b -p /opt/archiconda3
# now runnig conda3 install 
# sudo /opt/archiconda3/bin/conda install scikit-learn
# then add conda to the path 
# expot PATH="/opt/archiconda3/bin:$PATH"
# conda create -n [Virtualenv_Name] python=3.7 scikit-learn
# source activate py3k
# list the environments with conda info --envs
# then install opencv through the sh script
# sudo sh install_opencv4.0.0_Nano.sh
# wait ... it takes over an hour on the nano
# after this you will have scikit-learn installed on the virtual environment
# and also opencv 4.0.0
# remember to move cv2.so to cv2.so
# cd /usr/local/python/cv2/python-3.6
#
#mv cv2.cpython-36m-aarch64-linux-gnu.so cv2.so
############################################

########### scikit-learn for Microsucks Windows #########
# on windows what worked was py -m pip install scikit-learn
############################################


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import cv2
import os
import glob
import numpy

#from os import listdir
#from os.path import isfile, join


import FaceRecognitionUtils as fr
import faceDetection
embedder = fr.LoadFaceRecognizerModel ()
net_model = faceDetection.LoadPretrainedFaceRecognitionModel()

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

def Init ():
    return

'''
# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
 
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])


# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
'''
def extratFoldersFromPath (mypath):
    folders = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        for items in dirnames: 
            fullfile = os.path.join(dirpath,items)
            folders.append(fullfile)

    return folders

def extratFilesFromPath (mypath):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        for items in filenames: 
            fullfile = os.path.join(mypath,items)
            files.append(fullfile)
            print(fullfile)
    return files

def extractLastPathName (file):
    return os.path.basename(os.path.normpath(file))

def createBlob (image):
    image = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)
    return image


def main():
    Init()
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")

    mypath = "learningDataset"
    
    folders = extratFoldersFromPath(mypath)
    for items in folders:
        print (items)
        name = extractLastPathName(items)
        files = extratFilesFromPath( items)
        for eachfile in files:
            image = cv2.imread(eachfile)
            (h, w) = image.shape[:2]
            imageBlob = createBlob (image)

            mat_detections = faceDetection.ExtractDetectedFaces(imageBlob,net_model)
            for i in range(0, mat_detections.shape[2]):
                # only grab faces with 60% confidence or more
                if mat_detections[0, 0, i, 2] > 0.7:
                    box = mat_detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face_image = image[startY:endY, startX:endX] 
            
                    desiredWidth=600
                    newHeight = faceDetection.GetNewHeightMaintainRatio (desiredWidth,startX,endX,startY,endY)
        
                    if face_image.any() :
                        face_image = cv2.resize(face_image, (desiredWidth, newHeight))
                        vec = fr.CreateFaceEmbedding (face_image,embedder)
                        knownNames.append(name)
                        knownEmbeddings.append(vec.flatten())

    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("embeddings.dat", "wb")
    f.write(pickle.dumps(data))
    f.close()


    # load the face embeddings
    data = pickle.loads(open("embeddings.dat", "rb").read())
     
    # encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    # write the actual face recognition model to disk
    f = open("recognizer.dat", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
     
    # write the label encoder to disk
    f = open("labelencoder.dat", "wb")
    f.write(pickle.dumps(le))
    f.close()


    



'''
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
	    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	    name = imagePath.split(os.path.sep)[-2]
        for item in name:
            print(item)
        sys.exit()
'''
        # load the image, resize it to have a width of 600 pixels (while
	    # maintaining the aspect ratio), and then grab the image
	    # dimensions
	    #image = cv2.imread(imagePath)
	    #image = imutils.resize(image, width=600)
	    #(h, w) = image.shape[:2]
     
    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    #knownEmbeddings = []
    #knownNames = []
 
# initialize the total number of faces processed
#total = 0

if __name__ == "__main__":
   main()

