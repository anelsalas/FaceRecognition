# Anel Salas 08/19
# Requires pip install scikit-learn 
# scikit-learn requires scipy
# to install first make sure you run python 3 pip
# $ curl -O https://bootstrap.pypa.io/get-pip.py
# $ sudo python3 get-pip.py
# didn't work pip3 install scikit-learn
# didn't work error was that python3 -m pip install --no-binary spacy
# PEP 517 for wheels was not installed so could not build scipy or scikit-learn



########### scikit-learn for the Nano on Linux###########
# so I went the anaconda route: downloaaded the jetson dockerfile and grabed the scrip
# that installed anaconda for the nano: archiconda3.sh and ran it:
# sudo sh archiconda.sh -b -p /opt/archiconda3
# now runnig conda3 install 
# sudo /opt/archiconda3/bin/conda install scikit-learn
############################################

########### scikit-learn for Microsucks Windows #########
# on windows what worked was py -m pip install scikit-learn
# also py -m pip install opencv-contrib-python
############################################



# import the necessary packages
#from imutils import paths
import numpy as np
#import argparse
#import imutils
import pickle
import cv2
import os


def LoadFaceRecognizerModel ():
    # The models can be downloaded from the OpenFace project here:
    # http://cmusatyalab.github.io/openface/models-and-accuracies/
    #return cv2.dnn.readNetFromTorch("models/faceRecognition-nn4.v2.t7")
    #return cv2.dnn.readNetFromTorch("models/faceRecognition-nn4.small2.v1.t7")
    print("Loading face recognizer deep neural network model ...")
    embedder = cv2.dnn.readNetFromTorch("models/OpenFace-nn4.small2.v1.t7")
    return embedder

def CreateBlobFromFaceImage (face):
    """
    Parameters
    image	input image (with 1-, 3- or 4-channels).
    size	spatial size for output image
    mean	scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
    scalefactor	multiplier for image values.
    swapRB	flag which indicates that swap first and last channels in 3-channel image is necessary.
    crop	flag which indicates whether image will be cropped after resize or not
    ddepth	Depth of output blob. Choose CV_32F or CV_8U.

    returns a 4-dimensional Mat with NCHW dimensions order.
    N: batch size
    C: channel
    H: height
    W: width
    so that "NCHW" means a data whose layout is (batch_size, channel, height, width)

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
        (96, 96), (0, 0, 0), swapRB=True, crop=False)
    """
    # construct a blob for the face ROI only
    faceBlob = cv2.dnn.blobFromImage(face,  1.0 / 255,
        (96, 96), (0, 0, 0), swapRB=True, crop=False)
    return faceBlob

def CreateFaceEmbedding (face,embedder):
    faceBlob = CreateBlobFromFaceImage (face)
    # Create the 128-D vector which describes the face.
    # add the name of the person + corresponding face
    # embedding to their respective lists
    '''
    knownNames.append(name)
    knownEmbeddings.append(vec.flatten())
    total += 1
    '''
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return vec

#def LearnWithScikit ():
    # reference here: https://scikit-learn.org/stable/about.html#citing-scikit-learn



 

