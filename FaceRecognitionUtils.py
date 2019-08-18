# Anel Salas 08/19
###################
# The model responsible for actually quantifying each face in an 
# image is from the OpenFace project (https://cmusatyalab.github.io/openface/)
# a Python and Torch # implementation of face recognition with deep learning. 
# This implementation comes from Schroff et al.’s 2015 CVPR publication, 
# FaceNet: A Unified Embedding for Face Recognition and Clustering
# https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf.
###################
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
import numpy#import argparse
#import imutils
import pickle
import cv2
import os

class FaceRecognizer(object):
    embedder = None
    recognizer = None
    labelEnconder = None

    def __init__(self):
        # The models can be downloaded from the OpenFace project here:
        # http://cmusatyalab.github.io/openface/models-and-accuracies/
        #return cv2.dnn.readNetFromTorch("models/faceRecognition-nn4.v2.t7")
        #return cv2.dnn.readNetFromTorch("models/OpenFace-faceRecognition-nn4.small2.v1.t7")
        self.embedder = cv2.dnn.readNetFromTorch("models/OpenFace-nn4.small2.v1.t7")
        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open("recognizer.dat", "rb").read())
        self.labelEnconder = pickle.loads(open("labelencoder.dat", "rb").read())

    def CreateBlobFromFaceImage (self,face):
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
        faceBlob = cv2.dnn.blobFromImage(face,  1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        return faceBlob

    def CreateFaceEmbedding (self,face):
        faceBlob = self.CreateBlobFromFaceImage (face)
        # Creates the 128-D vector which describes the face.
        self.embedder.setInput(faceBlob)
        vector = self.embedder.forward()
        return vector

    def Predict (self,vec):
        # perform classification to recognize the face
        preds = self.recognizer.predict_proba(vec)[0]
        # get the prediction with highest probability
        j = numpy.argmax(preds)
        proba = preds[j] # get the probability % of certeinty
        # get the name from the names array
        name = self.labelEnconder.classes_[j]
        return proba,name

    def RecognizeFromFace (self, face):
        if face.size is not 0: 
            myvec = self.CreateFaceEmbedding(face)
            proba,name = self.Predict(myvec)
            return proba,name

    def RecognizeAllFacesFromFrame (self, mat_detections,frame):
        for i in range(0, mat_detections.shape[2]):
            # only grab faces >60% detected confidence
            if mat_detections[0, 0, i, 2] > 0.6:
                (h,w) = frame.shape[:2];
                box = mat_detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                confidence = mat_detections[0, 0, i, 2]
                face_image = frame[startY:endY, startX:endX]
                if face_image.size is not 0: #frame on error returns an emtpy image
                    proba,name = self.RecognizeFromFace (face_image)
                    frame = self.DrawBoundingBox(name,proba,box,frame)
        return frame

    def DrawBoundingBox (self,name,proba,box,frame):
        (startX, startY, endX, endY) = box.astype("int")
        # draw the bounding box of the face along with the
        # associated probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return frame
        

