# Anel Salas 07/19
# This script captures from a video source and sends each frame
# to the OpenCV Deep Neural Network Module instead of Dlib.
# This module is fast with CPUs and does not requires CUDA nor 
# GPUs. It is better than the face_recognition module from Dlib Geitgey
# only because it recognizes faces smaller than 80x80px.




#import libraries
import os   
#print(os.__file__)
import cv2
import numpy
import sys
import platform
#import dlib
#import dlibFaceRecognition
import pprint
import math
import pickle

import FaceRecognitionUtils as fr
import ObjectTracker


#geyguy face_recognition library globals
known_face_encodings = []
knon_face_metadata = []
usingplatform = ""

def Init ():
    print("Python version: ", sys.version)
    print("OpenCV Version: {}".format(cv2.__version__))
    usingplatform = GetPlatform ()
    OpenCVUsingCuda()


def GetPlatform ():
    usingplatform = platform.machine();
    if usingplatform  == "aarch64":
        print("Platform", usingplatform, "Using Jetson Nano\n")
    elif usingplatform  == "AMD64":
        print("Platform: " , usingplatform, "Running on a Windows System")
    return usingplatform

def GetVideoObject():
    usingplatform = platform.machine();
    video_capture = None
    if usingplatform  == "aarch64":
        # to find the device use gst-device-monitor-1.0 and it will list the camera specs, 
        # from there I grabbed the with and height 
        # read here: https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html?gi-language=c
        # also, this dude in github has a gstream python script to open bunch of cameras, but mine is simpler
        # https://gist.github.com/jkjung-avt/86b60a7723b97da19f7bfa3cb7d2690e#
        gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               #'videoconvert ! appsink').format(0, 800, 600) # no speed improvement when reduce video size
               'videoconvert ! appsink').format(0, 1280, 720)
        video_capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        #video_capture = cv2.VideoCapture(0,cv2.CAP_GSTREAMER)
        #video_capture = cv2.VideoCapture("http://10.1.10.165:8081", cv2.CAP_GSTREAMER)
        if video_capture.isOpened() !=  True:  
            print("Cannot find camera, quitting")
            sys.exit()
    elif usingplatform  == "AMD64": 
        #print("Running on a Windows System")
        #video_capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
        #video_capture = cv2.VideoCapture(1+cv2.CAP_DSHOW)
        video_capture = cv2.VideoCapture(1+cv2.CAP_DSHOW)
        if video_capture.isOpened() !=  True:  
            print("Cannot find camera, quitting")
            sys.exit()

    elif video_capture is None:
        print("Cannot find camera, quitting")
        sys.exit()
    
    return video_capture

def DlibUsingCuda():
    # this checks if CUDA was build with Dlib, but you can also do sudo tegrastats on the nano
    print(cuda.get_num_devices())
    if dlib.DLIB_USE_CUDA:
        print("Successfully using CUDA")
    else:
        print("Dlib Not using CUDA")

def OpenCVUsingCuda():
    buildinfo = cv2.getBuildInformation().split("\n")
    foundSomething=0
    for item in buildinfo:
        if "Use Cuda" in item:
            print (item.strip())
            foundSomething+=1
        if "NVIDIA" in item:
            print (item.strip())
            foundSomething+=1

    if foundSomething == 0:
        print("Could Not Find CUDA")

def CaptureVideo():
    net_model = LoadPretrainedFaceRecognitionModel()
    video_capture = GetVideoObject()
    process_this_frame = 0
    trackingBoundingBox = None
    trackerReady = False
    box = None
    trackerr = None
    success= None
    name = ""
    recognizer = fr.FaceRecognizer()

    #trackerr = make_myTracker(0,0,0,0)
    listOfTrackedFaces = []

    while True: 
        # process each frame
        ret, frame = video_capture.read()
        if ret == False:
            continue

        blob = GetBlobFromFrame(frame)
        if process_this_frame >= 16:
            process_this_frame = 0

        if process_this_frame == 0 :
            trackingBoundingBox = None
            box = None
            del listOfTrackedFaces[:]

            mat_detections = ExtractDetectedFaces(blob,net_model)
            for i in range(0, mat_detections.shape[2]):
                # only grab faces >60% detected confidence
                if mat_detections[0, 0, i, 2] > 0.6:
                    (h,w) = frame.shape[:2];
                    box = mat_detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    confidence = mat_detections[0, 0, i, 2]
                    face_image = frame[startY:endY, startX:endX]
                    if face_image.size is 0: #frame on error returns an emtpy image
                        continue
                    # draw square around detected
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    trackerr = None
                    trackerr = ObjectTracker.make_myTracker(startX,startY,endX,endY)
                    trackerr.start(frame)
                    proba,name = recognizer.RecognizeFromFace (face_image)
                    mytuple = (proba,name,trackerr)
                    listOfTrackedFaces.append(mytuple)

        process_this_frame += 1
        for item in range(len(listOfTrackedFaces)):
            proba,name,trackerr = listOfTrackedFaces[item]
            (success, trackingBoundingBox) = trackerr.update(frame)
            # check to see if the tracking was a success
            if success:
               process_this_frame += 1
               text = "{}: {:.2f}% item:{}".format(name, proba * 100,item)
               (x, y, w, h) = [int(v) for v in trackingBoundingBox]
               cv2.rectangle(frame, (x-1, y), (x + w, y + h),(0, 255, 0), 2)
               cv2.putText(frame,text,(x,y-1),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)

        # Display frame 
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit
        keypressed = 0xFF & cv2.waitKey(1)
        if keypressed == ord('q') or keypressed == ord('Q') :
           break

        continue;

def GetBlobFromFrame(frame):
    #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    return blob

def ExtractDetectedFaces (blob,net_model):
    #input the blob into the net_model and get back the mat_detections from the page using net_model.forward()
    net_model.setInput(blob)
    mat_detections = net_model.forward()
    return mat_detections

def LoadPretrainedFaceRecognitionModel ():
    #import the pre trained face detection models provided in the OpenCV repository
    # https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
    # the weights have to be downloaded with this script: 
    # https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/download_weights.py
    print("Loading face detection deep neural network model ...")
    
    net_model = cv2.dnn.readNetFromCaffe('models/facedetection-deploy.prototxt', 'models/facedetection-weights.caffemodel')
    if net_model is None:
        print("Error getting pre-trained models, Check if weights.caffemodel and deploy.prototxt files exist")
        sys.exit()
    return net_model

def ReleaseResources (video_capture):
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def GetNewHeightMaintainRatio (desiredWidth,startX,endX,startY,endY):
    faceHeight = int(endY - startY)
    faceWidth = int(endX - startX)
    faceAspectRatio = faceWidth / faceHeight
    newHeight = int(desiredWidth/faceAspectRatio)
    return newHeight

def DrawBox(mat_detections,frame,i,count):
    (h,w) = frame.shape[:2];
    box = mat_detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = mat_detections[0, 0, i, 2]

    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
    facestr = 'Conf: {0:.{1}},Faces: {2}, pos:{3}'.format(  confidence,2, count,startX)

    #cv2.putText(frame, facestr , (startX + 6, startY - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    #Draw the detected face
    newWidth = 96
    newHeight = GetNewHeightMaintainRatio (newWidth,startX,endX,startY,endY)#int(newWidth/faceAspectRatio)

    face_image = frame[startY:endY, startX:endX] 

    if count == 1:
        xPos = 0
    else:
        xPos = (count-1)*newWidth

    if face_image.any() and count > 0:
        face_image = cv2.resize(face_image, (newWidth, newHeight))
        frame[30:newHeight+30, xPos:xPos + newWidth] = face_image
        # Some images have no size and create an error
        # so make sure you run the embedding after a resizing.
        #qembeddingVector = fr.CreateFaceEmbedding (face_image,embedder)

        #send face to face recog engine
        vec = fr.CreateFaceEmbedding (face_image,embedder)
        vec = embedder.forward()
        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = numpy.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
        if proba > 0.70:

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    if name is None:
        name = " "
    return face_image,name

def IterateOverDetectedFaces (mat_detections,frame):
    count = 0
    for i in range(0, mat_detections.shape[2]):
        # only grab faces with 60% confidence or more
        if mat_detections[0, 0, i, 2] > 0.6:
            count +=1
            face_box = DrawBox(mat_detections,frame,i,count)


def main():
    Init()
    CaptureVideo()

if __name__ == "__main__":
   main()


