# Anel Salas 07/19
# This script captures from a video source and sends each frame
# to the OpenCV Deep Neural Network Module instead of Dlib.
# This module is fast with CPUs and does not requires CUDA nor 
# GPUs. It is better than the face_recognition module from Dlib Geitgey
# only because it recognizes faces smaller than 80x80px.

# TODO: Compare the capture image with known faces


#import libraries
import os   
#print(os.__file__)
import cv2
import numpy
import sys
import platform


def GetVideoObject():
    usingplatform = platform.machine();
    print(usingplatform)
    video_capture = None
    if usingplatform  == "aarch64":
        print("Using Jetson Nano\n")
        video_capture = cv2.VideoCapture(0)
        #video_capture = cv2.VideoCapture("http://10.1.10.165:8081", cv2.CAP_GSTREAMER)
        #video_capture.open()
        if video_capture.isOpened() !=  True:  
            print("Cannot find camera, quitting")
            sys.exit()

    elif usingplatform  == "AMD64": 
        print("Running on a Windows System")
        video_capture = cv2.VideoCapture(1+cv2.CAP_DSHOW)
        if video_capture.isOpened() !=  True:  
            print("Cannot find camera, quitting")
            sys.exit()

    elif video_capture is None:
        print("Cannot find camera, quitting")
        sys.exit()
    
    return video_capture

def DlibUsingCuda():
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
    #print( platform.machine())
    #sys.exit()
    model = ReadModel()
    video_capture = GetVideoObject()
   
    #video_capture = cv2.VideoCapture(0)

   
    while True:
        ret, frame = video_capture.read()
        blob = GetBlobFromFrame(frame)
        detections = ExtractDetectedFaces(blob,model)
        IterateOverDetectedFaces (detections,frame)
        # Display frame 
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit
        keypressed = 0xFF & cv2.waitKey(1)
        if keypressed == ord('q') or keypressed == ord('Q') :
            break
    


def GetBlobFromFrame(frame):
    #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    return blob

def ExtractDetectedFaces (blob,model):
    #input the blob into the model and get back the detections from the page using model.forward()
    model.setInput(blob)
    detections = model.forward()
    return detections

def ReadModel ():
    #import the models provided in the OpenCV repository
    model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
    if model is None:
        print("Error getting pre-trained models, Check if weights.caffemodel and deploy.prototxt files exist")
        sys.exit()
    return model

def ReleaseResources (video_capture):
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def DrawBox(detections,frame,i,w,h,count):        
    box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]
    #only draw box if confidence > 16.5
    #original algo checked on 165 but works better at 30% in my tests
    if (confidence > 0.17):
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        #cv2.rectangle(frame, (startX, startY - 35), (endX, endY), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "face", (startX + 6, startY - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        count = count + 1
        #frame[30:180, startX:startX + 150,startY] = box^M

def IterateOverDetectedFaces (detections,frame):
    # extract their start and end points
    count = 0
    (h,w) = frame.shape[:2];
    for i in range(0, detections.shape[2]):
        DrawBox(detections,frame,i,w,h,count)

def main():
    print("OpenCV Version: {}".format(cv2.__version__))
    OpenCVUsingCuda()
    CaptureVideo()

if __name__ == "__main__":
   main()


