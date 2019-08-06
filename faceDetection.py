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
import dlib
import dlibFaceRecognition
import pprint
import face_recognition

'''
All these stupid globals are because there is no constructor available for 
mmod_rectangle which is returned inside the mmod_rectangles object.
From the documentation:
   This detector (cnn_face_detector) returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
   These objects can be accessed by simply iterating over the mmod_rectangles object
   The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

   http://dlib.net/cnn_face_detector.py.html says:
   "You can get the mmod_human_face_detector.dat file from:\n"
   "    http://dlib.net/files/mmod_human_face_detector.dat.bz2"
'''
'''
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector1.dat")
img = dlib.load_rgb_image("obama_small.jpg")
mmod_rects = cnn_face_detector(img,1)
mmod_singleRectangle = mmod_rects.pop()

dlibFaceRecognition.ClearListOfDetectedFaces (mmod_rects)
'''

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
        #print("Using Jetson Nano\n")
        video_capture = cv2.VideoCapture(0)
        #video_capture = cv2.VideoCapture("http://10.1.10.165:8081", cv2.CAP_GSTREAMER)
        #video_capture.open() # no need to open! it seems capture does that already
        if video_capture.isOpened() !=  True:  
            print("Cannot find camera, quitting")
            sys.exit()

    elif usingplatform  == "AMD64": 
        #print("Running on a Windows System")
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
    net_model = ReadModel()
    video_capture = GetVideoObject()
    process_this_frame = 0
    while True: 
        # process each frame
        ret, frame = video_capture.read()
        blob = GetBlobFromFrame(frame)
        mat_detections = ExtractDetectedFaces(blob,net_model)
        # for this to work with gey guy face_recognition, I need to return
        # a list of tuples of found face locations in css (top, right, bottom, left) order
        #dlib_mmod_rects = IterateOverDetectedFaces (mat_detections,frame)
        IterateOverDetectedFaces (mat_detections,frame)
        #geyguyFaceLocations = dlibFaceRecognition.GeyGuyFormatedDetectedFaces(dlib_mmod_rects,frame)

        #if process_this_frame == 7:
        #    dlib_face_encodings = dlibFaceRecognition.GeyGuyFaceEncodings(geyguyFaceLocations,frame)

        #face_encodings = face_recognition.face_encodings(frame, detected_faces)
        # Display frame 
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit
        keypressed = 0xFF & cv2.waitKey(1)
        if keypressed == ord('q') or keypressed == ord('Q') :
            break
        process_this_frame += 1
        if process_this_frame == 8:
            process_this_frame = 0
# The following functions are used by geyguy so I'm putting them as reference

###############################################################################
'''
def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)


def ConvertDetectionsForGeyGuyCode ():
     if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]
'''
###############################################################################


def GetBlobFromFrame(frame):
    #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    return blob

def ExtractDetectedFaces (blob,net_model):
    #input the blob into the net_model and get back the mat_detections from the page using net_model.forward()
    net_model.setInput(blob)
    mat_detections = net_model.forward()
    return mat_detections

# TODO:
"""
What I need to do is to be able to return the same that face_recognition.face_locatinos(rgb_small_frame) returns
then pass that to face_recofntion.face_encondings()
        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_recognition.face_locations() does this:
            return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]

            and _raw_face_locations() does this:

            cnn_face_detector(img, number_of_times_to_upsample)
                '''
                This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
                These objects can be accessed by simply iterating over the mmod_rectangles object
                The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
                Sooo: mmod_rectangle = [dlib.rectangle, confidencescore]
                
                It is also possible to pass a list of images to the detector.
                    - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
            
                In this case it will return a mmod_rectangless object.
                This object behaves just like a list of lists and can be iterated over.
                '''
"""

def ReadModel ():
    #import the models provided in the OpenCV repository
    net_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
    if net_model is None:
        print("Error getting pre-trained models, Check if weights.caffemodel and deploy.prototxt files exist")
        sys.exit()
    return net_model

def ReleaseResources (video_capture):
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def DrawBox(mat_detections,frame,i,w,h,count):        
    box = mat_detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = mat_detections[0, 0, i, 2]
    #only draw box if confidence > 16.5
    #original algo checked on 165 but works better at 30% in my tests
    if (confidence > 0.17):
        mydbrect = dlib.rectangle(startX,startY,endX-1,endY-1)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        #cv2.rectangle(frame, (startX, startY - 35), (endX, endY), (0, 0, 255), cv2.FILLED)
        #face = 'face : {0}, {1}, {2}, {3}, {4}'.format(  startX, mydbrect.bottom(), startY, endX, endY  )
        face = 'confidence : {0:.{1}}'.format(  confidence, 3 )
        cv2.putText(frame, face , (startX + 6, startY - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        count = count + 1
        #frame[30:180, startX:startX + 150,startY] = box^M

    return box




def IterateOverDetectedFaces (mat_detections,frame):
    # extract their start and end points
    count = 0
    (h,w) = frame.shape[:2];
    for i in range(0, mat_detections.shape[2]):
        box = DrawBox(mat_detections,frame,i,w,h,count)
    
        #convert to dlib.rectangle
        #actually I need to convert to dlib.mmod_rectangles
        #dlib_rectangle = dlibFaceRecognition.OpenCVRectangleToDlibRectangle(box)
        #mmod_singleRectangle.confidence =  mat_detections[0, 0, i, 2]
        #mmod_singleRectangle.rect = dlib_rectangle
        

        #mmod_rects.append(mmod_singleRectangle)

    #return mmod_rects

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #small_frame = frame #cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
        #rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        #face_locations = face_recognition.face_locations(rgb_small_frame)#,2,"hog")
        #face_encodings = dlibFaceRecognition.face_recognition.face_encodings(frame, mmod_rectangles)
        #face_encodings = face_recognition.face_encodings(frame, mmod_rects)




def main():
    Init()
    dlibFaceRecognition.load_known_faces()
    CaptureVideo()

if __name__ == "__main__":
   main()


