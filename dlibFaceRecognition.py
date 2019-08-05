import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import dlib
import os


known_face_encodings = []
knon_face_metadata = []
pose_predictor_68_point = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#os.getcwd())


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass

# OpenCV assumes that right and bottom boundaries are NOT inclusive
# while Dlib assumes they are
def OpenCVRectangleToDlibRectangle (box):
    #mydbrect.top() is equal to startY
    #mydbrect.left() is equal to startX
    #mydbrect.right() is equal to endX
    #mydbrect.bottom() is equal to endY
    (startX, startY, endX, endY) = box.astype("int")
    dlibRectangle = dlib.rectangle(startX,startY,endX-1,endY-1)
    #return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]

    return dlibRectangle

def ClearListOfDetectedFaces (mmod_rects):
    for mmodrect in mmod_rects:
        mmod_rects.remove(mmodrect)
        print("removed one item")

def GeyGuyFormatedDetectedFaces (dlib_mmod_rects,img):
    #img = dlib.load_rgb_image(frame)

    #return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]

    #max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
    #return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in dlib_mmod_rects]
    return [_rect_to_css(face.rect) for face in dlib_mmod_rects]

def GeyGuyFaceEncodings (known_face_locations,img):
    #dlib.rectangle(faceLocs[3], faceLocs[0], faceLocs[1], faceLocs[2])
    face_locations = [dlib.rectangle(faceLocs[3], faceLocs[0], faceLocs[1], faceLocs[2]) for faceLocs in known_face_locations]
    #path = os.getcwd() + "shape_predictor_68_face_landmarks.dat"
    #raw_face_landmars =  [pose_predictor_68_point(img, face_location) for face_location in face_locations]
    print("only locations done")

    #raw_landmarks = _raw_face_landmarks(img, known_face_locations)
    #return [np.array(face_qencoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]



def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])



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
    return max(css[0], 0), min(css[1],image_shape[1]),min(css[2],image_shape[0]),max(css[3], 0)

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



def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]
