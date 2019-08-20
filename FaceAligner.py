# Anel Salas 2019
# you can find shape predictor shape_predictor_68_face_landmarks.dat here:
# http://dlib.net/files/

#from .helpers import FACIAL_LANDMARKS_IDXS
#from .helpers import shape_to_np
from collections import OrderedDict
import numpy as np
import cv2
import dlib


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

 
class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # dictionary that maps the indexes of the facial
        # landmarks to specific face region
        #For dlib’s 68-point facial landmark detector:
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
        	("mouth", (48, 68)),
        	("inner_mouth", (60, 68)),
        	("right_eyebrow", (17, 22)),
        	("left_eyebrow", (22, 27)),
        	("right_eye", (36, 42)),
        	("left_eye", (42, 48)),
        	("nose", (27, 36)),
        	("jaw", (0, 17))
        ])
 
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def convertOpencvMatToDLibRect (self,image):
        image = resize(image, width=800)
        (he,wi)=image.shape[:2]
        rect = dlib.rectangle( left= 0, top= 0, right=800, bottom= he)
        return image,rect

    def extractEyesCoordinates (self,shape):
        (lStart, lEnd) = self.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        return leftEyePts,rightEyePts

    def getCenterOfMassOfEyes (self,leftEyePts,rightEyePts):
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        return leftEyeCenter,rightEyeCenter

    def getAngleOfEyeCentroids (self,leftEyeCenter,rightEyeCenter):
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        return angle,dX,dY

    def getNewScaleOfImage (self,dX,dY,desiredRightEyeX):
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        return scale


    # image_small : The RGB input image.
    # gray : The grayscale input image.
    # rect : The bounding box rectangle produced by dlib’s HOG face detector.
    def align(self, image_small):
        image,rect = self.convertOpencvMatToDLibRect (image_small)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        leftEyePts,rightEyePts = self.extractEyesCoordinates(shape)

        leftEyeCenter,rightEyeCenter = self.getCenterOfMassOfEyes(leftEyePts,rightEyePts)
 
        angle,dX,dY = self.getAngleOfEyeCentroids (leftEyeCenter,rightEyeCenter)

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        scale = self.getNewScaleOfImage (dX,dY,desiredRightEyeX)

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
 
        # return the aligned face
        return output
