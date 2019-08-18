# Anel Salas 08/2019

import os   
import cv2
import numpy
import sys

# Remember that the class for the tracker 
# is not been destroyd automagically by OpenCV
# on init

class Tracker(object):
    mtracker =  None
    trackingBoundingBox = (0,0,0,0)

    def __init__(self):
        self.mtracker = cv2.TrackerCSRT_create() # CSRT, KCF,Boosting,MIL,TLD,MedianFlow,MOSSE
        self.trackingBoundingBox = (0,0,0,0)

    # returns the tracker object
    def start (self,frame):
        # again, Init DOES NOT inits the object properly
        # you need to delete this object when the tracked 
        # ROI is out of the frame
        self.mtracker.init(frame,self.trackingBoundingBox)

    # returns: (bool success, OpenCV 2D Box)
    def update (self,frame):
        return self.mtracker.update(frame)

def make_myTracker(startX,startY,endX,endY):
    tracker = Tracker()
    tracker.trackingBoundingBox=(startX,startY,endX-startX,endY-startY)
    return tracker
