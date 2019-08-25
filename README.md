# FaceRecognition
The purpose of this project is to customize and optimize machine learning Face recognition algorithms to be used with NVidia's Jeton Nano.

For now they all run in python, but the plan is to move them into C++.

Mainly I’ve created 4 main classes into a simple pipeline that allows me to test and optimize each Ai module to run smoothly with the slow jetson nano processor.

## FART
The FART modules are separated into files and instantiated as needed inside FaceDetection.py (Face Detection will be moved to its own file soon)

Face detection: Runs on every frame. It’s an OpenCV Ai algorithm for face detection.

Align: Runing only every 16 frames, this algorithm is used to improve recognition. The DLib Ai algorithm is based on the 1millisecond alignment algorithm published on 2014(https://www.semanticscholar.org/paper/One-millisecond-face-alignment-with-an-ensemble-of-Kazemi-Sullivan/1824b1ccace464ba275ccc86619feaa89018c0ad) 

Recognition: Also every 16 frames, OpenCV recognition algorithm using https://scikit-learn.org/

Tracking: Since is not processor intensive, I run it on every frame after recognition. Its one of the OpenCV tracking algorithms

This modularization allows me to exchange, remove and optimize the pipeline according to the needs of the platform, in this case the nano.
Working on the jetson nano and also on windows using a USB camera.

## setup:
jetson nano:

sudo sh archiconda.sh -b -p /opt/archiconda3

 now runnig conda3 install 
 
 sudo /opt/archiconda3/bin/conda install scikit-learn

## then 
 conda create -n [Virtualenv_Name] python=3.7 scikit-learn
 
 source activate py37
 
 then install opencv through the sh script
 
 sudo sh install_opencv4.0.0_Nano.sh

 ## wait ... it takes over an hour on the nano
 
 after this you will have scikit-learn installed on the virtual environment and also opencv 4.0.0

 remember to move cv2.so to cv2.so
 cd /usr/local/python/cv2/python-3.6
 

mv cv2.cpython-36m-aarch64-linux-gnu.so cv2.so

# after this conda will be located here:

/opt/archiconda3/bin/conda

# check it out with:

/opt/archiconda3/bin/conda --version

Conda allows you to create separate environments containing files, packages and their dependencies that will not interact with other environments. So I created a py37 environment with scikit-learn

## To dactivate 
To dactivate the py37 environment and return to base environment: conda activate


# windows:
 on windows what worked was 
 
py -m pip install opencv

py -m pip install scikit-learn

Also pip install dlib for the alignment algorithm

