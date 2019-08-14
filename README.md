# FaceRecognition
Face Recognition Python script.
Working on the jetson nano and also on windows using a USB camera.

setup:
jetson nano:
sudo sh archiconda.sh -b -p /opt/archiconda3
# now runnig conda3 install 
# sudo /opt/archiconda3/bin/conda install scikit-learn
# then 
# conda create -n [Virtualenv_Name] python=3.7 scikit-learn
# source activate py3k
# then install opencv through the sh script
# sudo sh install_opencv4.0.0_Nano.sh
# wait ... it takes over an hour on the nano
# after this you will have scikit-learn installed on the virtual environment
# and also opencv 4.0.0
# remember to move cv2.so to cv2.so
# cd /usr/local/python/cv2/python-3.6
#
#mv cv2.cpython-36m-aarch64-linux-gnu.so cv2.so


windos:
# on windows what worked was 
py -m pip install opencv
py -m pip install scikit-learn


