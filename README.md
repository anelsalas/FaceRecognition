# FaceRecognition
Face Recognition Python script.

Working on the jetson nano and also on windows using a USB camera.

## setup:
jetson nano:
sudo sh archiconda.sh -b -p /opt/archiconda3

 now runnig conda3 install 
 
 sudo /opt/archiconda3/bin/conda install scikit-learn

## then 
 conda create -n [Virtualenv_Name] python=3.7 scikit-learn
 
 export PATH="/opt/archiconda3/bin:$PATH"
 
 conda info --envs
 
 source activate py3k
 
 then install opencv through the sh script
 
 sudo sh install_opencv4.0.0_Nano.sh

## wait ... it takes over an hour on the nano
 after this you will have scikit-learn installed on the virtual environment
 and also opencv 4.0.0

## remember to move cv2.so to cv2.so
 cd /usr/local/python/cv2/python-3.6

## mv cv2.cpython-36m-aarch64-linux-gnu.so cv2.so

## after this conda will be located here:
/opt/archiconda3/bin/conda

## check it out with:
/opt/archiconda3/bin/conda --version

## Conda allows you to create separate environments containing files, packages and their dependencies that will not interact with other environments. So I created a py37 environment with scikit-learn


## windows:
 on windows what worked was 
 
py -m pip install opencv

py -m pip install scikit-learn


