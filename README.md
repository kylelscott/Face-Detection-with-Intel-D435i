# Face-Detection-with-Intel-D435i #
Facial recognition with the Intel Realsense D435i Depth Camera. The main code for this project is a derivation of that found on the official Intel Realsense repositoy: https://github.com/IntelRealSense/librealsense

## To begin... ##
* Install pyrealsense2 using the directions stated on the librealsense repository:  https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

* I had librealsense installed for python 2.7.
* Once installed, run the python script using "python realsense_face_detection.py" with the model files (deploy.prototxt.txt and res10_300x300_ssd_iter_140000.caffemodel) in the same directory. 
## To end.... ##
* You now have a facial recognition solution using the Intel Realsense D435i (or any 400 series I believe) depth camera.
