## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
#depth stream not needed
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Loading model")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
	if not color_frame:
            continue
            
	#begin timer to show frames per second
        start = time.time()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Begin the detection portion
        (h,w) = color_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0))
        net.setInput(blob, "data")

        detections = net.forward("detection_out")

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract confidence and prediction
            confidence = detections[0, 0, i, 2]

            # filter detections by confidence greater than minimum value
            print(confidence)
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box and write confidence
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(color_image, (startX, startY), (endX, endY),(255, 255, 255), 2)
                cv2.putText(color_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # show the output image
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Realsense", color_image)
        cv2.waitKey(1)

        #end timer to show frames per second
        end = time.time()
        sec = end-start
        print("time:{0}".format(sec))
	#show frames per second
        fps = 1/sec
        print("fps:{0}".format(fps))
finally:

    # Stop streaming
    pipeline.stop()
