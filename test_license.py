import cv2 as cv
from glob import glob
import os
import random
from ultralytics import YOLO

# pick pre-trained model
np_model = YOLO('models/best.pt')

# read video by index
video = cv.VideoCapture("samples/colombiaxde.mp4")
ret, frame = video.read()

#video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
#video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
#video.set(cv.CAP_PROP_FPS, 15)  # Baja los FPS para reducir carga

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
# Asegúrate de que el directorio existe

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./outputs/detection.avi', fourcc, 60.0, size)

# read frames
ret = True

while ret:
    ret, frame = video.read()

    if ret:
        # detect & track objects
        results = np_model.track(frame, persist=True, conf=0.5)

        # plot results
        composed = results[0].plot()
        #cv.imshow('Real-Time Detection', composed)
        # save video
        out.write(composed)
        
        #if cv.waitKey(1) & 0xFF == ord('q'):
           # break

out.release()
video.release()