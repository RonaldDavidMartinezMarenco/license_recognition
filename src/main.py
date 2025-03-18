from ultralytics import YOLO
import cv2

coco_model = YOLO("../models/yolo11n.pt")
license_plate_detector = YOLO("../models/best.pt")
vehicles = [2,3,5,7]
#Load video
cap = cv2.VideoCapture("../sample/cars.mp4")
ret = True
frame_n = -1 
while ret :
    frame_n += 1
    ret, frame = cap.read()
    if ret and frame_n < 10:
        pass
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
                detections_.append([ x1, y1, x2, y2, score])
    
    