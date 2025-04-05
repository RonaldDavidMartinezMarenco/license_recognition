from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import get_car, read_license_plate,write_csv

parser = argparse.ArgumentParser(description="Visualize license plate detection.")
parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
args = parser.parse_args()

video_path = args.video_path

results = {}
coco_model = YOLO('../models/yolo11n.pt')
license_plate_detector = YOLO('../models/best.pt')
mot_tracker = Sort() #object tracker

cap = cv2.VideoCapture(video_path)

vehicles = [2,3,5,7]

ret = True

#detecting vehicles and specify de class_id
frame_nmr = -1
while ret :
    frame_nmr += 1
    ret,frame = cap.read()
    if ret:        
        results[frame_nmr] = {}
        detections = coco_model(frame)[0]
        detections_ = []
        for det in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = det
            if int(class_id) in vehicles:
                detections_.append([(x1),(y1),(x2),(y2),score]) 
            
            
        #tracking vehicles with sort or deep sort, also you can use yolo tracking
        if len(detections_)>0:
             tracks_id = mot_tracker.update(np.asarray(detections_)) #contains bounding boxes and tracking information, also you can use YOLO
        else:
            tracks_id = []
        
        #detect license plates
        license_plates= license_plate_detector(frame)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = license_plate

            #assing specific license plate to a vehicle
            xcar1,ycar1,xcar2,ycar2,car_id = get_car(license_plate,tracks_id) #return the vehicle the license plate belongs to


            #crops the license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :] #cut the area of the rectangle

            if car_id != -1:
                #process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) #convert to a grayscale image to apply threshold
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV) #apply threshold, pixel that are less than 64 will be 255 and viceversa
                #read license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh) #return the license plate and the confidence score
                if license_plate_text is not None:
                                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                'text': license_plate_text,
                                                                                'bbox_score': score,
                                                                                'text_score': license_plate_text_score}}
#write results
write_csv(results,'../outputs/test.csv')    
    


    






            