import cv2
import numpy as np
import time
from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import get_car, read_license_plate,write_csv
from find_cameras import select_camera

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


CONFIDENCE_THRESHOLD = 0.75
CONFIDENCE_THRESHOLD_READ = 0.8

output_dir = "recovered_plates/real_time"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

camera_index = select_camera()
if camera_index is not None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    print(f"Usando la cámara con índice: {camera_index}")
else:
    exit()

car_tests = {}
coco_model = YOLO('../models/yolo11n.pt')
license_plate_detector = YOLO('../models/best.pt')
mot_tracker = Sort() #object tracker

vehicles = [2,3,5,7]

cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el cuadro.")
        break

    # Redimensionar el cuadro si es necesario

    # Detección de vehículos
    detections = coco_model(frame)[0]
    detections_ = []
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Seguimiento de vehículos
    if len(detections_) > 0:
        tracks_id = mot_tracker.update(np.asarray(detections_))
    else:
        tracks_id = []
        

    # Detección de matrículas
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        print(score)
        
        if(score > CONFIDENCE_THRESHOLD):
            # Asignar matrícula a un vehículo
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, tracks_id)
                
            # Dibujar cuadro delimitador del vehículo
            draw_border(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 10, line_length_x=200, line_length_y=200)
                
            # Dibujar cuadro delimitador de la matrícula
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)
            
            if car_id != -1:
                # Recortar la matrícula
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        
                # Procesar la matrícula
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Leer el texto de la matrícula
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Verificar si es la primera predicción o si el nuevo score es mayor que el almacenado
                if car_id not in car_tests:
                    # Primera predicción: solo se almacena si supera el umbral
                    if license_plate_text_score is not None and license_plate_text_score > CONFIDENCE_THRESHOLD_READ:
                        car_tests[car_id] = {
                            "text": license_plate_text,
                            "score": license_plate_text_score
                        }
                        print(f"Primera predicción para car_id {car_id}: {license_plate_text} (Confianza: {license_plate_text_score:.2f})")
                else:
                    # Actualizar solo si el nuevo score es mayor que el almacenado
                    if license_plate_text_score is not None and license_plate_text_score > car_tests[car_id]["score"] and license_plate_text != car_tests[car_id]["text"]:
                        car_tests[car_id] = {
                            "text": license_plate_text,
                            "score": license_plate_text_score
                        }       
                        print(f"Nuevo texto para car_id {car_id}: {license_plate_text} (Confianza: {license_plate_text_score:.2f})")
                
                if car_id in car_tests:             
                    stored_text = car_tests[car_id]["text"]
                    stored_score = car_tests[car_id]["score"]
                    
                    print(stored_text)
                    
                    output_path = os.path.join(output_dir, f"car_{car_id}_plate.jpg")
                    cv2.imwrite(output_path, license_plate_crop)
                    print(f"Matrícula guardada en: {output_path}")

                    
                    # Obtener el tamaño del texto
                    (text_width, text_height), _ = cv2.getTextSize(stored_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    

                    # Calcular posición del texto
                    frame_height, frame_width, _ = frame.shape
                    text_x = max(0, min(frame_width - text_width, int((x1 + x2 - text_width) / 2)))
                    text_y = min(frame_height - text_height, int(y2 + 30))  # Mostrar el texto debajo del cuadro de la matrícula

                    # Dibujar fondo blanco detrás del texto
                    background_x1 = text_x - 10
                    background_y1 = text_y - text_height - 10
                    background_x2 = text_x + text_width + 10
                    background_y2 = text_y + 10
                    cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), -1)

                    # Dibujar el texto de la matrícula
                    cv2.putText(frame, stored_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    
    # Mostrar el cuadro procesado
    cv2.imshow("Deteccion en tiempo real", frame)

    # Salir si el usuario presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

'''
1. Para simular la seguridad de prediccion del video, podemos hacer la prediccion no visible durante los primeros 5-10s, luego de dicho timepo se mostrara la prediccion en tiempo real, cual se calculara del maximo de las predicciones tomadas.
2. Para diferenciar el dia y la noche podremos o pasar por argumento el tipo (day-night) o hacer la deteccion automatica segun el brillo del frame, y determinar que filtro aplicar.

def is_night(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 50  # Umbral para determinar si es de noche

def preprocess_frame(frame, mode):
    if mode == 'night':
        # Aumentar brillo y contraste
        alpha = 1.5
        beta = 50
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Suavizar para reducir ruido
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame
'''