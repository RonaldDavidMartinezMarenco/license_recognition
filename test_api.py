from ultralytics import YOLO
import cv2 
from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv

def real_time_camera(cap):
    
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al acceder a la camara.")
                break
            results = model(frame)
            frame = results[0].plot()    
            cv2.imshow("Deteccion de carros",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
                break
        cap.release()
        cv2.destroyAllWindows()
        
load_dotenv()

# Obtener la clave de API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("API_KEY")
)

video_path = "media/video.mp4"
cap = cv2.VideoCapture("media/video.mp4")

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    try:
        result = CLIENT.infer(frame, model_id="license-plate-recognition-rxg4e/6")
        
        for pred in result.get("predictions", []):
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            class_name = pred["class"]
            confidence = pred["confidence"]
            pred_id = pred["class_id"]

            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f} {pred_id}", (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar el frame con detecciones
        cv2.imshow("Detección en Video", frame)

    except Exception as e:
        print("error al conectar API.{e}")
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()