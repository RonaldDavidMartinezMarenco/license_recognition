import ast
import os
import cv2
import numpy as np
import pandas as pd
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Visualize license plate detection.")
parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
args = parser.parse_args()

output_dir = "recovered_plates/videos"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

results = pd.read_csv('../outputs/test_interpolated.csv')

# load video
video_path = args.video_path

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = '../outputs/out.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    print(f"Car ID: {car_id}, License Plate Number: {license_plate[car_id]['license_plate_number']}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop
    output_path = os.path.join(output_dir, f"car_{car_id}_plate.jpg")
    cv2.imwrite(output_path, license_crop)
    print(f"License crop saved at: {output_path}")


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 10,
                        line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)
              
            try:
                # Obtener el tamaño del texto
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,  # Tamaño reducido
                    2)  # Grosor del texto

                # Calcular posición del texto
                frame_height, frame_width, _ = frame.shape
                text_x = max(0, min(frame_width - text_width, int((car_x2 + car_x1 - text_width) / 2)))
                text_y = min(frame_height - text_height, int(car_y2 + 20))  # Mostrar el texto debajo del cuadro del vehículo

                print(f"Text: {license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']}, Position: ({text_x}, {text_y})")
                
                #Dibujar fondo blanco
                background_x1 = text_x - 10
                background_y1 = text_y - text_height - 10
                background_x2 = text_x + text_width + 10
                background_y2 = text_y + 10
                
                cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), -1)
               
                print(f"Text: {license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']}, Position: ({text_x}, {text_y})")
                
                # Dibujar el texto en el cuadro
                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,  # Tamaño reducido
                            (0, 0, 0),
                            2)  # Grosor del texto
            except Exception as e:
                print(f"Error processing license crop for Car ID {df_.iloc[row_indx]['car_id']}: {e}")        
                
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)      
out.release()
cap.release()


print("Abriendo el video procesado con el reproductor predeterminado...")
try:
    subprocess.run(["start", output_video_path], shell=True, check=True)  # Para Windows
except Exception as e:
    print(f"Error al intentar abrir el video: {e}")