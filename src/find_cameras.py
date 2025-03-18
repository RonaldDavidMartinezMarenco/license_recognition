import cv2

def find_available_cameras():
    """ Escanea los puertos de la cámara y devuelve una lista con los índices y nombres. """
    available_cameras = []
    for index in range(5):  # Probar los primeros 5 puertos
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            name = f"Camara {index}"
            available_cameras.append((index, name))
            cap.release()
    return available_cameras

# Buscar cámaras disponibles
cameras = find_available_cameras()

if not cameras:
    print("No se encontraron camaras disponibles.")
else:
    print("Camaras disponibles:")
    for idx, name in cameras:
        print(f"[{idx}] - {name}")

    # Elegir la cámara
    selected = int(input("\nIngrese el numero de la camara que desea usar: "))

    if selected not in [cam[0] for cam in cameras]:
        print("Camara no válida.")
    else:
        cap = cv2.VideoCapture(selected, cv2.CAP_DSHOW)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al acceder a la camara.")
                break

            cv2.imshow(f"Camara {selected}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
                break

        cap.release()
        cv2.destroyAllWindows()
