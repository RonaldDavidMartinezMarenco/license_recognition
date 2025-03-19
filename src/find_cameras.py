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

def select_camera():
    """ Permite al usuario seleccionar una cámara y devuelve su índice. """
    cameras = find_available_cameras()

    if not cameras:
        print("No se encontraron cámaras disponibles.")
        return None
    else:
        print("Cámaras disponibles:")
        for idx, name in cameras:
            print(f"[{idx}] - {name}")

        # Elegir la cámara
        while True:
            try:
                selected = int(input("\nIngrese el número de la cámara que desea usar: "))
                if selected in [cam[0] for cam in cameras]:
                    return selected
                else:
                    print("Cámara no válida. Intente nuevamente.")
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número.")

# Ejemplo de uso
if __name__ == "__main__":
    camera_index = select_camera()
    if camera_index is not None:
        print(f"Se seleccionó la cámara con índice: {camera_index}")