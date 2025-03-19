import cv2
import matplotlib.pyplot as plt
import easyocr
import cv2
import easyocr
import matplotlib.pyplot as plt

def process_and_show_image(image_path):
    """
    Toma una imagen, la convierte a escala de grises, aplica un umbral inverso,
    extrae el texto con EasyOCR y lo muestra en consola.

    Args:
        image_path (str): Ruta de la imagen a procesar.
    """
    # Leer la imagen desde la ruta
    image = cv2.imread(image_path)

    if image is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta.")
        return

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral inverso con Otsu
    _, thresh_image = cv2.threshold(gray_image, 64, 255, cv2.THRESH_BINARY_INV )

    # Inicializar EasyOCR
    reader = easyocr.Reader(['en'])

    # Leer el texto de la imagen procesada
    result = reader.readtext(thresh_image)

    # Extraer solo el texto detectado
    detected_text = " ".join([text for (_, text, _) in result])

    if detected_text:
        print(f"Texto detectado: {detected_text}")
    else:
        print("No se detectó texto.")

    # Mostrar las imágenes
    plt.figure(figsize=(10, 5))

    # Imagen en escala de grises
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Escala de Grises")
    plt.axis("off")

    # Imagen con umbral inverso
    plt.subplot(1, 2, 2)
    plt.imshow(thresh_image, cmap="gray")
    plt.title("Umbral Inverso")
    plt.axis("off")

    plt.show()


process_and_show_image("c:/Users/ronal/license_recognition/samples/colombia4.jpg")