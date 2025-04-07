import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys

# Obtener el intérprete de Python actual
python_executable = sys.executable

def seleccionar_video():
    """Abrir un cuadro de diálogo para seleccionar un archivo de video."""
    video_path = filedialog.askopenfilename(
        title="Seleccionar video",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv")]
    )
    if video_path:
        video_label.config(text=f"Video seleccionado: {video_path}")
        return video_path
    else:
        messagebox.showwarning("Advertencia", "No se seleccionó ningún video.")
        return None

def ejecutar_deteccion():
    """Ejecutar el script de detección en tiempo real."""
    try:
        messagebox.showinfo("Información", "Iniciando detección en tiempo real...")
        subprocess.run([python_executable, 'real_time_detection.py', ], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Error al ejecutar la detección: {e}")

def ejecutar_procesamiento():
    """Ejecutar el pipeline completo."""
    video_path = seleccionar_video()
    if video_path:
        try:
            messagebox.showinfo("Información", "Iniciando procesamiento del pipeline...")
            subprocess.run([python_executable, 'pipeline.py', '--video_path', video_path], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar el pipeline: {e}")

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Interfaz de Detección de Matrículas")
ventana.geometry("600x300")

# Etiqueta para mostrar el video seleccionado
video_label = tk.Label(ventana, text="No se ha seleccionado ningún video.", wraplength=500)
video_label.pack(pady=10)

# Botón para ejecutar la detección en tiempo real
btn_deteccion = tk.Button(ventana, text="Detección en Tiempo Real", command=ejecutar_deteccion, bg="lightblue", fg="black")
btn_deteccion.pack(pady=10)

# Botón para ejecutar el pipeline completo
btn_procesamiento = tk.Button(ventana, text="Procesar Video (Pipeline)", command=ejecutar_procesamiento, bg="lightgreen", fg="black")
btn_procesamiento.pack(pady=10)

# Botón para salir de la aplicación
btn_salir = tk.Button(ventana, text="Salir", command=ventana.quit, bg="red", fg="white")
btn_salir.pack(pady=10)

# Iniciar el bucle principal de la interfaz
ventana.mainloop()