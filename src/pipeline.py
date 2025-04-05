import os
import subprocess
import sys

# Obtener el intérprete de Python actual
python_executable = sys.executable

video_path = "../samples/CarrosBalcones4.mp4"

print("Ejecutando main.py...")
subprocess.run([python_executable, 'main.py','--video_path', video_path])


print("Ejecutando add_missing_data.py...")
subprocess.run([python_executable, 'add_missing_data.py'])

print("Ejecutando visualize.py...")
subprocess.run([python_executable, 'visualize.py','--video_path', video_path]) 



 