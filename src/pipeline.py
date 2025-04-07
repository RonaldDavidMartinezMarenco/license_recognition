import os
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description="Pipeline de procesamiento de video.")
parser.add_argument("--video_path", type=str, required=True, help="Path al video de entrada.")
args = parser.parse_args()

# Obtener el intérprete de Python actual
python_executable = sys.executable

video_path = args.video_path

print("Ejecutando main.py...")
subprocess.run([python_executable, 'main.py','--video_path',video_path])


print("Ejecutando add_missing_data.py...")
subprocess.run([python_executable, 'add_missing_data.py'])

print("Ejecutando visualize.py...")
subprocess.run([python_executable, 'visualize.py','--video_path',video_path]) 



 