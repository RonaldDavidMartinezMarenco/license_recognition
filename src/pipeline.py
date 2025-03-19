import os
import subprocess
import sys

# Obtener el intérprete de Python actual
python_executable = sys.executable

# Paso 1: Ejecutar main.py
print("Ejecutando main.py...")
subprocess.run([python_executable, 'main.py'])

# Paso 2: Ejecutar add_missing_data.py
print("Ejecutando add_missing_data.py...")
subprocess.run([python_executable, 'add_missing_data.py'])

# Paso 3: Ejecutar visualize.py
print("Ejecutando visualize.py...")
subprocess.run([python_executable, 'visualize.py'])