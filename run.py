import subprocess
import os

def build_executable(script_path, output_dir, icon_path=None):
    # Construct the PyInstaller command
    command = ['pyinstaller', '-F']
    
    if icon_path:
        command.extend(['-i', icon_path])
    
    command.extend([script_path, '--distpath', output_dir])
    
    # Run the PyInstaller command
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:")
        print(e.stderr)


script_path = 'ais.py'
desktop_path = os.path.expanduser("~/Desktop") 
icon_path = 'ais.ico'  

# Build the executable
build_executable(script_path, desktop_path, icon_path)
