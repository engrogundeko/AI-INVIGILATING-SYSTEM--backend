import time
import os
import subprocess
import webbrowser

# from pathlib import Path

# Load environment variables from .env file
# dot_env_path = Path(__file__).parent / ".env"
# dotenv.load_dotenv(dotenv_path=dot_env_path)

os.environ["DIRECTORY"] = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\web_app"
os.environ["ENV_PATH"] = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\venv"

def open_directory_and_run_script(directory, env_path, script_name=None):
    # Get the path to the Python interpreter and uvicorn executable
    python_executable = os.path.join(env_path, "Scripts", "python.exe")
    uvicorn_executable = os.path.join(env_path, "Scripts", "uvicorn.exe")

    # Change to specified directory
    os.chdir(directory)
    
    # Run uvicorn using the virtual environment's Python interpreter
    process = subprocess.Popen([python_executable, uvicorn_executable, "offline.main:app", "--reload"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Delay to ensure server starts up
    time.sleep(60)
    
def open_browser_with_url(url):
    webbrowser.open(url, new=2)

if __name__ == "__main__":
    env_path = os.environ.get("ENV_PATH")
    directory = os.environ.get("DIRECTORY")

    if not env_path or not directory:
        raise ValueError(
            "Both ENV_PATH and DIRECTORY must be set in the environment variables."
        )

    url = "http://127.0.0.1:8000/docs"
    open_directory_and_run_script(directory, env_path)
    open_browser_with_url(url)
