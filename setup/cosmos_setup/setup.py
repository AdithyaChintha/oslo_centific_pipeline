
import subprocess
import os

def setup_cosmos():
    """
    Sets up the environment for Cosmos scene detection.
    """
    # Install git-lfs
    subprocess.run(["sudo", "apt-get", "install", "git-lfs"], check=True)
    subprocess.run(["git", "lfs", "install"], check=True)

    # Clone the model repository if it doesn't exist
    if not os.path.exists("models/Cosmos"):
        subprocess.run(["git", "clone", "https://huggingface.co/microsoft/Cosmos"], cwd="models", check=True)
