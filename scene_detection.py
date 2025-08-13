import subprocess
import os
from pathlib import Path

# ===== CONFIGURATION =====
VIDEO_PATH = "/home/nvcoe_admin/scenedetection/VID_20250720_152154_00_011.mp4"
REPO_URL = "https://github.com/nvidia-cosmos/cosmos-reason1.git"
REPO_DIR = Path.home() / "cosmos-reason1"
PROMPT_FILE = REPO_DIR / "prompts" / "event_segmentation.yaml"
HF_TOKEN = "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"  # Hugging Face token
# =========================

def run_cmd(cmd, cwd=None, shell=False):
    """Run a shell command with live output."""
    print(f"\n[CMD] {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    subprocess.run(cmd, shell=shell, cwd=cwd, check=True)

# def ensure_prompt_file():
#     PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
#     if not PROMPT_FILE.exists():
#         print(f"[+] Creating prompt file at {PROMPT_FILE}")
#         PROMPT_FILE.write_text(
# """system_prompt: |
#   Analyze the video and identify all distinct events or actions that occur. 
#   For each event:
#   - Provide a concise description.
#   - Include a start time and end time in HH:MM:SS format.
#   - Each event should be a complete, self-contained action.
#   Output a JSON object with a single key 'events', containing a list of objects 
#   with 'start_time', 'end_time', and 'description'.

# user_prompt: |
#   Video path: {{video_path}}
# """
#         )

def ensure_prompt_file():
    PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not PROMPT_FILE.exists():
        print(f"[+] Creating prompt file at {PROMPT_FILE}")
        PROMPT_FILE.write_text(
"""system_prompt: |
  You are a video captioning specialist analyzing an Insta360 equirectangular panoramic video.
  The video has distortion near poles and edges and shows multiple overlapping views.
  Carefully analyze the video and identify all distinct events or actions that occur,
  despite the distortions and multiple views.
  For each event:
    - Provide a concise description.
    - when ever you saw any person draw a bounding box around them and give me that particular video scene as output
    - Include start and end times in HH:MM:SS.ss format (hours:minutes:seconds.milliseconds).
    - Each event should be a complete, self-contained action.
  Output a JSON object with a single key 'events', containing a list of objects 
  with 'start_time', 'end_time', and 'description'.

user_prompt: |
  Video path: {{video_path}}
"""
        )


def main():
    # Add Hugging Face token to environment
    os.environ["HF_TOKEN"] = HF_TOKEN

    # Ensure ~/.local/bin is in PATH
    os.environ["PATH"] = f"{Path.home()}/.local/bin:" + os.environ["PATH"]

    # 1) Install system dependencies
    run_cmd(["sudo", "apt", "update"])
    run_cmd([
        "sudo", "apt", "install", "-y",
        "build-essential", "git", "git-lfs", "curl", "wget", "ffmpeg",
        "pkg-config", "python3", "python3-venv", "python3-pip", "cmake"
    ])
    run_cmd(["git", "lfs", "install"])

    # 2) Install pkgx (optional)
    run_cmd("brew install pkgx || curl https://pkgx.sh | sh", shell=True)

    # 3) Install astral uv
    run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)

    # 4) Install huggingface_hub CLI without interaction
    run_cmd(["uv", "tool", "install", "-U", "huggingface_hub[cli]"])

    # 5) Non-interactive Hugging Face login
    run_cmd(["hf", "auth", "login", "--token", HF_TOKEN, "--add-to-git-credential"])

    # 6) Clone or update repo
    if not REPO_DIR.exists():
        run_cmd(["git", "clone", REPO_URL, str(REPO_DIR)])
    else:
        run_cmd(["git", "pull"], cwd=REPO_DIR)

    # 7) Upgrade pip/setuptools/wheel
    run_cmd(["pip", "install", "-U", "pip", "setuptools", "wheel"])

    # 8) Install requirements
    #run_cmd("pip install -r requirements.txt", cwd=REPO_DIR, shell=True)

    # 9) Extra packages
    run_cmd([
        "pip", "install",
        "transformers", "vllm", "accelerate", "sentencepiece",
        "opencv-python", "pyyaml", "numpy", "ffmpeg-python"
    ])

    # 10) Ensure prompt file exists
    ensure_prompt_file()

    # 11) Run inference
    output_json = REPO_DIR / "output_events.json"
    run_cmd([
        "uv", "run", "scripts/inference.py",
        "--prompt", "prompts/event_segmentation.yaml",
        "--videos", VIDEO_PATH,
        "-v"
    ], cwd=REPO_DIR)

    # 12) Save output to file
    with open(output_json, "w") as f:
        subprocess.run([
            "uv", "run", "scripts/inference.py",
            "--prompt", "prompts/event_segmentation.yaml",
            "--videos", VIDEO_PATH,
            "-v"
        ], cwd=REPO_DIR, stdout=f, check=True)

    print(f"\n[+] Inference complete. Results saved to: {output_json}")

if __name__ == "__main__":
    main()
