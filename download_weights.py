"""
Download and extract the model weights during setup.

Usage:
    python3 download_weights.py
"""

from pathlib import Path
from scripts.f_environment import verify_results_weights_folder

if __name__ == "__main__":
    pwd = Path.cwd()
    print(f"Checking/downloading weights into: {pwd / 'weights'}")
    verify_results_weights_folder(pwd)
    print("Done.")