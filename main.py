# main.py
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from pipeline import main as run_pipeline

if __name__ == "__main__":
    run_pipeline()