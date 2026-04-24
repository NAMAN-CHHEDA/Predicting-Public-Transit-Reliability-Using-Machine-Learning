from pathlib import Path

# Repository root = folder that contains this file (portable for all teammates)
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

FINAL_DATA_CSV = DATA_CLEANED / "vta_final.csv"
STOPS_CSV = DATA_RAW / "stops.txt"

# Models and run outputs live next to the code (not under data/)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Backwards compatibility: some imports may use this name
SOLUTION_ROOT = PROJECT_ROOT
