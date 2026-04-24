from pathlib import Path

# Change PROJECT_ROOT if your extracted class project folder lives somewhere else.
PROJECT_ROOT = Path("/Users/nikhilkanaparthi/Desktop/Data 245 Project/Predicting-Public-Transit-Reliability-Using-Machine-Learning")

DATA_CLEANED = PROJECT_ROOT / 'data' / 'cleaned'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'

FINAL_DATA_CSV = DATA_CLEANED / 'vta_final.csv'
STOPS_CSV = DATA_RAW / 'stops.txt'

SOLUTION_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = SOLUTION_ROOT / 'outputs'
MODELS_DIR = SOLUTION_ROOT / 'models'

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
