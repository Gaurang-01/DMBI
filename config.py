from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Phishing_Legitimate_full.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
TARGET_COLUMN = "CLASS_LABEL"
ID_COLUMN = "id"
CLASS_NAMES = {0: "Legitimate", 1: "Phishing"}
