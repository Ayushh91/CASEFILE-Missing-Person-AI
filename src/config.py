from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_GEOLIFE = DATA / "raw" / "GeoLife"
PROCESSED = DATA / "processed"
SYNTHETIC = DATA / "synthetic"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42
for folder in (RAW_GEOLIFE, PROCESSED, SYNTHETIC, MODELS, REPORTS / "outputs"):
    folder.mkdir(parents=True, exist_ok=True)
