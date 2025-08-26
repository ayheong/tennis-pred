# src/models/predict.py
import json, joblib
from pathlib import Path
from src.features.build_features import load_encoders

# resolve project root (src/models → repo root)
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

def load_artifacts():
    ensemble = MODELS_DIR / "lgb_ensemble.pkl"
    single   = MODELS_DIR / "model_lgb.pkl"

    if ensemble.exists():
        models = joblib.load(ensemble)
        if not isinstance(models, (list, tuple)):  # safety
            models = [models]
        model_path_used = str(ensemble)
    elif single.exists():
        models = [joblib.load(single)]
        model_path_used = str(single)
    else:
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"No model file found.\n"
            f"Looked for:\n  {ensemble}\n  {single}\n"
            f"Current working dir: {cwd}"
        )

    encs_path = MODELS_DIR / "encoders.json"
    thr_path  = MODELS_DIR / "threshold.json"

    encs = load_encoders(encs_path) if encs_path.exists() else {}
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())["threshold"]
    else:
        thr = 0.5  # fallback

    return models, encs, thr
