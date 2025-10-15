Tennis Match Predictor API

This is a lightweight ML application where users select two tennis players and get a win probability, a recommended pick, and basic insights. It’s built with FastAPI, uses historical ATP match data, and applies custom engineered features (rank/age/height deltas, surface, recent form, H2H priors, Bradley–Terry feature matrix). It runs a simple ensemble of LightGBM models for stable predictions and includes a minimal UI.

---

Quickstart (Local)

1) Clone
    git clone https://github.com/<you>/tennis-pred.git
    cd tennis-pred

2) (Recommended) Create & activate venv
  Windows (PowerShell)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
  macOS/Linux
    python -m venv .venv
    source .venv/bin/activate

3) Install deps
    pip install -r requirements.txt

4) Run the API (from repo root)
    uvicorn src.app:app --reload
  Open: http://127.0.0.1:8000/docs

---

Using Docker

Build the image (from repo root):
    docker build -t tennis-pred .

Run the container (map port 8000):
    docker run --rm -p 8000:8000 tennis-pred

Open: http://127.0.0.1:8000/docs

---

CLI 
Run a one-off prediction without the server:
    python -m src.models.predict_names "Novak Djokovic" "Carlos Alcaraz" Hard 3 A 2024-06-01

---

<img width="780" height="897" alt="image" src="https://github.com/user-attachments/assets/f45c0c5e-bea1-475e-832a-628dd4ddaaa0" />
