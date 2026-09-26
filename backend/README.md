# Backend

The backend is a Python/FastAPI application. Causal estimators, data loading,
targeting, and budget optimization will be added as separate service modules in
later iterations.

## Run locally

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

The API is available at <http://127.0.0.1:8000>. Open
<http://127.0.0.1:8000/docs> for the interactive OpenAPI documentation.

## Generate sample campaign data

From the repository root:

```powershell
$env:PYTHONPATH = "backend"
python -m app.data.synthetic_data --out data\sample_campaign_data.csv --n 5000 --seed 42
```
