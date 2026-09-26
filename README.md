# EconoCausal

EconoCausal is an MVP causal AI platform for dynamic promotional pricing. It
will estimate whether a `$10` or `$20` discount causes an incremental purchase
relative to no discount, identify persuadable customers, and allocate a fixed
marketing budget.

## Architecture

The project uses a deliberately small two-part architecture:

- `backend/`: Python and FastAPI REST API. The future causal-analysis service
  will use Pandas, NumPy, scikit-learn, EconML, DoWhy, and SciPy.
- `frontend/`: React, TypeScript, and Vite single-page application. Plotly is
  included for future treatment-effect and targeting visualizations.

The initial causal design uses two binary comparisons because standard DML
estimators commonly use binary treatment:

1. `$10` discount versus `$0`
2. `$20` discount versus `$0`

The optimization layer will compare expected incremental revenue for `$0`, `$10`,
and `$20` subject to a marketing budget.

## Current MVP status

The project currently contains the runnable application shell, a FastAPI health
endpoint, and a basic discount-action frontend. Causal estimators, historical
data ingestion, targeting, optimization, REST analysis endpoints, charts, and
DoWhy refutation tests are intentionally not implemented yet.

## Run the backend

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

The backend runs at <http://127.0.0.1:8000>. API documentation is at
<http://127.0.0.1:8000/docs>, and the health check is at
<http://127.0.0.1:8000/api/health>.

## Run the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs at <http://localhost:5173>.

## Expected historical data

The planned input data will have one customer-level observation per row with:

`customer_id`, `age`, `income`, `previous_purchases`, `website_visits`,
`customer_tenure`, `previous_spend`, `engagement_score`, `treatment`,
`discount_amount`, `purchase`, and `revenue`.

Treatment actions are `$0`, `$10`, and `$20`, represented by the discount amount.

## Generate and validate sample data

The committed sample campaign is generated with:

```powershell
$env:PYTHONPATH = "backend"
python -m app.data.synthetic_data --out data\sample_campaign_data.csv --n 5000 --seed 42
```

To validate a CSV from Python:

```python
from app.data.data_loader import load_csv

campaign = load_csv("data/sample_campaign_data.csv")
```
