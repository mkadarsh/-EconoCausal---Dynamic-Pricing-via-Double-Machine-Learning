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

## Causal DAG

The DoWhy graph in
[backend/app/causal/dag.py](</D:/Github Projects/-EconoCausal-- Dynamic Pricing via Double Machine Learning/backend/app/causal/dag.py>)
represents each binary treatment contrast produced by preprocessing:

- `T10`: a `$10` discount versus `$0`
- `T20`: a `$20` discount versus `$0`
- `purchase`: the primary causal outcome
- `revenue`: a downstream outcome of purchase
- observed confounders: `age`, `income`, `previous_purchases`,
  `website_visits`, `customer_tenure`, `previous_spend`, and
  `engagement_score`

The observed customer attributes point to both treatment assignment and
purchase. This encodes the adjustment set used for identification without
inventing unsupported relationships. The DAG assumes these features are
pre-treatment, sufficiently capture common causes, and that both treatment
levels have overlap. Because the campaign is observational, unmeasured
confounding, treatment-selection bias, measurement error, and the synthetic
data-generating assumptions remain limitations; DoWhy identification does not
prove those assumptions are true.

## Propensity scores and matching

[backend/app/causal/propensity.py](</D:/Github Projects/-EconoCausal-- Dynamic Pricing via Double Machine Learning/backend/app/causal/propensity.py>)
fits a scikit-learn logistic regression for each binary contrast (`$10` vs
`$0` and `$20` vs `$0`). It returns customer-level propensity scores,
treated/control summaries, common-support statistics, plotting-ready score
distributions, and one-to-one nearest-neighbor matches.

Matching balances the observed covariates represented in the propensity model.
It does not guarantee elimination of unobserved confounding or establish that
the observational assumptions are correct.

## Double Machine Learning

[backend/app/causal/dml.py](</D:/Github Projects/-EconoCausal-- Dynamic Pricing via Double Machine Learning/backend/app/causal/dml.py>)
fits separate EconML `LinearDML` models for `$10` versus `$0` and `$20` versus
`$0`. Random-forest outcome nuisance models and logistic treatment nuisance
models are cross-fitted by EconML. The result includes customer-level `ITE_10`
or `ITE_20`, their average treatment effect, effect-distribution summaries,
optional confidence intervals, and model metadata. `train_dml_models` combines
both outputs into a customer-mappable table with `customer_id`, `ITE_10`, and
`ITE_20`.

DML relies on consistency, no interference between customers, positivity
(overlap), and conditional exchangeability given the observed pre-treatment
features. Cross-fitting reduces overfitting bias in nuisance estimation but
does not validate these causal assumptions or remove unobserved confounding.

### Counterfactual purchase probabilities

`train_dml_models` also returns one row per customer with:

- `baseline_purchase_probability`: a RandomForestClassifier estimate of
  purchase under the observed covariates and `$0`, trained on the control arm.
- `purchase_probability_10` and `purchase_probability_20`: an MVP
  counterfactual approximation defined as
  `clip(baseline_purchase_probability + ITE, 0, 1)`.
- `ITE_10` and `ITE_20`: EconML causal effect estimates.

This construction is intentionally separate from ordinary predictive
probabilities: the increment comes from DML, not from a raw treated/control
mean or correlation. Clipping keeps the approximation in the probability
domain, but it does not make an ITE a guaranteed individual outcome. The
probabilities describe model-based expected potential outcomes under the
assumptions above, not certainty about what a particular customer would do.

## Customer response segmentation

[backend/app/causal/uplift.py](</D:/Github Projects/-EconoCausal-- Dynamic Pricing via Double Machine Learning/backend/app/causal/uplift.py>)
classifies the counterfactual output into descriptive response groups:
`Persuadable`, `Sure Buyer`, `Unlikely Buyer`, `Potential Negative Response`,
or `Unclassified Response` when configured rules do not provide enough evidence
for one of the primary groups. Thresholds are defined in
[backend/app/causal/uplift_config.py](</D:/Github Projects/-EconoCausal-- Dynamic Pricing via Double Machine Learning/backend/app/causal/uplift_config.py>)
and can be supplied through `SegmentationConfig`.

Classification is rule-based on estimated probabilities and ITEs. Negative
effects take precedence, followed by sure buyers, persuadable customers, and
unlikely buyers, so overlapping boundary cases are deterministic. These
segments describe model estimates, not guaranteed individual behavior.
