# EconoCausal

EconoCausal is a working MVP for causal promotional pricing. It helps a
marketing team estimate whether a discount is likely to cause an incremental
purchase, rather than only predicting which customers are likely to buy.

The MVP compares three actions:

- `$0` — no discount
- `$10` — ten-dollar discount
- `$20` — twenty-dollar discount

It estimates the `$10 vs $0` and `$20 vs $0` contrasts independently, then
uses the estimates to segment customers and allocate a fixed marketing budget.
The application is intentionally small, local, and in-memory: it is designed
for experimentation and learning, not production deployment.

## Overview

Traditional campaign targeting often uses a predictive model to answer:

> Who is likely to purchase?

That answer is useful, but it does not answer:

> Who is more likely to purchase because we offered a discount?

EconoCausal addresses the second question with observational causal modeling.
Customer characteristics influence both promotion assignment and purchase in
the synthetic campaign data, creating a realistic confounding problem. The
backend adjusts for measured pre-treatment characteristics and produces
customer-level treatment-effect estimates, counterfactual purchase
probabilities, uplift curves, causal robustness checks, and budget-constrained
recommendations.

## Predictive AI vs Causal AI

Predictive AI estimates an outcome such as:

```text
P(purchase = 1 | customer characteristics)
```

This is a forecast of what may happen under the observed or typical campaign
conditions. A high-probability customer may purchase without receiving any
promotion.

Causal AI estimates a contrast between potential outcomes:

```text
ITE_10 = P(purchase | do(discount = $10)) - P(purchase | do(discount = $0))
ITE_20 = P(purchase | do(discount = $20)) - P(purchase | do(discount = $0))
```

An Individual Treatment Effect (ITE) estimates the incremental response for a
customer. It can identify:

- **Persuadable customers** whose purchase probability increases meaningfully
  with a discount.
- **Sure buyers** who have a high baseline probability and little estimated
  incremental response.
- **Unlikely buyers** who have a low baseline probability and little response.
- **Potential negative response** customers whose estimated effect is negative.

An ITE is an estimate, not a guarantee of an individual customer's behavior.

## Key Features

- **Causal DAG** describing observed confounders, treatment, purchase, and
  revenue relationships.
- **Propensity score analysis** with logistic treatment-assignment models,
  common-support statistics, and nearest-neighbor matching.
- **Double Machine Learning** using EconML `LinearDML` for two binary
  treatment contrasts.
- **Individual Treatment Effects** for `$10 vs $0` and `$20 vs $0`.
- **Counterfactual purchase probabilities** for no discount, `$10`, and `$20`.
- **Uplift analysis** ranked by estimated customer response.
- **Qini curves** and cumulative incremental-gain curves with random-targeting
  baselines.
- **DoWhy refutation tests**, including placebo, random common cause, and
  subset checks where supported by the installed version.
- **Budget optimization** using an exact discrete mixed-integer optimization
  formulation.
- **Personalized discount allocation** with one action per customer and a
  total discount-cost constraint.
- **React/Plotly dashboard** for data upload, causal analysis, customer
  insights, optimization, and audit results.
- **Data drift architecture/limitations:** the code has a clean validation and
  preprocessing boundary where drift monitoring can be added, but automated
  production drift detection, alerting, and retraining are not implemented in
  this MVP.

## Architecture

```text
                    Local browser
                         |
                         | REST/JSON, multipart CSV upload
                         v
             +---------------------------+
             | React + TypeScript + Vite |
             | Plotly dashboard          |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             | FastAPI REST API          |
             | In-memory application     |
             | state and workflow routes |
             +-------------+-------------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
 +----------------------+        +----------------------+
 | Data and preprocessing|        | Causal analysis      |
 | CSV validation        |        | DAG / propensity     |
 | synthetic generator   |        | EconML DML / ITE     |
 +----------+-----------+        | uplift / DoWhy audit |
            |                    +----------+-----------+
            |                               |
            +---------------+---------------+
                            v
                 +--------------------------+
                 | Discrete optimization    |
                 | SciPy MILP               |
                 | $0 / $10 / $20 allocation|
                 +--------------------------+
```

The backend keeps the current dataset and analysis results in memory. Uploading
or generating a new dataset clears downstream model, audit, and optimization
results.

## Technology Stack

### Frontend

- React 18
- TypeScript
- Vite
- `react-plotly.js`
- `plotly.js-dist-min`

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic Settings
- Pandas
- NumPy

### Causal ML

- scikit-learn
- EconML
- DoWhy

### Optimization

- SciPy `milp`
- SciPy sparse constraint matrices

### Visualization

- Plotly charts in the React dashboard
- Plotly-ready uplift, Qini, propensity, and allocation data from the API

## Dataset

The synthetic generator writes `data/sample_campaign_data.csv`. Uploaded CSV
files must contain the following columns:

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier retained for result mapping and excluded from ML features. |
| `age` | Customer age. |
| `income` | Customer income estimate. |
| `previous_purchases` | Historical number of purchases. |
| `website_visits` | Historical website visit count. |
| `customer_tenure` | Customer tenure in the synthetic campaign period. |
| `previous_spend` | Historical customer spend. |
| `engagement_score` | Synthetic engagement measure derived from customer activity and noise. |
| `treatment` | Observed discount action: `0`, `10`, or `20`. |
| `discount_amount` | Discount amount; must match `treatment`. |
| `purchase` | Binary purchase outcome: `0` or `1`. |
| `revenue` | Observed revenue for the campaign observation. |

The generator intentionally makes engagement, purchase history, income, and
other characteristics affect both promotion assignment and purchase. This
creates confounding rather than a completely randomized treatment.

The causal pipeline creates two filtered binary datasets:

1. `$10 vs $0`, excluding `$20` observations.
2. `$20 vs $0`, excluding `$10` observations.

## Causal Assumptions

The estimates depend on assumptions that cannot be proven by the application:

- **Observed confounders:** age, income, previous purchases, website visits,
  customer tenure, previous spend, and engagement score are treated as
  pre-treatment variables that may influence both discount assignment and
  purchase.
- **Treatment:** each binary contrast represents receiving a specific discount
  versus receiving no discount.
- **Outcome:** the primary causal outcome is binary `purchase`.
- **Consistency:** the recorded discount corresponds to a well-defined
  intervention.
- **Positivity/overlap:** comparable customers must have a non-zero chance of
  receiving each treatment level.
- **No interference:** one customer's discount does not affect another
  customer's purchase outcome.
- **Conditional exchangeability:** after adjusting for the observed covariates,
  treatment assignment is assumed independent of potential outcomes.

The data is observational and the synthetic data-generating process is only an
example. Unobserved confounding, selection bias, measurement error,
misspecified models, poor overlap, and treatment interference can invalidate
the estimates. DoWhy refutation tests are robustness checks, not proof of
causal identification.

## Installation

### Backend setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell execution policy prevents activation, run commands with the
environment's interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### Frontend setup

In a second terminal:

```powershell
Set-Location frontend
npm install
```

## Running the Project

Start the backend from the repository root:

```powershell
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

The backend is available at <http://127.0.0.1:8000>.

Start the frontend in a second terminal:

```powershell
Set-Location frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

The frontend is available at <http://127.0.0.1:5173>.

The frontend uses `http://localhost:8000/api` by default. To point it at
another backend:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000/api"
npm run dev
```

FastAPI's interactive documentation is available at
<http://127.0.0.1:8000/docs>.

## Demo Workflow

1. Open the dashboard at <http://127.0.0.1:5173>.
2. Open **Data Upload**.
3. Click **Generate Demo Dataset**, or upload a CSV matching the required
   schema.
4. Review row count, columns, missing values, treatment distribution,
   purchase rate, and validation errors.
5. Open **Causal Analysis** and run the causal models.
6. Review ATE, ITE distributions, counterfactual probabilities, segments, and
   customer-level estimates.
7. Review uplift and Qini curves, including the random-targeting baseline.
8. Open **Causal Audit** and inspect the returned DoWhy refutation results and
   limitations.
9. Open **Budget Optimization**, set the marketing budget and expected order
   value, and optionally set a maximum number of targeted customers.
10. Run optimization and review the recommended `$0`, `$10`, or `$20` action
    for each customer.

## API Documentation

All application endpoints are prefixed with `/api`.

### Health

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check that the backend is available. |

### Data

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/data/generate` | Generate and load a synthetic campaign dataset. |
| `POST` | `/api/data/upload` | Upload and validate a campaign CSV as multipart form data. |
| `GET` | `/api/data/summary` | Return row count, columns, treatment counts, and purchase rate. |

### Causal analysis

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/causal/train` | Preprocess data and train the `$10 vs $0` and `$20 vs $0` DML models. |
| `GET` | `/api/causal/summary` | Return model status, ATEs, counts, intervals, and metadata. |
| `GET` | `/api/causal/results` | Return customer-level probabilities and ITEs. |
| `GET` | `/api/causal/customer/{customer_id}` | Return one customer's profile, estimates, segment, and recommendation. |
| `GET` | `/api/causal/segments` | Return customer response segments and segment statistics. |
| `GET` | `/api/causal/uplift` | Return uplift, Qini, cumulative-gain, and random-baseline chart data. |
| `GET` | `/api/causal/propensity` | Return propensity scores, overlap, and matching statistics. |
| `GET` | `/api/causal/audit` | Run and return DoWhy refutation results for both contrasts. |

### Optimization

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/optimization/run` | Allocate discrete discount actions under a budget. |
| `GET` | `/api/optimization/results` | Retrieve the latest allocation and campaign summary. |

Prerequisite failures return an appropriate `4xx` response. Invalid data and
model inputs return `422`, missing workflow prerequisites generally return
`404` or `409`, and unavailable optional causal dependencies return `503`.

## Testing

### Backend tests

From the repository root:

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend -q
```

The backend tests cover data generation and validation, preprocessing,
propensity analysis, DML, DAG construction, uplift and Qini evaluation,
refutation behavior, optimization constraints, and API prerequisites.

To check Python syntax and imports:

```powershell
python -m compileall -q backend\app
```

### Frontend type checking and production build

```powershell
Set-Location frontend
npm run build
```

The build runs TypeScript project checking followed by the Vite production
build. No separate frontend lint script is currently configured.

## Limitations

- The MVP uses synthetic data by default and does not represent a real
  campaign.
- Uploaded data is observational; causal estimates depend on the assumptions
  listed above.
- The two discount levels are modeled as separate binary contrasts rather than
  with a single native multi-treatment estimator.
- Counterfactual purchase probabilities are an MVP construction based on a
  control-arm baseline model plus the estimated ITE, clipped to `[0, 1]`.
- ITEs and recommendations are estimates, not guarantees of individual
  behavior or revenue.
- DoWhy refutation tests can be unavailable or inconclusive depending on data
  and library compatibility. A passing check does not prove identification.
- Application state is in memory and is lost when the backend restarts.
- There is no authentication, authorization, database, job queue, or
  multi-user isolation.
- Upload validation has a 50 MB limit but is not a full data-governance or
  schema-management system.
- Automated data drift detection, alerting, feature monitoring, and
  retraining are not implemented.
- Plotly is bundled into the frontend build, producing a relatively large
  JavaScript bundle for this MVP.

## Future Improvements

- Integrate real campaign data with documented governance and privacy controls.
- Use a better native multi-treatment causal estimator or generalized
  propensity framework.
- Add a production database and durable analysis-job storage.
- Add authentication, authorization, tenant isolation, and audit logging.
- Add a model registry with versioned datasets, models, assumptions, and
  results.
- Add automated retraining and model-performance monitoring.
- Add advanced drift monitoring for features, treatment assignment, outcomes,
  overlap, and ITE distributions.
- Add asynchronous jobs for large datasets and long-running refutation tests.
- Add richer business objectives, inventory constraints, and revenue outcomes.
- Add CI, frontend linting, end-to-end browser tests, and observability.
- Deploy the system to a secure cloud environment.
