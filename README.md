# Well Test Interpretation

ML-based well test interpretation and pressure transient analysis system using advanced curve fitting and uncertainty quantification.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Curve Fitting | **scipy.optimize** - non-linear least squares |
| Curve Fitting | **lmfit** - parameter estimation and modeling |
| Uncertainty | **uncertainties** - error propagation |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- scipy.optimize - Non-linear curve fitting for pressure transient analysis
- lmfit - Advanced parameter estimation with constraints
- uncertainties - Error propagation and uncertainty quantification
- FastAPI - Modern async web framework

## Overview

This project uses machine learning models and curve fitting to analyze well test pressure data,
identify flow regimes, and estimate reservoir parameters such as permeability
and skin factor.

### Models

- **Pressure Analyzer** (scipy.optimize + lmfit) - Identifies flow regimes from
  pressure derivative data: wellbore storage, radial flow, boundary dominated.
- **Reservoir Estimator** (lmfit + uncertainties) - Estimates permeability (md) and
  skin factor from pressure transient data with uncertainty bounds.

## Setup

```bash
pip install -r requirements.txt
python train.py
```

## Running the Application

```bash
python app.py
```

The application runs on port 5012.

## API Endpoints

| Endpoint         | Method | Description                          |
|------------------|--------|--------------------------------------|
| `/`              | GET    | Web dashboard                        |
| `/api/analyze`   | POST   | Identify flow regimes                |
| `/api/estimate`  | POST   | Estimate permeability and skin       |
| `/api/models`    | GET    | Model information                    |
| `/api/health`    | GET    | Health check                         |

## Testing

```bash
python test_api.py
```

## Project Structure

```
well-test-interpretation/
  well_test_interpretation/
    __init__.py
    data_generator.py
    models/
      __init__.py
      pressure_analyzer.py
      reservoir_estimator.py
    utils/
      __init__.py
      preprocessor.py
  templates/
    index.html
  outputs/models/
  train.py
  app.py
  test_api.py
  requirements.txt
  setup.py
```

---

Elaborado por Ing. Kelvin Cabrera
