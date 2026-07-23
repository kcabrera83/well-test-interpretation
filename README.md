# Well Test Interpretation

ML-based well test interpretation and pressure transient analysis system.

## Overview

This project uses machine learning models to analyze well test pressure data,
identify flow regimes, and estimate reservoir parameters such as permeability
and skin factor.

### Models

- **Pressure Analyzer** (GradientBoosting) - Identifies flow regimes from
  pressure derivative data: wellbore storage, radial flow, boundary dominated.
- **Reservoir Estimator** (RandomForest) - Estimates permeability (md) and
  skin factor from pressure transient data.

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

## License

MIT

---

Elaborado por Ing. Kelvin Cabrera
