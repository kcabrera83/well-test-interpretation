# Architecture - Well Test Interpretation

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (ML Models)       |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`generate_pressure_transient_data`)
- **Samples**: 5,000 pressure transient records
- **Parameters**: Time, pressure, flow rate, wellbore/reservoir geometry
- **Derived features**: log_time, dp_dlogt, normalized_pressure, flow_normalized_dp, pressure_squared

### Feature Engineering

The preprocessor computes derived features from raw pressure data:

| Feature | Description |
|---------|-------------|
| log_time | Logarithm of time |
| pressure_psi | Raw pressure reading |
| dp_dlogt | Pressure derivative w.r.t. log time |
| flow_rate_bbl_d | Flow rate |
| normalized_pressure | Pressure normalized to initial conditions |
| flow_normalized_dp | Pressure derivative normalized by flow rate |
| pressure_squared | Squared pressure (for gas wells) |

### Model Layer

#### Pressure Analyzer (Flow Regime Classifier)
- **Algorithm**: GradientBoostingClassifier
- **Task**: Multi-class classification (3 flow regimes)
- **Input**: 9 features (raw + derived)
- **Output**: Flow regime label + probability distribution
- **Serialization**: joblib (`.joblib`)

#### Reservoir Estimator
- **Algorithm**: RandomForestRegressor (multi-output)
- **Task**: Predict permeability (md) and skin factor
- **Input**: 7 features (time, pressure, flow rate, wellbore geometry)
- **Output**: permeability_md, skin_factor arrays
- **Serialization**: joblib (`.joblib`)

### Preprocessing Pipeline

1. Raw pressure data received as arrays
2. Feature engineering: compute derivatives, normalized values
3. StandardScaler applied to all features
4. Train/test split with stratification (classification) or random (regression)

### API Layer

- **Framework**: Flask
- **Port**: 5012
- **Format**: JSON request/response
- **Endpoints**: 5 (analyze, estimate, models, health, docs)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Charts**: Plotly.js for time series visualization
- **Theme**: Dark theme UI

## Data Flow

### Flow Regime Analysis Flow
1. User provides time, pressure, flow rate arrays
2. Feature engineering computes derived features
3. GradientBoostingClassifier predicts regime per time step
4. Response includes predictions + probability arrays

### Reservoir Estimation Flow
1. User provides pressure data + well/reservoir parameters
2. Features extracted from input arrays
3. RandomForestRegressor predicts permeability + skin per time step
4. Response includes arrays of estimates

## Project Structure

```
well-test-interpretation/
├── well_test_interpretation/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic pressure transient data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pressure_analyzer.py       # GradientBoosting classifier
│   │   └── reservoir_estimator.py     # RandomForest regressor (multi-output)
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py            # Feature engineering + scaling
├── templates/
│   └── index.html                     # Dashboard UI
├── outputs/models/                    # Saved model artifacts
├── train.py                           # Training pipeline
├── app.py                             # Flask API server
├── test_api.py                        # API test suite
├── requirements.txt
└── setup.py
```

## Model Evaluation

### Flow Regime Classifier
- Train accuracy: ~0.90+
- Test accuracy: ~0.85+
- Feature importances available

### Reservoir Estimator
- Permeability R2: ~0.90+
- Skin factor R2: ~0.85+
- Feature importances available

---

*Elaborado por Ing. Kelvin Cabrera*
