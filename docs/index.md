# Well Test Interpretation

Pressure transient analysis with flow regime identification and permeability estimation.

## Overview

This machine learning system is part of the **Oil & Gas ML Portfolio** by [Kelvin J Cabrera](https://github.com/kcabrera83).

### Architecture

```
Synthetic Data Generator -> Scikit-Learn Training -> Flask API -> Streamlit Dashboard
```

### Key Features

- **Data Generation:** Domain-aware synthetic data with physical correlations
- **Model Training:** Scikit-learn pipelines (Random Forest, Isolation Forest) with MLflow tracking
- **API:** Flask/FastAPI with API key authentication
- **Dashboard:** Streamlit interactive web interface
- **Testing:** pytest with model evaluation metrics (R2, accuracy, F1)

### Quick Start

```bash
pip install -r requirements.txt
python train.py
python app.py
streamlit run streamlit_app.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info and available models |
| POST | `/predict/{model}` | Run inference (requires API key) |

!!! info "Authentication"
    All API endpoints require an `X-API-Key` header. Set `API_KEY` in your environment.
