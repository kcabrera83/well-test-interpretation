# API Documentation - Well Test Interpretation

## Base URL

```
http://localhost:5012
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "analyzer": true,
    "estimator": true
  }
}
```

---

### GET /api/models

Return information about loaded models.

**Response:**
```json
{
  "flow_regime_classifier": {
    "type": "GradientBoostingClassifier",
    "loaded": true,
    "flow_regimes": {
      "0": "wellbore_storage",
      "1": "radial_flow",
      "2": "boundary_dominated"
    }
  },
  "reservoir_estimator": {
    "type": "RandomForestRegressor (permeability + skin)",
    "loaded": true
  }
}
```

---

### POST /api/analyze

Analyze pressure data to identify flow regimes.

**Request:**
```json
{
  "time_hours": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
  "pressure_psi": [4500, 4400, 4350, 4200, 4100, 3900, 3800],
  "flow_rate_bbl_d": [500, 500, 500, 500, 500, 500, 500]
}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| time_hours | list[float] | Time measurements (hours) |
| pressure_psi | list[float] | Pressure readings (psi) |
| flow_rate_bbl_d | list[float] | Flow rate measurements (bbl/day) |

**Response:**
```json
{
  "predictions": ["wellbore_storage", "wellbore_storage", "radial_flow", "radial_flow", "radial_flow", "boundary_dominated", "boundary_dominated"],
  "probabilities": [
    [0.85, 0.10, 0.05],
    [0.70, 0.25, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.85, 0.10],
    [0.05, 0.80, 0.15],
    [0.05, 0.10, 0.85],
    [0.10, 0.10, 0.80]
  ],
  "n_samples": 7
}
```

**Flow Regimes:**
| Code | Regime | Description |
|------|--------|-------------|
| 0 | wellbore_storage | Early-time wellbore storage effect |
| 1 | radial_flow | Middle-time infinite acting radial flow |
| 2 | boundary_dominated | Late-time boundary dominated flow |

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | No JSON data | `{"error": "No JSON data provided"}` |
| 400 | Empty arrays | `{"error": "Empty data arrays"}` |
| 503 | Model not loaded | `{"error": "Flow regime model not loaded"}` |
| 500 | Processing error | `{"error": "<details>"}` |

---

### POST /api/estimate

Estimate permeability and skin factor from pressure transient data.

**Request:**
```json
{
  "time_hours": [0.1, 0.5, 1.0, 5.0, 10.0],
  "pressure_psi": [4500, 4400, 4350, 4200, 4100],
  "flow_rate_bbl_d": [500, 500, 500, 500, 500],
  "wellbore_radius_ft": [0.35, 0.35, 0.35, 0.35, 0.35],
  "reservoir_pressure_psi": [5000, 5000, 5000, 5000, 5000],
  "drainage_radius_ft": [1500, 1500, 1500, 1500, 1500],
  "formation_thickness_ft": [50, 50, 50, 50, 50]
}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| time_hours | list[float] | Time measurements (hours) |
| pressure_psi | list[float] | Pressure readings (psi) |
| flow_rate_bbl_d | list[float] | Flow rate (bbl/day) |
| wellbore_radius_ft | list[float] | Wellbore radius (ft) |
| reservoir_pressure_psi | list[float] | Initial reservoir pressure (psi) |
| drainage_radius_ft | list[float] | Drainage radius (ft) |
| formation_thickness_ft | list[float] | Formation thickness (ft) |

**Response:**
```json
{
  "permeability_md": [125.3, 130.1, 128.5, 127.8, 129.0],
  "skin_factor": [1.25, 1.30, 1.28, 1.22, 1.31],
  "n_samples": 5
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | No JSON data | `{"error": "No JSON data provided"}` |
| 400 | Empty arrays | `{"error": "Empty data arrays"}` |
| 503 | Model not loaded | `{"error": "Reservoir estimator model not loaded"}` |
| 500 | Processing error | `{"error": "<details>"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

---

## Error Codes

- **200**: Success
- **400**: Bad request (missing or invalid parameters)
- **503**: Model not loaded (run train.py first)
- **500**: Internal server error

---

*Elaborado por Ing. Kelvin Cabrera*
