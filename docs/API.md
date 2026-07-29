# API Reference

## Authentication

All API requests require an API key sent via the `X-API-Key` header.

```python
import requests

headers = {"X-API-Key": "your-api-key"}
response = requests.post("http://localhost:5000/predict/model", json={...}, headers=headers)
```

## Endpoints

### GET `/`

Returns service information and available models.

**Response:**

```json
{"service": "Well Test Interpretation", "models": ["model1", "model2"]}
```

### POST `/predict/{model_name}`

Run inference on a trained model.

**Request Body:**

```json
{"features": {"feature_1": 0.5, "feature_2": 1.2}}
```

**Response:**

```json
{"prediction": 0.85, "model": "model_name"}
```

## Error Codes

| Code | Description |
|------|-------------|
| 401 | Invalid or missing API key |
| 404 | Model not found |
| 422 | Invalid input format |
