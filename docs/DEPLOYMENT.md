# Deployment Guide - Well Test Interpretation

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5012

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t well-test-interpretation .
docker run -p 5012:5012 well-test-interpretation
```

## Docker Compose

```yaml
version: '3.8'
services:
  well-test-interpretation:
    build: .
    ports:
      - "5012:5012"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5012 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5012 app:app
  ```
- Set `debug=False` in `app.py` (already set)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart
- Models serialized with joblib for efficient loading

## Training Pipeline

1. `python train.py` generates synthetic pressure transient data
2. Feature engineering (derivatives, normalization) applied
3. Models trained and evaluated on test sets
4. Artifacts saved to `outputs/models/`:
   - `pressure_analyzer.joblib` - Flow regime classifier
   - `reservoir_estimator.joblib` - Permeability/skin estimator

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests

---

*Elaborado por Ing. Kelvin Cabrera*
