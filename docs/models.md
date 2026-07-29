# Models

## Training Pipeline

The training pipeline follows these steps:

1. **Data Generation:** `data_generator.py` creates synthetic data with domain-specific physical correlations
2. **Preprocessing:** Feature scaling via `StandardScaler`
3. **Training:** Random Forest or Isolation Forest via scikit-learn
4. **Evaluation:** R2 score, accuracy, classification report
5. **Logging:** MLflow tracks parameters, metrics, and artifacts
6. **Serialization:** Model + scaler + metadata saved as `.pkl` via joblib

## Model Files

Models are stored in `outputs/models/*.pkl` as dictionaries with:

| Key | Description |
|-----|-------------|
| `model` | Trained scikit-learn estimator |
| `scaler` | Fitted StandardScaler |
| `feature_names` | List of input feature names |
| `target_name` | Target variable name |
| `metrics` | Evaluation metrics from training |

## Retraining

To retrain:

```bash
python train.py
```

This will update the model files in `outputs/models/` and log results to MLflow.
