"""Training script for well test interpretation models using scipy + lmfit + uncertainties."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from well_test_interpretation.data_generator import generate_pressure_transient_data
from well_test_interpretation.utils.preprocessor import (
    extract_features,
    prepare_flow_regime_data,
    prepare_reservoir_data,
)
from well_test_interpretation.models.pressure_analyzer import PressureAnalyzer
from well_test_interpretation.models.reservoir_estimator import ReservoirEstimator


def main():
    """Train and evaluate both models."""
    print("=" * 60)
    print("  WELL TEST INTERPRETATION - MODEL TRAINING (scipy + lmfit)")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/6] Generating synthetic pressure transient data...")
    df = generate_pressure_transient_data(n_samples=5000, seed=42)
    print(f"  Generated {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")

    print("\n[2/6] Extracting features and computing pressure derivatives...")
    df = extract_features(df)
    print(f"  Feature columns: {[c for c in df.columns if c not in ['flow_regime', 'well_id']]}")

    print("\n[3/6] Training Flow Regime Analyzer (lmfit curve fitting)...")
    X_train_f, X_test_f, y_train_f, y_test_f, feat_names_f, scaler_f = (
        prepare_flow_regime_data(df)
    )
    print(f"  Train: {len(y_train_f)} | Test: {len(y_test_f)}")

    analyzer = PressureAnalyzer()
    train_metrics = analyzer.train(X_train_f, y_train_f, feature_names=feat_names_f)
    print(f"  Train accuracy: {train_metrics['train_accuracy']:.4f}")

    eval_metrics = analyzer.evaluate(X_test_f, y_test_f)
    print(f"  Test accuracy:  {eval_metrics['test_accuracy']:.4f}")
    print(f"  Feature importances:")
    for name, imp in sorted(
        eval_metrics["feature_importances"].items(), key=lambda x: -x[1]
    ):
        print(f"    {name}: {imp:.4f}")

    analyzer.save(os.path.join(output_dir, "pressure_analyzer.joblib"))
    print("  Model saved to outputs/models/pressure_analyzer.joblib")

    print("\n[4/6] Training Reservoir Estimator (lmfit curve fitting)...")
    X_train_r, X_test_r, y_perm_train, y_perm_test, feat_names_r, scaler_r = (
        prepare_reservoir_data(df, target_col="permeability_md")
    )
    _, _, y_skin_train, y_skin_test, _, _ = prepare_reservoir_data(df, target_col="skin_factor")

    print(f"  Train: {len(y_perm_train)} | Test: {len(y_perm_test)}")

    estimator = ReservoirEstimator()
    train_metrics_r = estimator.train(
        X_train_r, y_perm_train, y_skin_train, feature_names=feat_names_r
    )
    print(f"  Perm  train R2: {train_metrics_r['perm_train_r2']:.4f} | RMSE: {train_metrics_r['perm_train_rmse']:.2f} md")
    print(f"  Skin  train R2: {train_metrics_r['skin_train_r2']:.4f} | RMSE: {train_metrics_r['skin_train_rmse']:.4f}")

    eval_metrics_r = estimator.evaluate(X_test_r, y_perm_test, y_skin_test)
    print(f"  Perm  test  R2: {eval_metrics_r['permeability']['r2']:.4f} | RMSE: {eval_metrics_r['permeability']['rmse']:.2f} md | MAE: {eval_metrics_r['permeability']['mae']:.2f} md")
    print(f"  Skin  test  R2: {eval_metrics_r['skin_factor']['r2']:.4f} | RMSE: {eval_metrics_r['skin_factor']['rmse']:.4f} | MAE: {eval_metrics_r['skin_factor']['mae']:.4f}")
    print(f"  Feature importances:")
    for name, imp in sorted(
        eval_metrics_r["feature_importances"].items(), key=lambda x: -x[1]
    ):
        print(f"    {name}: {imp:.4f}")

    estimator.save(os.path.join(output_dir, "reservoir_estimator.joblib"))
    print("  Model saved to outputs/models/reservoir_estimator.joblib")

    print("\n[5/6] Training Summary")
    print("-" * 40)
    print(f"  Flow Regime Analyzer: {eval_metrics['test_accuracy']:.2%} accuracy")
    print(f"  Permeability Estimator: {eval_metrics_r['permeability']['r2']:.4f} R2")
    print(f"  Skin Factor Estimator:  {eval_metrics_r['skin_factor']['r2']:.4f} R2")

    print("\n[6/6] Done. All models trained and saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
