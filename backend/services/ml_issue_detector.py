import pandas as pd
from typing import Optional, Dict

def detect_ml_issues(
    df: pd.DataFrame, 
    target_column: str, 
    train_metrics: Optional[Dict[str, float]] = None, 
    test_metrics: Optional[Dict[str, float]] = None, 
    feature_importance: Optional[list] = None
) -> dict:
    """
    Inspects dataset and model outputs to automatically detect common ML issues.
    """
    warnings = []
    detected_issues = []
    recommendations = []
    
    # 1. Check for insufficient data
    if len(df) < 500:
        detected_issues.append("insufficient_data")
        warnings.append(f"Very small dataset detected ({len(df)} rows).")
        recommendations.append("Collect more data to ensure model generalization and avoid severe overfitting.")
        
    # 2. Check for class imbalance (if classification task)
    if target_column in df.columns:
        if df[target_column].dtype in ['object', 'category', 'bool'] or df[target_column].nunique() < 20:
            counts = df[target_column].value_counts(normalize=True)
            if not counts.empty:
                minority_ratio = counts.min()
                if minority_ratio < 0.10:
                    detected_issues.append("severe_class_imbalance")
                    warnings.append(f"Severe class imbalance: minority class is only {minority_ratio*100:.1f}% of the target.")
                    recommendations.append("Apply SMOTE, adjust class weights, or use stratified sampling.")
                elif minority_ratio < 0.25:
                    detected_issues.append("moderate_class_imbalance")
                    warnings.append(f"Moderate class imbalance: minority class is {minority_ratio*100:.1f}%.")
                    recommendations.append("Monitor F1-Score instead of Accuracy. Consider resampling.")
                    
    # 3. Check for overfitting and suspicious performance
    if train_metrics and test_metrics:
        train_acc = train_metrics.get("accuracy", 0.0)
        test_acc = test_metrics.get("accuracy", 0.0)
        
        # Overfitting gap
        if train_acc - test_acc > 0.10:
            detected_issues.append("overfitting")
            warnings.append(f"Overfitting detected: Train Accuracy ({train_acc:.2f}) is much higher than Test Accuracy ({test_acc:.2f}).")
            recommendations.append("Apply regularization, reduce model complexity (e.g. tree depth), or increase training data size.")
            
        # Target Leakage / Too good to be true
        if test_acc >= 0.995:
            detected_issues.append("data_leakage_risk")
            warnings.append("Suspiciously high test accuracy (≥99.5%).")
            recommendations.append("Check for target leakage. Ensure features mathematically derived from the target are removed.")
            
    # 4. Check for weak or low-information features
    if feature_importance:
        # Expected format: [{"feature": "f1", "importance": 0.8}, ...]
        total_features = len(feature_importance)
        weak_features = [f["feature"] for f in feature_importance if f.get("importance", 1) < 0.01]
        
        if total_features > 0 and (len(weak_features) / total_features) > 0.5:
            detected_issues.append("many_weak_features")
            warnings.append(f"High percentage of weak features: {len(weak_features)} out of {total_features} contribute <1% to predictions.")
            recommendations.append("Perform feature selection or PCA to drop noisy columns and improve model speed/robustness.")
            
        # One feature dominating
        if total_features > 1:
            top_importance = max(f.get("importance", 0) for f in feature_importance)
            if top_importance > 0.85:
                detected_issues.append("single_feature_dominance")
                warnings.append(f"A single feature controls {top_importance*100:.1f}% of the model logic.")
                recommendations.append("Review this dominant feature carefully for potential target leakage.")
                
    if not warnings:
        warnings.append("No critical ML issues detected during automated inspection.")
        recommendations.append("Proceed with standard validation and A/B testing.")

    # Deduplicate while preserving order
    return {
        "warnings": list(dict.fromkeys(warnings)),
        "detected_issues": list(dict.fromkeys(detected_issues)),
        "recommendations": list(dict.fromkeys(recommendations))
    }
