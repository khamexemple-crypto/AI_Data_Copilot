import numpy as np

def explain_model(trained_model, feature_names: list[str], X_sample=None) -> dict:
    """
    Extracts feature importances from a trained model and generates a natural language summary.
    Gracefully handles models that do not natively support feature_importances_.
    """
    importances = None
    
    # Check for native tree-based feature importances
    if hasattr(trained_model, "feature_importances_"):
        importances = trained_model.feature_importances_
    # Fallback for linear models
    elif hasattr(trained_model, "coef_"):
        # For multi-class classification, coef_ can be a 2D array. We take the mean absolute value.
        coefs = np.atleast_2d(trained_model.coef_)
        importances = np.mean(np.abs(coefs), axis=0)
    
    if importances is None:
        return {
            "feature_importance": [],
            "natural_language_summary": [
                "The selected model does not expose feature importances natively.",
                "Consider using a tree-based model (like RandomForest or XGBoost) for built-in explainability."
            ]
        }
    
    # Normalize importances so they sum to 1 (if not all zeros)
    total_importance = np.sum(importances)
    if total_importance > 0:
        importances = importances / total_importance
        
    # Pair features with their importances and sort descending
    feat_imp_pairs = [
        {"feature": str(f), "importance": float(imp)}
        for f, imp in zip(feature_names, importances)
    ]
    feat_imp_pairs.sort(key=lambda x: x["importance"], reverse=True)
    
    # Generate Natural Language Summary
    summary = []
    top_features = feat_imp_pairs[:3] # Focus on top 3
    if len(top_features) > 0:
        top_names = [f["feature"] for f in top_features]
        top_pcts = [f"{f['importance']*100:.1f}%" for f in top_features]
        
        summary.append(f"The model's predictions are primarily driven by: **{', '.join(top_names)}**.")
        summary.append(f"Specifically, '{top_names[0]}' accounts for {top_pcts[0]} of the decision weight.")
        
        if len(top_names) > 1:
             summary.append(f"Other significant factors include '{top_names[1]}' ({top_pcts[1]}).")
             
        cumulative_importance = sum(f["importance"] for f in top_features)
        if cumulative_importance > 0.8:
            summary.append("These top features heavily dominate the model's logic, suggesting a highly focused decision boundary.")
    else:
         summary.append("All features appear to have zero importance. This usually indicates an untrained or degenerate model.")

    return {
        "feature_importance": feat_imp_pairs,
        "natural_language_summary": summary
    }
