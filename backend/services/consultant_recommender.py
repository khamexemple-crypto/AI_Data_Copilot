def deduplicate_list(seq: list) -> list:
    """Removes duplicates while preserving order."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def generate_consultant_recommendations(
    analysis_results: dict,
    ml_results: dict | None = None,
    data_quality_results: dict | None = None
) -> dict:
    """
    Proactively generates grounded consultant recommendations based on analysis results.
    """
    top_insights = []
    recommendations = []
    next_steps = []
    risk_flags = []

    # 1. Process Data Quality Rules
    if data_quality_results:
        missing = data_quality_results.get("missing_values", {})
        if isinstance(missing, dict):
            missing_cols = [col for col, count in missing.items() if count > 0]
            if missing_cols:
                risk_flags.append(f"Missing data detected in {len(missing_cols)} column(s).")
                recommendations.append(f"Impute missing values for critical columns ({', '.join(missing_cols[:3])}) to avoid model bias.")
                next_steps.append("Investigate data collection pipeline to identify the root cause of missing values.")
        
        dupes = data_quality_results.get("duplicates", 0)
        if dupes > 0:
            recommendations.append(f"Remove {dupes} duplicate records to prevent data leakage and skewed distributions.")

    # 2. Process Analysis Insights
    if analysis_results:
        # Promote top insights identified by the Analyst Agent
        insights = analysis_results.get("insights", [])
        if insights:
            top_insights.extend(insights[:3])
            
        anomalies = analysis_results.get("anomalies", [])
        if anomalies:
            risk_flags.append(f"Detected anomalies/outliers: {anomalies[0]}")
            recommendations.append("Isolate and investigate anomalous records. Consider using robust scaling methods.")

    # 3. Process ML Results & Risks
    if ml_results:
        best_model = ml_results.get("best_model")
        metrics = ml_results.get("metrics", {})
        
        if best_model:
            top_insights.append(f"The '{best_model}' algorithm emerged as the best performing model.")
            next_steps.append(f"Prepare the '{best_model}' pipeline for A/B testing or shadow deployment.")
        
        # Heuristics for common ML issues
        acc = metrics.get("accuracy", 0.0)
        train_acc = metrics.get("train_accuracy", acc)
        f1 = metrics.get("f1_score", acc)

        # Overfitting check
        if train_acc - acc > 0.1:
            risk_flags.append("High risk of overfitting: large gap between training and validation accuracy.")
            recommendations.append("Apply regularization or reduce model complexity (e.g., limit tree depth, add dropout layers).")
        
        # Class imbalance check
        if acc - f1 > 0.15:
            risk_flags.append("Potential class imbalance detected (Accuracy is significantly higher than F1-Score).")
            recommendations.append("Apply resampling techniques (SMOTE, undersampling) or adjust classification thresholds.")

        # Data Leakage check
        if acc > 0.99:
            risk_flags.append("Suspiciously high accuracy (>99%). High risk of data leakage.")
            recommendations.append("Review feature set for target leakage (features mathematically derived from the target variable).")

    # 4. Fallbacks (Ensure outputs are never empty)
    if not next_steps:
        next_steps = ["Review findings with domain experts.", "Define specific business KPIs for deployment tracking."]
    if not top_insights:
        top_insights.append("Exploratory analysis completed without identifying major structural deviations.")
    if not recommendations:
        recommendations.append("Proceed to advanced feature engineering and predictive modeling.")

    return {
        "top_insights": deduplicate_list(top_insights),
        "recommendations": deduplicate_list(recommendations),
        "next_steps": deduplicate_list(next_steps),
        "risk_flags": deduplicate_list(risk_flags)
    }
