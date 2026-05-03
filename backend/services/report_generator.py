import os
from datetime import datetime

try:
    from markdown_pdf import MarkdownPdf, Section
    PDF_ENABLED = True
except ImportError:
    PDF_ENABLED = False


def _iter_feature_importance(feature_importance):
    """
    Normalize feature importance into (feature, score) pairs.
    Accepts dicts, list[dict], list[tuple], and ignores malformed rows.
    """
    if isinstance(feature_importance, dict):
        yield from feature_importance.items()
        return

    if isinstance(feature_importance, list):
        for item in feature_importance:
            if isinstance(item, dict):
                feature = item.get("feature") or item.get("name")
                score = item.get("importance", item.get("score", item.get("value", "N/A")))
                if feature:
                    yield feature, score
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                yield item[0], item[1]

def generate_report(
    dataset_summary: dict,
    analysis_results: dict,
    visualization_summaries: dict,
    ml_results: dict,
    xai_results: dict,
    recommendations: dict,
    output_format: str = "markdown"
) -> dict:
    """
    Converts structured technical outputs into a polished Markdown consultant report.
    Optionally exports to PDF.
    """
    md_lines = []
    
    # 1. Executive Summary
    md_lines.append("# Data Science Consultant Report")
    md_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
    md_lines.append("## 1. Executive Summary")
    md_lines.append("This report presents a comprehensive analysis of the provided dataset, highlighting key insights, data quality metrics, and predictive modeling results. The goal is to provide actionable recommendations based on data evidence.\n")

    # 2. Dataset Overview
    md_lines.append("## 2. Dataset Overview")
    if dataset_summary:
        md_lines.append(f"- **Rows:** {dataset_summary.get('rows', 'N/A')}")
        md_lines.append(f"- **Columns:** {dataset_summary.get('columns', 'N/A')}")
        features = dataset_summary.get('features', [])
        if features:
            md_lines.append(f"- **Key Features:** {', '.join(features)}")
    else:
        md_lines.append("No dataset overview available.")
    md_lines.append("\n")

    # 3. Data Quality Analysis
    md_lines.append("## 3. Data Quality Analysis")
    if analysis_results and "data_quality" in analysis_results:
        dq = analysis_results["data_quality"]
        md_lines.append(f"- **Duplicates Found:** {dq.get('duplicates', 0)}")
        missing = dq.get('missing_values', {})
        if missing:
            md_lines.append("- **Missing Values:**")
            for col, count in missing.items():
                md_lines.append(f"  - {col}: {count}")
    else:
        md_lines.append("Data quality metrics were not provided.\n")
    md_lines.append("\n")

    # 4. Exploratory Data Insights
    md_lines.append("## 4. Exploratory Data Insights")
    if analysis_results and "insights" in analysis_results:
        for insight in analysis_results["insights"]:
            md_lines.append(f"- {insight}")
    else:
        md_lines.append("No exploratory insights available.")
    md_lines.append("\n")

    # 5. Visual Analysis Summary
    md_lines.append("## 5. Visual Analysis Summary")
    if visualization_summaries and "charts" in visualization_summaries:
        for viz in visualization_summaries["charts"]:
            md_lines.append(f"### {viz.get('title', 'Chart')}")
            md_lines.append(f"{viz.get('description', '')}\n")
    else:
        md_lines.append("No visual summaries provided.\n")

    # 6. Machine Learning Results
    md_lines.append("## 6. Machine Learning Results")
    if ml_results:
        md_lines.append(f"- **Best Model:** {ml_results.get('best_model', 'N/A')}")
        metrics = ml_results.get('metrics', {})
        if metrics:
            md_lines.append("- **Metrics:**")
            for m, val in metrics.items():
                md_lines.append(f"  - {m.capitalize()}: {val}")
    else:
        md_lines.append("No machine learning results available.\n")
    md_lines.append("\n")

    # 7. Model Explanation
    md_lines.append("## 7. Model Explanation")
    if xai_results and "feature_importance" in xai_results:
        fi = xai_results["feature_importance"]
        feature_pairs = list(_iter_feature_importance(fi))
        if feature_pairs:
            md_lines.append("The following features were the most influential in the model's predictions:")
            for feat, score in feature_pairs:
                md_lines.append(f"- **{feat}**: {score}")
        else:
            md_lines.append("Feature importance format not recognized.")
    else:
        md_lines.append("Model explanations not provided.\n")
    md_lines.append("\n")

    # 8. Business Recommendations
    md_lines.append("## 8. Business Recommendations")
    if recommendations and "actions" in recommendations:
        for action in recommendations["actions"]:
            md_lines.append(f"- {action}")
    else:
        md_lines.append("No specific business recommendations generated.")
    md_lines.append("\n")
    
    # 9. Conclusion
    md_lines.append("## 9. Conclusion")
    md_lines.append("The analysis confirms several key trends that can be leveraged for strategic decision-making. Continuous monitoring and data quality improvements are recommended for future iterations.")

    markdown_text = "\n".join(md_lines)
    result = {"markdown": markdown_text}

    if output_format.lower() == "pdf":
        if PDF_ENABLED:
            pdf_path = os.path.join(os.getcwd(), f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
            try:
                pdf = MarkdownPdf(toc_level=2)
                pdf.add_section(Section(markdown_text))
                pdf.save(pdf_path)
                result["pdf_path"] = pdf_path
            except Exception as e:
                result["pdf_error"] = str(e)
        else:
            result["pdf_error"] = "markdown-pdf package is not installed."

    return result
