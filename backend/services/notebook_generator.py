import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os
from datetime import datetime
from typing import Optional


def generate_analysis_notebook(
    analysis_bundle: dict,
    output_path: str = "generated_notebook.ipynb",
) -> str:
    """
    Programmatically builds a complete Jupyter notebook (.ipynb) from
    structured analysis results produced by the AI Data Copilot pipeline.

    Parameters
    ----------
    analysis_bundle : dict with keys like dataset_summary, data_quality,
                      insights, visualizations, ml_results, xai_results,
                      recommendations, dataset_path (optional).
    output_path     : where to write the .ipynb file.

    Returns
    -------
    Absolute path to the created notebook.
    """
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = []

    # ── helpers ───────────────────────────────
    def md(text: str):
        cells.append(new_markdown_cell(text))

    def code(source: str):
        cells.append(new_code_cell(source))

    # ── 1. Title & Context ────────────────────
    ds_name = analysis_bundle.get("dataset_name", "Dataset")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    md(
        f"# 📊 AI Data Copilot — Analysis Report\n\n"
        f"**Dataset:** {ds_name}  \n"
        f"**Generated:** {date_str}  \n"
        f"**Engine:** AI Data Copilot (multi-agent system)\n\n"
        f"---"
    )

    # ── 2. Imports & Dataset Loading ──────────
    md("## 1. Setup & Data Loading")

    dataset_path = analysis_bundle.get("dataset_path", "data/dataset.csv")
    code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import plotly.express as px\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        f'df = pd.read_csv("{dataset_path}")\n'
        'print(f"Shape: {df.shape}")\n'
        "df.head()"
    )

    # ── 3. Dataset Overview ───────────────────
    summary = analysis_bundle.get("dataset_summary", {})
    md("## 2. Dataset Overview")
    if summary:
        rows = summary.get("rows", "N/A")
        cols = summary.get("columns", "N/A")
        features = summary.get("features", [])
        md(
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Rows | {rows:,} |\n"
            f"| Columns | {cols} |\n"
            f"| Key Features | {', '.join(features[:8])} |"
        )
    code(
        "df.info()\n"
        "df.describe()"
    )

    # ── 4. Data Quality / Cleaning ────────────
    md("## 3. Data Quality Analysis")
    dq = analysis_bundle.get("data_quality", {})
    if dq:
        dupes = dq.get("duplicates", 0)
        missing = dq.get("missing_values", {})
        missing_lines = "\n".join(f"- **{col}**: {cnt}" for col, cnt in missing.items() if cnt > 0) or "- None detected."
        md(
            f"**Duplicates found:** {dupes}  \n\n"
            f"**Missing values:**\n{missing_lines}"
        )

    code(
        "# Duplicates\n"
        'print(f"Duplicates: {df.duplicated().sum()}")\n'
        "\n"
        "# Missing values\n"
        "df.isnull().sum()[df.isnull().sum() > 0]"
    )
    code(
        "# Basic cleaning\n"
        "df_clean = df.drop_duplicates()\n"
        "df_clean = df_clean.dropna()  # or use imputation\n"
        'print(f"Rows after cleaning: {len(df_clean)}")'
    )

    # ── 5. EDA / Insights ─────────────────────
    md("## 4. Exploratory Data Analysis")
    insights = analysis_bundle.get("insights", [])
    if insights:
        insight_md = "\n".join(f"- {i}" for i in insights)
        md(f"**Key insights identified by AI agents:**\n\n{insight_md}")

    code(
        "# Correlation matrix\n"
        "numeric_df = df_clean.select_dtypes(include='number')\n"
        "if not numeric_df.empty:\n"
        "    fig = px.imshow(numeric_df.corr(), text_auto='.2f',\n"
        "                    title='Correlation Heatmap', color_continuous_scale='RdBu_r')\n"
        "    fig.show()"
    )

    # ── 6. Visualizations ─────────────────────
    md("## 5. Visual Analysis")
    viz_list = analysis_bundle.get("visualizations", [])
    if viz_list:
        for v in viz_list:
            md(f"### {v.get('title', 'Chart')}\n{v.get('description', '')}")

    code(
        "# Distribution of numeric columns\n"
        "for col in numeric_df.columns[:4]:\n"
        "    fig = px.histogram(df_clean, x=col, title=f'Distribution of {col}',\n"
        "                       marginal='box', template='plotly_white')\n"
        "    fig.show()"
    )

    # ── 7. Machine Learning ───────────────────
    md("## 6. Machine Learning")
    ml = analysis_bundle.get("ml_results", {})
    if ml:
        best = ml.get("best_model", "N/A")
        metrics = ml.get("metrics", {})
        metric_lines = "\n".join(f"- **{k.capitalize()}**: {v}" for k, v in metrics.items())
        md(f"**Best model:** {best}\n\n{metric_lines}")

        leaderboard = ml.get("leaderboard", [])
        if leaderboard:
            header = "| Model | " + " | ".join(k for k in leaderboard[0] if k != "Model") + " |"
            sep = "|---|" + "|".join("---" for k in leaderboard[0] if k != "Model") + "|"
            rows_md = "\n".join(
                "| " + " | ".join(str(v) for v in row.values()) + " |"
                for row in leaderboard
            )
            md(f"**Model Leaderboard:**\n\n{header}\n{sep}\n{rows_md}")

    target_col = analysis_bundle.get("target_column", "target")
    code(
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import classification_report\n"
        "\n"
        f'target = "{target_col}"\n'
        "if target in df_clean.columns:\n"
        "    X = df_clean.drop(columns=[target]).select_dtypes(include='number')\n"
        "    y = df_clean[target]\n"
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "    model = RandomForestClassifier(random_state=42)\n"
        "    model.fit(X_train, y_train)\n"
        "    print(classification_report(y_test, model.predict(X_test)))\n"
        "else:\n"
        '    print(f"Target column \'{target}\' not found.")'
    )

    # ── 8. Explainability ─────────────────────
    md("## 7. Model Explainability")
    xai = analysis_bundle.get("xai_results", {})
    if xai:
        nl_summary = xai.get("natural_language_summary", [])
        if nl_summary:
            md(" ".join(nl_summary))

    code(
        "# Feature importance (tree-based models)\n"
        "if target in df_clean.columns:\n"
        "    importances = model.feature_importances_\n"
        "    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})\n"
        "    feat_df = feat_df.sort_values('Importance', ascending=True)\n"
        "    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',\n"
        "                 title='Feature Importances', template='plotly_white')\n"
        "    fig.show()"
    )

    # ── 9. Recommendations / Conclusion ───────
    md("## 8. Recommendations & Conclusion")
    recs = analysis_bundle.get("recommendations", {})
    if isinstance(recs, dict):
        actions = recs.get("actions", recs.get("recommendations", []))
    elif isinstance(recs, list):
        actions = recs
    else:
        actions = []

    if actions:
        rec_md = "\n".join(f"- {a}" for a in actions)
        md(f"**Business Recommendations:**\n\n{rec_md}")

    md(
        "---\n\n"
        "*This notebook was automatically generated by the AI Data Copilot.*"
    )

    # ── Write notebook ────────────────────────
    nb.cells = cells
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return os.path.abspath(output_path)
