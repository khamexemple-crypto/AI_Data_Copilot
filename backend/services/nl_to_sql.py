import re
from typing import Dict, List

# ──────────────────────────────────────────────
# PATTERN-BASED NL-TO-SQL TRANSLATOR (v1)
# ──────────────────────────────────────────────
# A lightweight, rule-based translator that covers the most common
# analytical queries without requiring a full LLM call.
# Designed to be extended later with LLM-backed translation.


def _find_best_table(tables: List[str], query: str) -> str:
    """Heuristic: pick the table whose name appears in the query, or the first table."""
    q_lower = query.lower()
    for t in tables:
        if t.lower() in q_lower:
            return t
    return tables[0] if tables else "unknown_table"


def _find_column(columns: List[str], hint: str) -> str:
    """Fuzzy match a column name from a natural-language hint."""
    hint_lower = hint.lower().strip()
    # Exact match
    for c in columns:
        if c.lower() == hint_lower:
            return c
    # Substring match
    for c in columns:
        if hint_lower in c.lower() or c.lower() in hint_lower:
            return c
    return columns[0] if columns else "*"


# Each pattern is (compiled regex, handler function).
# Handlers receive (match, table, columns) and return an SQL string.

def _handle_top_n(match, table, columns):
    n = int(match.group("n"))
    col_hint = match.group("col").strip()
    col = _find_column(columns, col_hint)
    return f"SELECT * FROM {table} ORDER BY {col} DESC LIMIT {n}"


def _handle_count(match, table, columns):
    return f"SELECT COUNT(*) AS total FROM {table}"


def _handle_average(match, table, columns):
    col_hint = match.group("col").strip()
    col = _find_column(columns, col_hint)
    return f"SELECT AVG({col}) AS average_{col} FROM {table}"


def _handle_sum(match, table, columns):
    col_hint = match.group("col").strip()
    col = _find_column(columns, col_hint)
    return f"SELECT SUM({col}) AS total_{col} FROM {table}"


def _handle_group_by(match, table, columns):
    metric_hint = match.group("metric").strip()
    group_hint = match.group("group").strip()
    metric_col = _find_column(columns, metric_hint)
    group_col = _find_column(columns, group_hint)
    return f"SELECT {group_col}, SUM({metric_col}) AS total_{metric_col} FROM {table} GROUP BY {group_col} ORDER BY total_{metric_col} DESC LIMIT 20"


def _handle_grouped_aggregate(match, table, columns):
    """Handles 'sum revenue by region', 'total sales by category', 'average price by store'."""
    agg = match.group("agg").strip().lower()
    col_hint = match.group("col").strip()
    group_hint = match.group("group").strip()
    metric_col = _find_column(columns, col_hint)
    group_col = _find_column(columns, group_hint)
    agg_fn = "AVG" if agg in ("average", "avg", "mean", "moyenne") else "SUM"
    alias = f"{'avg' if agg_fn == 'AVG' else 'total'}_{metric_col}"
    return f"SELECT {group_col}, {agg_fn}({metric_col}) AS {alias} FROM {table} GROUP BY {group_col} ORDER BY {alias} DESC LIMIT 20"


def _handle_show_all(match, table, columns):
    return f"SELECT * FROM {table} LIMIT 50"


_PATTERNS = [
    # "show top 10 customers by revenue"
    (re.compile(r"(?:show|get|list|display)\s+(?:the\s+)?top\s+(?P<n>\d+)\s+.+?\s+by\s+(?P<col>\w+)", re.I), _handle_top_n),
    # "count rows" / "how many records" / "combien de lignes"
    (re.compile(r"(?:count|how many|combien|nombre)", re.I), _handle_count),
    # GROUPED AGGREGATES — must be checked BEFORE plain aggregates
    # "sum revenue by region" / "total sales by category" / "average price by store"
    (re.compile(r"(?P<agg>sum|total|somme|average|avg|mean|moyenne)\s+(?:of|de|du|des)?\s*(?P<col>\w+)\s+(?:by|per|par)\s+(?P<group>\w+)", re.I), _handle_grouped_aggregate),
    # "revenue by region" / "sales per category" (implicit sum group-by)
    (re.compile(r"(?P<metric>\w+)\s+(?:by|per|par)\s+(?P<group>\w+)", re.I), _handle_group_by),
    # PLAIN AGGREGATES — only match when there is no "by/per/par" following
    # "average of salary" / "moyenne de prix"
    (re.compile(r"(?:average|avg|mean|moyenne)\s+(?:of|de|du|des)?\s*(?P<col>\w+)", re.I), _handle_average),
    # "sum of revenue" / "total de ventes"
    (re.compile(r"(?:sum|total|somme)\s+(?:of|de|du|des)?\s*(?P<col>\w+)", re.I), _handle_sum),
    # "show all data" / "display table"
    (re.compile(r"(?:show|display|affiche|voir)\s+(?:all|tout|table|data|donn)", re.I), _handle_show_all),
]


def nl_to_sql(
    query: str,
    schema: Dict[str, List[Dict[str, str]]],
) -> Dict[str, str]:
    """
    Translates a simple natural-language analytical query into a safe SQL SELECT.

    Parameters
    ----------
    query  : user's natural-language question
    schema : {table_name: [{"column": ..., "type": ...}, ...]}

    Returns
    -------
    {"sql": "SELECT ...", "matched_pattern": "pattern_name", "table": "..."}
    or {"error": "Could not translate ..."}
    """
    if not schema:
        return {"error": "No database schema available."}

    tables = list(schema.keys())
    table = _find_best_table(tables, query)
    columns = [col["column"] for col in schema.get(table, [])]

    for pattern, handler in _PATTERNS:
        match = pattern.search(query)
        if match:
            try:
                sql = handler(match, table, columns)
                return {
                    "sql": sql,
                    "matched_pattern": pattern.pattern,
                    "table": table,
                }
            except Exception as e:
                return {"error": f"Pattern matched but SQL generation failed: {e}"}

    # Fallback: simple SELECT * with LIMIT
    return {
        "sql": f"SELECT * FROM {table} LIMIT 20",
        "matched_pattern": "fallback",
        "table": table,
    }
