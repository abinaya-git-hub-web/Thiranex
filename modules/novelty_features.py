"""
=============================================================================
Thiranex Solutions — Novelty Enterprise AI Features Suite
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

class DataLineageTracker:
    """
    Manages audit logging, step version history, and step-by-step undo/rollback capabilities.
    """
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        
    def record_step(self, step_name: str, df: pd.DataFrame, metrics: Dict[str, Any] = None):
        snapshot = {
            "step_index": len(self.history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step_name": step_name,
            "row_count": len(df),
            "col_count": len(df.columns),
            "df_snapshot": df.copy(),
            "metrics": metrics or {}
        }
        self.history.append(snapshot)

    def rollback_to_step(self, step_index: int) -> Tuple[Optional[pd.DataFrame], str]:
        if 1 <= step_index <= len(self.history):
            self.history = self.history[:step_index]
            return self.history[-1]["df_snapshot"].copy(), f"Rolled back to Step {step_index}: '{self.history[-1]['step_name']}'"
        return None, "Invalid step index for rollback."

    def get_audit_trail_df(self) -> pd.DataFrame:
        logs = []
        for h in self.history:
            logs.append({
                "Step #": h["step_index"],
                "Timestamp": h["timestamp"],
                "Operation": h["step_name"],
                "Rows": h["row_count"],
                "Columns": h["col_count"],
                "Details": str(h["metrics"])
            })
        return pd.DataFrame(logs)

def detect_data_drift(df_baseline: pd.DataFrame, df_current: pd.DataFrame) -> Dict[str, Any]:
    """
    Smart Data Health Monitor: Calculates distribution shift, mean drift, and pattern anomalies between baseline and current data.
    """
    if df_baseline is None or df_current is None or df_baseline.empty or df_current.empty:
        return {"drift_detected": False, "drift_summary": "Insufficient data for drift analysis."}

    num_cols = df_baseline.select_dtypes(include=[np.number]).columns.intersection(df_current.columns)
    drift_details = []
    has_drift = False

    for c in num_cols:
        b_mean = df_baseline[c].mean()
        c_mean = df_current[c].mean()
        b_std = df_baseline[c].std()
        
        if b_std > 0:
            z_shift = abs(c_mean - b_mean) / b_std
            if z_shift > 0.5: # Significant shift
                has_drift = True
                drift_details.append({
                    "Column": c,
                    "Baseline_Mean": round(float(b_mean), 2),
                    "Current_Mean": round(float(c_mean), 2),
                    "Z_Shift": round(float(z_shift), 2),
                    "Status": "⚠️ High Shift" if z_shift > 1.5 else "⚡ Moderate Shift"
                })

    return {
        "drift_detected": has_drift,
        "drift_count": len(drift_details),
        "details_df": pd.DataFrame(drift_details) if drift_details else pd.DataFrame(),
        "summary": f"Detected distribution drift in {len(drift_details)} numerical columns." if has_drift else "No significant data drift detected."
    }

def talk_to_your_data_query(df: pd.DataFrame, query_str: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Translates natural language questions into pandas filter/query expressions.
    Examples:
    - 'show annual spend > 5000'
    - 'filter age between 20 and 40'
    - 'show rows where city is New York'
    - 'top 10 credit score'
    """
    if df is None or df.empty or not query_str.strip():
        return df, "Query string empty."

    q_lower = query_str.lower().strip()
    
    try:
        # Match 'top N by COLUMN'
        top_match = re.search(r"top\s+(\d+)\s+(?:by\s+)?([a-zA-Z0-9_]+)", q_lower)
        if top_match:
            n = int(top_match.group(1))
            col_target = top_match.group(2)
            # Find matching column name
            matched_col = [c for c in df.columns if col_target in c.lower()]
            if matched_col:
                res = df.nlargest(n, matched_col[0])
                return res, f"Query Executed: Showing top {n} rows sorted by '{matched_col[0]}'."

        # Match 'COLUMN > NUMBER' or 'COLUMN < NUMBER'
        num_match = re.search(r"([a-zA-Z0-9_]+)\s*(>|<|>=|<=|==|=)\s*(\d+(?:\.\d+)?)", q_lower)
        if num_match:
            col_name = num_match.group(1)
            op = num_match.group(2)
            if op == "=": op = "=="
            val = float(num_match.group(3))
            matched_col = [c for c in df.columns if col_name in c.lower()]
            if matched_col:
                target = matched_col[0]
                expr = f"`{target}` {op} {val}"
                res = df.query(expr)
                return res, f"Query Executed: `{expr}` ({len(res)} matching records found)"

        # Match 'where COLUMN is VALUE'
        text_match = re.search(r"(?:where|is|in)\s+([a-zA-Z0-9_]+)\s+(?:is|=|in)\s+['\"]?([a-zA-Z0-9_\s]+)['\"]?", q_lower)
        if text_match:
            col_name = text_match.group(1)
            target_val = text_match.group(2).strip()
            matched_col = [c for c in df.columns if col_name in c.lower()]
            if matched_col:
                target = matched_col[0]
                res = df[df[target].astype(str).str.lower().str.contains(target_val, na=False)]
                return res, f"Query Executed: Filtered '{target}' containing '{target_val}' ({len(res)} records found)"

        # Fallback: substring search across all columns
        res = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(q_lower).any(), axis=1)]
        return res, f"Search Executed: Keyword '{query_str}' matched {len(res)} rows."

    except Exception as e:
        return df, f"Could not parse natural query: {str(e)}"

def generate_smart_metadata_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart Metadata Generator: Produces an automated Data Dictionary documenting data types, descriptions, null counts, and sample values.
    """
    dict_rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        null_cnt = int(s.isna().sum())
        unique_cnt = int(s.nunique())
        sample_vals = ", ".join([str(v) for v in s.dropna().unique()[:4]])
        
        # Inferred Description
        col_lower = col.lower()
        if "id" in col_lower:
            desc = "Unique record identifier"
        elif "date" in col_lower or "time" in col_lower:
            desc = "Timestamp / date attribute"
        elif "email" in col_lower:
            desc = "Customer electronic mail address"
        elif "phone" in col_lower:
            desc = "Contact phone number"
        elif "spend" in col_lower or "amount" in col_lower or "salary" in col_lower or "price" in col_lower:
            desc = "Financial monetary measurement"
        elif "score" in col_lower or "age" in col_lower or "count" in col_lower:
            desc = "Quantitative numerical metric"
        else:
            desc = "Categorical / text feature attribute"

        dict_rows.append({
            "Column Name": col,
            "Data Type": dtype,
            "Inferred Description": desc,
            "Null Count": null_cnt,
            "Completeness %": round(((len(df) - null_cnt) / max(len(df), 1)) * 100.0, 1),
            "Cardinality": unique_cnt,
            "Sample Values": sample_vals
        })
    return pd.DataFrame(dict_rows)

def validate_custom_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom Validation Framework: Validates user-defined rules (Range checks, Regex pattern matching, Non-Null constraints).
    Rule structure: {'column': 'Age', 'rule_type': 'Range', 'min': 0, 'max': 120}
    """
    if df is None or df.empty or not rules:
        return {"passed": True, "violations_count": 0, "log": []}

    violations = []
    total_violating_rows = 0

    for r in rules:
        col = r.get("column")
        rule_type = r.get("rule_type")
        if col not in df.columns:
            continue

        if rule_type == "Range Check":
            min_val, max_val = r.get("min", -1e9), r.get("max", 1e9)
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            cnt = int(invalid_mask.sum())
            if cnt > 0:
                total_violating_rows += cnt
                violations.append(f"❌ Range Violation on '{col}': {cnt} rows outside [{min_val}, {max_val}]")
        
        elif rule_type == "Non-Null Constraint":
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                total_violating_rows += cnt
                violations.append(f"❌ Null Violation on '{col}': {cnt} missing values found.")

        elif rule_type == "Regex Match":
            pattern = r.get("regex", r".*")
            invalid_mask = df[col].dropna().astype(str).apply(lambda x: not bool(re.match(pattern, x)))
            cnt = int(invalid_mask.sum())
            if cnt > 0:
                total_violating_rows += cnt
                violations.append(f"❌ Pattern Violation on '{col}': {cnt} values fail regex '{pattern}'")

    return {
        "passed": len(violations) == 0,
        "violations_count": total_violating_rows,
        "log": violations if violations else ["✅ All custom validation rules passed successfully!"]
    }
