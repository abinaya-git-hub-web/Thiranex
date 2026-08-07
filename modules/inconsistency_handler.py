"""
=============================================================================
Thiranex Solutions — Inconsistent Data Handler Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import re
from typing import Dict, Any, List, Tuple

def handle_inconsistencies(df: pd.DataFrame, text_case: str = "Title Case", date_cols: List[str] = None, phone_cols: List[str] = None, convert_units: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes text casing, date formats, phone numbers, and unit conversions.
    """
    df_clean = df.copy()
    changes_log = []

    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        if text_case == "Upper Case":
            df_clean[c] = df_clean[c].astype(str).str.upper()
        elif text_case == "Lower Case":
            df_clean[c] = df_clean[c].astype(str).str.lower()
        elif text_case == "Title Case":
            df_clean[c] = df_clean[c].astype(str).str.title().str.strip()
    if len(cat_cols) > 0:
        changes_log.append(f"Standardized text casing to {text_case}")

    if date_cols:
        for dc in date_cols:
            if dc in df_clean.columns:
                df_clean[dc] = pd.to_datetime(df_clean[dc], errors="coerce").dt.strftime("%Y-%m-%d")
                changes_log.append(f"Parsed dates in '{dc}' to YYYY-MM-DD")

    if phone_cols:
        for pc in phone_cols:
            if pc in df_clean.columns:
                def _fmt_phone(val):
                    digits = re.sub(r"\D", "", str(val))
                    if len(digits) == 10:
                        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                    elif len(digits) == 11 and digits.startswith("1"):
                        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
                    return val
                df_clean[pc] = df_clean[pc].apply(_fmt_phone)
                changes_log.append(f"Standardized phone numbers in '{pc}'")

    if convert_units:
        for col, mode in convert_units.items():
            if col in df_clean.columns:
                def _convert(val):
                    try:
                        s_val = str(val).lower()
                        num = float(re.findall(r"[-+]?\d*\.\d+|\d+", s_val)[0])
                        if mode == "kg -> lbs" and "kg" in s_val:
                            return f"{round(num * 2.20462, 1)} lbs"
                        elif mode == "lbs -> kg" and "lbs" in s_val:
                            return f"{round(num / 2.20462, 1)} kg"
                        return val
                    except Exception:
                        return val
                df_clean[col] = df_clean[col].apply(_convert)
                changes_log.append(f"Converted units in '{col}' ({mode})")

    return df_clean, {"changes": changes_log}
