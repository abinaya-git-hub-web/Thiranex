"""
=============================================================================
Thiranex Solutions — Duplicate Management Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple

def handle_duplicates(df: pd.DataFrame, method: str = "Exact Duplicates", keep: str = "first", similarity_threshold: float = 0.85, text_col: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Purges exact or fuzzy duplicates from the DataFrame.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    if method == "Exact Duplicates":
        keep_arg = keep if keep in ["first", "last"] else False
        df_clean = df_clean.drop_duplicates(keep=keep_arg).reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        return df_clean, {"method": "Exact Deduplication", "initial_rows": initial_rows, "final_rows": len(df_clean), "removed_records": removed}

    elif method == "Fuzzy Record Linkage" and text_col and text_col in df_clean.columns:
        series = df_clean[text_col].astype(str).tolist()
        to_drop = set()
        n = len(series)
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, min(i + 50, n)):
                if j in to_drop:
                    continue
                ratio = SequenceMatcher(None, series[i].lower(), series[j].lower()).ratio()
                if ratio >= similarity_threshold:
                    to_drop.add(j)

        df_clean = df_clean.drop(index=list(to_drop)).reset_index(drop=True)
        removed = len(to_drop)
        return df_clean, {"method": f"Fuzzy Linkage ({text_col} >= {similarity_threshold})", "initial_rows": initial_rows, "final_rows": len(df_clean), "removed_records": removed}

    return df_clean, {"method": "None", "initial_rows": initial_rows, "final_rows": initial_rows, "removed_records": 0}
