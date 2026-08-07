"""
=============================================================================
Thiranex Solutions — Visual Pipeline Builder & Automation Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import json
from typing import Dict, Any, List, Tuple, Optional
from modules import cleaning_engine as ce
from modules import profiling_engine as pe

DEFAULT_PIPELINE_PRESET = [
    {"step_id": 1, "action": "Impute Missing Values", "strategy": "Auto-Select", "enabled": True},
    {"step_id": 2, "action": "Remove Duplicates", "method": "Exact Duplicates", "enabled": True},
    {"step_id": 3, "action": "Standardize Text & Dates", "casing": "Title Case", "enabled": True},
    {"step_id": 4, "action": "Handle Outliers", "method": "IQR Method", "handling": "Capping (Winsorizing)", "enabled": True},
    {"step_id": 5, "action": "Optimize Memory Types", "enabled": True}
]

def execute_pipeline(df: pd.DataFrame, pipeline_steps: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes a sequence of ETL data cleaning and transformation steps on the DataFrame.
    Returns: (cleaned_df, step_execution_logs, overall_impact_metrics)
    """
    if df is None or df.empty:
        return df, [], {}

    df_current = df.copy()
    initial_rows, initial_cols = df_current.shape
    initial_profile = pe.generate_comprehensive_profile(df_current)
    initial_score = initial_profile.get("scorecard", {}).get("overall_score", 0)

    logs = []
    
    for step in pipeline_steps:
        if not step.get("enabled", True):
            continue
            
        act = step.get("action")
        
        if act == "Impute Missing Values":
            strat = step.get("strategy", "Auto-Select")
            df_current, meta = ce.clean_missing_values(df_current, strategy=strat)
            logs.append({"step": act, "details": f"Imputed missing cells via {strat} (imputed {meta.get('imputed_cells', 0)} cells)"})
            
        elif act == "Remove Duplicates":
            mth = step.get("method", "Exact Duplicates")
            df_current, meta = ce.handle_duplicates(df_current, method=mth)
            logs.append({"step": act, "details": f"Removed {meta.get('removed_records', 0)} duplicate rows via {mth}"})
            
        elif act == "Standardize Text & Dates":
            casing = step.get("casing", "Title Case")
            df_current, meta = ce.standardize_data(df_current, text_case=casing)
            logs.append({"step": act, "details": f"Standardized text to {casing} & dates to ISO format"})
            
        elif act == "Handle Outliers":
            mth = step.get("method", "IQR Method")
            hnd = step.get("handling", "Capping (Winsorizing)")
            df_current, meta = ce.detect_and_handle_outliers(df_current, method=mth, action=hnd)
            logs.append({"step": act, "details": f"Handled {meta.get('outliers_detected', 0)} outliers via {mth} ({hnd})"})
            
        elif act == "Optimize Memory Types":
            df_current, meta = ce.optimize_memory_dtypes(df_current)
            logs.append({"step": act, "details": f"Memory downcasted: saved {meta.get('saved_pct', 0)}% RAM"})

    final_rows, final_cols = df_current.shape
    final_profile = pe.generate_comprehensive_profile(df_current)
    final_score = final_profile.get("scorecard", {}).get("overall_score", 0)

    impact = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "initial_cols": initial_cols,
        "final_cols": final_cols,
        "initial_score": initial_score,
        "final_score": final_score,
        "score_improvement": round(final_score - initial_score, 1)
    }

    return df_current, logs, impact

def export_pipeline_preset(pipeline_steps: List[Dict[str, Any]]) -> str:
    """Exports pipeline config as JSON string."""
    return json.dumps(pipeline_steps, indent=2)

def import_pipeline_preset(json_str: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Imports pipeline config from JSON string."""
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data, None
        return DEFAULT_PIPELINE_PRESET, "Invalid JSON structure. Expected a list of pipeline step dictionaries."
    except Exception as e:
        return DEFAULT_PIPELINE_PRESET, f"Error parsing pipeline JSON: {str(e)}"
