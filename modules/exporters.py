"""
=============================================================================
Thiranex Solutions — Multi-Format Export Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import io
import json
from typing import Tuple, Dict, Any, Optional
from modules import reporting_engine as re_engine

def export_dataframe_bytes(df: pd.DataFrame, fmt: str) -> Tuple[bytes, str, str]:
    """
    Exports a DataFrame into binary bytes with format extension and mime-type.
    Formats: 'CSV', 'Excel (.xlsx)', 'JSON', 'Parquet'
    """
    if fmt == "CSV":
        csv_str = df.to_csv(index=False)
        return csv_str.encode("utf-8"), "thiranex_cleaned_data.csv", "text/csv"
    elif fmt == "Excel (.xlsx)":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        return buffer.getvalue(), "thiranex_cleaned_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "JSON":
        json_str = df.to_json(orient="records", indent=2)
        return json_str.encode("utf-8"), "thiranex_cleaned_data.json", "application/json"
    elif fmt == "Parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue(), "thiranex_cleaned_data.parquet", "application/octet-stream"
    else:
        csv_str = df.to_csv(index=False)
        return csv_str.encode("utf-8"), "thiranex_cleaned_data.csv", "text/csv"

def export_report_bytes(df_clean: pd.DataFrame, profile: dict, audit_log: pd.DataFrame, data_dict: pd.DataFrame, report_fmt: str) -> Tuple[bytes, str, str]:
    """
    Generates multi-format report exports: 'PDF', 'Excel Workbook', 'HTML', 'JSON Metadata'
    """
    if report_fmt == "Excel Workbook":
        b_data = re_engine.generate_excel_report(df_clean, profile, audit_log, data_dict)
        return b_data, "thiranex_quality_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif report_fmt == "PDF":
        b_data = re_engine.generate_pdf_report(df_clean, profile, audit_log)
        return b_data, "thiranex_quality_report.pdf", "application/pdf"
    elif report_fmt == "HTML":
        html_str = re_engine.generate_html_report(df_clean, profile, audit_log)
        return html_str.encode("utf-8"), "thiranex_quality_report.html", "text/html"
    elif report_fmt == "JSON Metadata":
        meta_dict = {
            "scorecard": profile.get("scorecard", {}),
            "structure": profile.get("structure", {}),
            "detected_types": profile.get("detected_types", {}),
            "audit_trail": audit_log.to_dict(orient="records") if audit_log is not None else []
        }
        json_bytes = json.dumps(meta_dict, indent=2).encode("utf-8")
        return json_bytes, "thiranex_metadata_report.json", "application/json"
    else:
        html_str = re_engine.generate_html_report(df_clean, profile, audit_log)
        return html_str.encode("utf-8"), "thiranex_quality_report.html", "text/html"
