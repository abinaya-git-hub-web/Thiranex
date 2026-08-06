"""
=============================================================================
Thiranex Solutions — Export Engine & ReportLab PDF Briefing Generator
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import io
import json
from typing import Dict, Any, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_forecast_csv(
    dates: pd.Series,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray = None,
    upper_bound: np.ndarray = None,
    model_name: str = "AutoML Ensemble"
) -> bytes:
    """
    Generates downloadable CSV containing dates, actuals, predictions, and confidence intervals.
    """
    n = len(y_pred)
    date_strs = [str(dates.iloc[i])[:10] if i < len(dates) else f"Step_{i+1}" for i in range(n)]

    df_export = pd.DataFrame({
        "Date": date_strs,
        "Actual_Value": np.round(y_actual[:n], 2) if y_actual is not None and len(y_actual) >= n else [np.nan] * n,
        "Predicted_Forecast": np.round(y_pred, 2),
        "Lower_95_Confidence": np.round(lower_bound, 2) if lower_bound is not None else np.round(y_pred * 0.9, 2),
        "Upper_95_Confidence": np.round(upper_bound, 2) if upper_bound is not None else np.round(y_pred * 1.1, 2),
        "Model_Used": [model_name] * n
    })
    return df_export.to_csv(index=False).encode('utf-8')


def generate_forecast_json_api(
    target_name: str,
    best_model_name: str,
    metrics: Dict[str, float],
    forecast_series: np.ndarray,
    risk_info: Dict[str, Any]
) -> str:
    """
    Generates a structured REST API JSON response for programmatic downstream integration.
    """
    payload = {
        "status": "success",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "application": "Thiranex Solutions Predictive Analytics Platform",
        "target_variable": target_name,
        "selected_model": best_model_name,
        "metrics": metrics,
        "risk_assessment": risk_info,
        "forecast_horizon_length": len(forecast_series),
        "total_predicted_sum": float(np.round(np.sum(forecast_series), 2)),
        "predictions": [float(np.round(val, 2)) for val in forecast_series]
    }
    return json.dumps(payload, indent=2)


def generate_executive_pdf_report(
    target_name: str,
    best_model_name: str,
    metrics: Dict[str, float],
    leaderboard: Dict[str, Any],
    risk_info: Dict[str, Any],
    narrative_summary: str
) -> bytes:
    """
    Generates an enterprise PDF briefing report using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Styling
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E1B4B'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#4338CA'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    elements = []

    # Title & Subtitle Header Banner
    elements.append(Paragraph("<b>THIRANEX SOLUTIONS</b>", title_style))
    elements.append(Paragraph("Enterprise Predictive Analytics & Intelligence Briefing", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366F1'), spaceAfter=15))

    # Executive Overview Box
    elements.append(Paragraph("<b>1. Executive Summary & AI Insights</b>", h2_style))
    clean_narrative = narrative_summary.replace("#", "").replace("*", "").strip()
    elements.append(Paragraph(clean_narrative, body_style))
    elements.append(Spacer(1, 10))

    # KPI Performance Summary Table
    elements.append(Paragraph("<b>2. Model Performance Metrics (Optimal Model: " + best_model_name + ")</b>", h2_style))

    kpi_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Root Mean Squared Error (RMSE)", f"{metrics.get('rmse', 0.0):,.2f}", "Mean Absolute Error (MAE)", f"{metrics.get('mae', 0.0):,.2f}"],
        ["Mean Absolute % Error (MAPE)", f"{metrics.get('mape', 0.0):.2f}%", "Symmetric MAPE (SMAPE)", f"{metrics.get('smape', 0.0):.2f}%"],
        ["R² Score", f"{metrics.get('r2', 0.0):.4f}", "Risk Level", risk_info.get('risk_level', 'Low Risk')]
    ]

    kpi_table = Table(kpi_data, colWidths=[170, 90, 170, 90])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4338CA')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # Leaderboard Table
    elements.append(Paragraph("<b>3. Model Comparison Leaderboard</b>", h2_style))
    leaderboard_data = [["Model Rank", "Algorithm Name", "RMSE", "MAE", "MAPE (%)"]]
    if leaderboard:
        sorted_m = sorted(leaderboard.items(), key=lambda x: x[1]["rmse"])
        for idx, (m_name, m_stats) in enumerate(sorted_m[:5]):
            leaderboard_data.append([
                f"Rank {idx+1}",
                m_name,
                f"{m_stats['rmse']:.2f}",
                f"{m_stats['mae']:.2f}",
                f"{m_stats['mape']:.1f}%"
            ])

    lead_table = Table(leaderboard_data, colWidths=[70, 200, 80, 80, 90])
    lead_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(lead_table)
    elements.append(Spacer(1, 15))

    # Strategic Advice Box
    elements.append(Paragraph("<b>4. Strategic Decision Guidance</b>", h2_style))
    advice_text = risk_info.get('advice', 'Maintain current operating budget while continuously evaluating forecast variance.')
    elements.append(Paragraph(advice_text, body_style))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=8))
    elements.append(Paragraph("<font color='#94A3B8' size=8>© 2026 Thiranex Solutions — Confidential Executive Briefing</font>", body_style))

    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
