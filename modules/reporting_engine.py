"""
=============================================================================
Thiranex Solutions — Automated Multi-Format Reporting Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

# OpenPyXL imports for multi-tab formatted Excel workbooks
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_excel_report(df_clean: pd.DataFrame, profile: Dict[str, Any], audit_log: pd.DataFrame, data_dict: pd.DataFrame) -> bytes:
    """
    Generates a multi-tab enterprise Excel workbook containing:
    1. Executive Summary & Quality Scorecard
    2. Cleaned Dataset
    3. Column Quality Profile
    4. Audit Trail Log
    5. Data Dictionary
    """
    wb = openpyxl.Workbook()
    
    # ---------------------------------------------------------
    # Sheet 1: Executive Summary & Scorecard
    # ---------------------------------------------------------
    ws_sum = wb.active
    ws_sum.title = "Executive Summary"
    ws_sum.views.sheetView[0].showGridLines = True

    # Styling setup
    header_fill = PatternFill(start_color="1E1E2F", end_color="1E1E2F", fill_type="solid") # Dark Navy
    accent_fill = PatternFill(start_color="6366F1", end_color="6366F1", fill_type="solid") # Indigo Accent
    card_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    
    title_font = Font(name="Segoe UI", size=18, bold=True, color="FFFFFF")
    sub_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    kpi_font = Font(name="Segoe UI", size=22, bold=True, color="1E1E2F")
    lbl_font = Font(name="Segoe UI", size=10, italic=True, color="64748B")

    # Header Banner
    ws_sum.merge_cells("A1:G2")
    cell_hdr = ws_sum["A1"]
    cell_hdr.value = "THIRANEX SOLUTIONS — DATA QUALITY & CLEANING REPORT"
    cell_hdr.font = title_font
    cell_hdr.fill = header_fill
    cell_hdr.alignment = Alignment(horizontal="center", vertical="center")

    scorecard = profile.get("scorecard", {})
    overall_score = scorecard.get("overall_score", 0)
    badge = scorecard.get("badge", "N/A")

    # Scorecard KPI Cards
    kpis = [
        ("A4:B5", "OVERALL DATA HEALTH", f"{overall_score} / 100", f"Status: {badge}"),
        ("C4:D5", "COMPLETENESS SCORE", f"{scorecard.get('completeness_score', 0)}%", f"Missing: {scorecard.get('total_missing', 0)} cells"),
        ("E4:F5", "ACCURACY SCORE", f"{scorecard.get('accuracy_score', 0)}%", f"Outliers: {scorecard.get('total_outliers', 0)}"),
        ("G4:H5", "CONSISTENCY SCORE", f"{scorecard.get('consistency_score', 0)}%", f"Duplicates: {scorecard.get('total_duplicates', 0)}")
    ]

    for rng, title, val, sub in kpis:
        ws_sum.merge_cells(rng)
        top_left = ws_sum[rng.split(":")[0]]
        top_left.value = f"{title}\n\n{val}\n{sub}"
        top_left.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        top_left.fill = card_fill
        top_left.font = Font(name="Segoe UI", size=11, bold=True, color="0F172A")

    # Recommendations Section
    ws_sum["A7"] = "Actionable Recommendations & Quality Insights"
    ws_sum["A7"].font = Font(name="Segoe UI", size=13, bold=True, color="6366F1")
    
    recs = scorecard.get("recommendations", [])
    for idx, r in enumerate(recs, start=8):
        ws_sum[f"A{idx}"] = f"• {r.replace('**', '')}"
        ws_sum[f"A{idx}"].font = Font(name="Segoe UI", size=10)

    # ---------------------------------------------------------
    # Sheet 2: Cleaned Dataset
    # ---------------------------------------------------------
    ws_data = wb.create_sheet(title="Cleaned Data")
    for r in dataframe_to_rows(df_clean, index=False, header=True):
        ws_data.append(r)
        
    # Format header row
    for cell in ws_data[1]:
        cell.fill = header_fill
        cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center")

    # ---------------------------------------------------------
    # Sheet 3: Column Profiles
    # ---------------------------------------------------------
    if "columns_summary" in profile and isinstance(profile["columns_summary"], pd.DataFrame):
        ws_col = wb.create_sheet(title="Column Quality Profile")
        for r in dataframe_to_rows(profile["columns_summary"], index=False, header=True):
            ws_col.append(r)
        for cell in ws_col[1]:
            cell.fill = accent_fill
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")

    # ---------------------------------------------------------
    # Sheet 4: Audit Log
    # ---------------------------------------------------------
    if audit_log is not None and not audit_log.empty:
        ws_audit = wb.create_sheet(title="Audit Trail Log")
        for r in dataframe_to_rows(audit_log, index=False, header=True):
            ws_audit.append(r)
        for cell in ws_audit[1]:
            cell.fill = header_fill
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")

    # ---------------------------------------------------------
    # Sheet 5: Data Dictionary
    # ---------------------------------------------------------
    if data_dict is not None and not data_dict.empty:
        ws_dict = wb.create_sheet(title="Data Dictionary")
        for r in dataframe_to_rows(data_dict, index=False, header=True):
            ws_dict.append(r)
        for cell in ws_dict[1]:
            cell.fill = accent_fill
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")

    # Auto-adjust column widths
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 40)

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()

def generate_pdf_report(df_clean: pd.DataFrame, profile: Dict[str, Any], audit_log: pd.DataFrame) -> bytes:
    """
    Generates a PDF Executive Briefing via ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle("TitleStyle", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#1E1E2F"), spaceAfter=10)
    h2_style = ParagraphStyle("H2Style", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#6366F1"), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("BodyStyle", fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#334155"))

    elements = []

    # Title Banner
    elements.append(Paragraph("<b>THIRANEX SOLUTIONS</b>", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#6366F1"))))
    elements.append(Paragraph("Enterprise Data Cleaning & Quality Executive Briefing", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", body_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#6366F1"), spaceAfter=15))

    scorecard = profile.get("scorecard", {})
    overall_score = scorecard.get("overall_score", 0)
    badge = scorecard.get("badge", "N/A")

    # Scorecard Table
    score_data = [
        ["Overall Health Score", "Completeness", "Accuracy", "Consistency", "Timeliness"],
        [f"{overall_score} / 100 ({badge})", f"{scorecard.get('completeness_score', 0)}%", f"{scorecard.get('accuracy_score', 0)}%", f"{scorecard.get('consistency_score', 0)}%", f"{scorecard.get('timeliness_score', 92)}%"]
    ]
    t_score = Table(score_data, colWidths=[110, 100, 100, 100, 100])
    t_score.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E1E2F")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1"))
    ]))
    elements.append(t_score)
    elements.append(Spacer(1, 15))

    # Executive Summary Text
    elements.append(Paragraph("Executive Summary & Recommendations", h2_style))
    recs = scorecard.get("recommendations", [])
    for r in recs:
        clean_r = r.replace("**", "").replace("⚠️", "[Warning]").replace("🔄", "[Duplicate]").replace("📈", "[Outlier]").replace("✅", "[OK]")
        elements.append(Paragraph(f"• {clean_r}", body_style))
        elements.append(Spacer(1, 4))
        
    elements.append(Spacer(1, 10))

    # Audit Trail Table
    if audit_log is not None and not audit_log.empty:
        elements.append(Paragraph("Data Cleaning Audit Log", h2_style))
        log_rows = [["Step #", "Timestamp", "Operation", "Rows"]]
        for _, row in audit_log.iterrows():
            log_rows.append([str(row.get("Step #", "")), str(row.get("Timestamp", "")), str(row.get("Operation", "")), str(row.get("Rows", ""))])
        
        t_log = Table(log_rows[:10], colWidths=[50, 130, 230, 80])
        t_log.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#6366F1")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0"))
        ]))
        elements.append(t_log)

    doc.build(elements)
    return buffer.getvalue()

def generate_html_report(df_clean: pd.DataFrame, profile: Dict[str, Any], audit_log: pd.DataFrame) -> str:
    """
    Generates a standalone interactive HTML report with glassmorphism styling.
    """
    scorecard = profile.get("scorecard", {})
    overall_score = scorecard.get("overall_score", 0)
    badge = scorecard.get("badge", "N/A")
    badge_color = scorecard.get("badge_color", "#10B981")

    table_html = df_clean.head(15).to_html(classes="styled-table", index=False)
    audit_html = audit_log.to_html(classes="styled-table", index=False) if audit_log is not None and not audit_log.empty else "<p>No cleaning operations recorded.</p>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Thiranex Solutions — Data Quality Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1e2f 100%);
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }}
        .kpi-card {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-val {{
            font-size: 32px;
            font-weight: bold;
            color: #6366f1;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: bold;
            background: {badge_color};
            color: #ffffff;
        }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        .styled-table th {{
            background-color: #6366f1;
            color: white;
            text-align: left;
            padding: 12px;
        }}
        .styled-table td {{
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="glass-card">
            <h1>⚡ Thiranex Solutions — Data Cleaning & Quality Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="glass-card">
            <h2>Data Quality Scorecard</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div>Overall Health Score</div>
                    <div class="kpi-val">{overall_score} / 100</div>
                    <div class="badge">{badge}</div>
                </div>
                <div class="kpi-card">
                    <div>Completeness</div>
                    <div class="kpi-val">{scorecard.get('completeness_score', 0)}%</div>
                    <div>Missing Cells: {scorecard.get('total_missing', 0)}</div>
                </div>
                <div class="kpi-card">
                    <div>Accuracy</div>
                    <div class="kpi-val">{scorecard.get('accuracy_score', 0)}%</div>
                    <div>Outliers: {scorecard.get('total_outliers', 0)}</div>
                </div>
                <div class="kpi-card">
                    <div>Consistency</div>
                    <div class="kpi-val">{scorecard.get('consistency_score', 0)}%</div>
                    <div>Duplicates: {scorecard.get('total_duplicates', 0)}</div>
                </div>
            </div>
        </div>

        <div class="glass-card">
            <h2>Cleaning Audit Log</h2>
            {audit_html}
        </div>

        <div class="glass-card">
            <h2>Cleaned Dataset Preview (First 15 Rows)</h2>
            {table_html}
        </div>
    </div>
</body>
</html>"""
    return html_content
