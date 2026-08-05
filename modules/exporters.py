"""
Module: exporters.py
Description: PDF Report Generation Studio using ReportLab and CSV Data Exporter for Thiranex Solutions.
"""

import io
import pandas as pd
from typing import List, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_personas_report(df: pd.DataFrame, personas: List[Dict[str, Any]], method_name: str) -> bytes:
    """
    Generates an executive PDF Customer Personas & Segmentation Report with ReportLab.

    Returns:
        bytes: Binary content of generated PDF file.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom Color Palette for PDF
    primary_color = colors.HexColor("#8B5CF6")
    dark_bg = colors.HexColor("#0F172A")
    light_text = colors.HexColor("#475569")

    # Title Banner Style
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=light_text,
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    header_style = ParagraphStyle(
        "Header",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    # 1. Header Banner
    story.append(Paragraph("THIRANEX SOLUTIONS", title_style))
    story.append(Paragraph(f"Executive Customer Personas & Segmentation Report — Method: <b>{method_name}</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=15))

    # 2. Executive Summary Metrics Box
    total_cust = len(df)
    total_spend = df["Total Spend ($)"].sum() if "Total Spend ($)" in df.columns else df["Monetary_Val"].sum() if "Monetary_Val" in df.columns else 0.0

    summary_data = [
        [
            Paragraph("<b>Total Customers Profiled</b>", body_style),
            Paragraph("<b>Total Revenue Analyzed</b>", body_style),
            Paragraph("<b>Active Segments Count</b>", body_style)
        ],
        [
            Paragraph(f"<b>{total_cust:,}</b>", title_style),
            Paragraph(f"<b>${total_spend:,.2f}</b>", title_style),
            Paragraph(f"<b>{len(personas)}</b>", title_style)
        ]
    ]

    summary_table = Table(summary_data, colWidths=[180, 180, 180])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # 3. Personas Breakdown Section
    story.append(Paragraph("<b>Detailed Customer Personas & Marketing Roadmap</b>", ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, textColor=dark_bg)))
    story.append(Spacer(1, 8))

    table_headers = ["Segment Persona", "Size / Share", "Avg Spend & AOV", "Key Demographics", "Recommended Strategy & Offer"]
    table_rows = [[Paragraph(h, header_style) for h in table_headers]]

    for p in personas:
        row = [
            Paragraph(f"<b>{p['creative_name']}</b><br><font color='#8B5CF6'>Risk: {p['risk_level']}</font>", body_style),
            Paragraph(f"{p['count']:,} cust<br>({p['percentage']}%)", body_style),
            Paragraph(f"${p['avg_spend']:,.0f} spend<br>AOV: ${p['avg_aov']}", body_style),
            Paragraph(f"Age: {p['demographics']['age_range']}<br>Inc: {p['demographics']['avg_income']}", body_style),
            Paragraph(f"<b>Strategy:</b> {p['strategy']}<br><b>Offer:</b> {p['recommended_offers']}", body_style)
        ]
        table_rows.append(row)

    persona_table = Table(table_rows, colWidths=[110, 65, 85, 110, 170])
    persona_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), dark_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(persona_table)
    story.append(Spacer(1, 20))

    # 4. Footer & Copyright Notice
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph("© 2026 Thiranex Solutions — Built with ❤️ for Enterprise Intelligence", ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#94A3B8"), alignment=1)))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Exports full DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
