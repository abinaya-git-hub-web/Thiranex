"""
Module: exporters.py
Description: Generates multi-format export files including CSV datasets and PDF Executive Reports.
"""

import io
import pandas as pd
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Converts DataFrame into UTF-8 encoded CSV bytes for download.

    Args:
        df (pd.DataFrame): DataFrame to export.

    Returns:
        bytes: Encoded CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf_report(df: pd.DataFrame, kpis: Dict[str, Any], mappings: Dict[str, Optional[str]]) -> bytes:
    """
    Generates a professional PDF Executive Sales Report for Thiranex Solutions using ReportLab.

    Args:
        df (pd.DataFrame): Filtered sales dataset.
        kpis (Dict[str, Any]): Calculated KPI metrics.
        mappings (Dict[str, Optional[str]]): Column role mappings.

    Returns:
        bytes: PDF binary content stream.
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

    styles = getSampleStyleSheet()

    # Custom ReportLab Paragraph Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#6366F1")
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )

    elements = []

    # Title & Subtitle Header
    elements.append(Paragraph("THIRANEX SOLUTIONS", title_style))
    elements.append(Paragraph("Executive Sales & Revenue Performance Report", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#6366F1"), spaceBefore=2, spaceAfter=12))

    # Executive Summary Paragraph
    summary_text = (
        f"This official executive report summarizes key sales performance metrics for Thiranex Solutions. "
        f"During the analyzed period, total generated revenue reached <b>${kpis.get('total_revenue', 0.0):,.2f}</b> "
        f"across <b>{kpis.get('total_orders', 0):,}</b> total customer orders. "
        f"Average Order Value (AOV) stands at <b>${kpis.get('aov', 0.0):,.2f}</b>, with top performing product "
        f"<b>{kpis.get('top_product', 'N/A')}</b>."
    )
    elements.append(Paragraph("Executive Performance Brief", section_heading))
    elements.append(Paragraph(summary_text, body_style))
    elements.append(Spacer(1, 12))

    # Key Metrics Table Grid
    elements.append(Paragraph("Key Performance Indicators (KPIs)", section_heading))
    
    kpi_data = [
        ["Metric Name", "Current Value", "Period Trend (% Change)"],
        ["Total Revenue", f"${kpis.get('total_revenue', 0.0):,.2f}", f"{kpis.get('revenue_delta', 0.0):+.1f}%"],
        ["Total Orders Count", f"{kpis.get('total_orders', 0):,}", f"{kpis.get('orders_delta', 0.0):+.1f}%"],
        ["Average Order Value (AOV)", f"${kpis.get('aov', 0.0):,.2f}", f"{kpis.get('aov_delta', 0.0):+.1f}%"],
        ["Total Units Sold", f"{kpis.get('units_sold', 0):,}", f"{kpis.get('units_delta', 0.0):+.1f}%"],
        ["Active Unique Customers", f"{kpis.get('active_customers', 0):,}", f"{kpis.get('customers_delta', 0.0):+.1f}%"]
    ]

    kpi_table = Table(kpi_data, colWidths=[200, 170, 170])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366F1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    elements.append(kpi_table)
    elements.append(Spacer(1, 16))

    # Top 5 Products Table
    prod_col = mappings.get("product")
    rev_col = mappings.get("revenue")
    qty_col = mappings.get("quantity")

    if prod_col and rev_col and prod_col in df.columns:
        elements.append(Paragraph("Top Performing Products Breakdown", section_heading))
        agg_cols = {rev_col: "sum"}
        if qty_col:
            agg_cols[qty_col] = "sum"
        
        top_df = df.groupby(prod_col).agg(agg_cols).sort_values(rev_col, ascending=False).head(5).reset_index()

        top_table_data = [["Product Name", "Total Revenue ($)", "Units Sold"]]
        for _, row in top_df.iterrows():
            qty_val = str(int(row[qty_col])) if qty_col in row else "N/A"
            top_table_data.append([
                str(row[prod_col]),
                f"${float(row[rev_col]):,.2f}",
                qty_val
            ])

        top_table = Table(top_table_data, colWidths=[250, 150, 140])
        top_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(top_table)

    elements.append(Spacer(1, 24))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceBefore=10, spaceAfter=10))
    elements.append(Paragraph("© 2026 Thiranex Solutions — Enterprise Analytics Department", subtitle_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
