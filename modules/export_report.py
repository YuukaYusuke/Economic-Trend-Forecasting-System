import io
from datetime import datetime

import pandas as pd


def report_dataframe(iso, name, live, predicted, direction, confidence, metrics: dict):
    return pd.DataFrame(
        [
            {"Field": "Mata Uang", "Value": f"{iso} — {name}"},
            {"Field": "Tanggal", "Value": datetime.now().strftime("%Y-%m-%d %H:%M")},
            {"Field": "Kurs Live", "Value": f"{live:,.4f}"},
            {"Field": "Prediksi LSTM", "Value": f"{predicted:,.4f}"},
            {"Field": "Arah", "Value": direction},
            {"Field": "Keyakinan", "Value": confidence},
            {"Field": "MAE (uji)", "Value": metrics.get("mae", "—")},
            {"Field": "RMSE (uji)", "Value": metrics.get("rmse", "—")},
            {"Field": "MAPE (uji)", "Value": metrics.get("mape", "—")},
            {"Field": "Disclaimer", "Value": "Bukan saran investasi."},
        ]
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_pdf_bytes(title: str, summary_df: pd.DataFrame) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    table = Table(data, colWidths=[160, 300])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return buf.getvalue()
