"""Palet & layout seragam — terinspirasi dashboard fintech (Stripe, Linear)."""

CHART = {
    "primary": "#0d9488",
    "primary_soft": "rgba(13, 148, 136, 0.15)",
    "secondary": "#6366f1",
    "up": "#10b981",
    "down": "#ef4444",
    "warn": "#f59e0b",
    "muted": "#94a3b8",
    "grid": "#e2e8f0",
    "bg": "#f8fafc",
    "text": "#0f172a",
    "text_soft": "#64748b",
}

CHART_HEIGHT = 440
CHART_MARGIN = dict(l=48, r=32, t=72, b=48)


def apply_layout(fig, title: str, y_title: str = "Nilai", height: int = CHART_HEIGHT):
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=CHART["text"], family="DM Sans")),
        font=dict(family="DM Sans, sans-serif", color=CHART["text"], size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART["bg"],
        hovermode="x unified",
        height=height,
        margin=CHART_MARGIN,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART["grid"],
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=CHART["grid"],
        linecolor=CHART["grid"],
        title_font=dict(size=12, color=CHART["text_soft"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=CHART["grid"],
        linecolor=CHART["grid"],
        title_text=y_title,
        title_font=dict(size=12, color=CHART["text_soft"]),
    )
    return fig
