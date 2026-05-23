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
CHART_MARGIN = dict(l=56, r=28, t=52, b=88)

CHART_LEGEND = dict(
    orientation="h",
    yanchor="top",
    y=-0.22,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor=CHART["grid"],
    borderwidth=1,
    font=dict(size=11),
    tracegroupgap=12,
)


def apply_layout(
    fig,
    title: str,
    y_title: str = "Nilai",
    height: int = CHART_HEIGHT,
    *,
    show_legend: bool = True,
    bottom_margin: int | None = None,
):
    margin = dict(CHART_MARGIN)
    if bottom_margin is not None:
        margin["b"] = bottom_margin
    layout = dict(
        title=dict(
            text=title,
            font=dict(size=16, color=CHART["text"], family="DM Sans"),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        font=dict(family="DM Sans, sans-serif", color=CHART["text"], size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART["bg"],
        hovermode="x unified",
        height=height,
        margin=margin,
    )
    if show_legend:
        layout["legend"] = dict(CHART_LEGEND)
    else:
        layout["showlegend"] = False
    fig.update_layout(**layout)
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
