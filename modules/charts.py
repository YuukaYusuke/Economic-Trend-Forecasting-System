import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.chart_theme import CHART, CHART_HEIGHT, apply_layout
from modules.i18n import t


def history_chart(df: pd.DataFrame, iso: str, name: str):
    cutoff = df["date"].max() - pd.DateOffset(years=5)
    recent = df[df["date"] >= cutoff]
    avg_5y = recent["rate"].mean() if len(recent) else df["rate"].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rate"],
            mode="lines",
            name=t("chart_historical"),
            line=dict(color=CHART["primary"], width=2.5),
            fill="tozeroy",
            fillcolor=CHART["primary_soft"],
        )
    )
    fig.add_hline(
        y=avg_5y,
        line_dash="dash",
        line_color=CHART["warn"],
        line_width=2,
        annotation_text=t("chart_avg_5y", avg=avg_5y),
        annotation_font_color=CHART["text_soft"],
    )
    fig.add_trace(
        go.Scatter(
            x=[df["date"].iloc[-1]],
            y=[df["rate"].iloc[-1]],
            mode="markers+text",
            text=[t("chart_last_marker")],
            textposition="top center",
            name=t("chart_last_point"),
            marker=dict(size=11, color=CHART["down"], line=dict(width=2, color="white")),
        )
    )
    return apply_layout(
        fig,
        f"USD/{iso} — {name}",
        y_title=t("chart_y_rate"),
    )


def prediction_chart(df_tail: pd.DataFrame, iso: str, live: float, pred: float, arah: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_tail["date"],
            y=df_tail["rate"],
            mode="lines",
            name=t("chart_hist_short"),
            line=dict(color=CHART["primary"], width=2.5),
            fill="tozeroy",
            fillcolor=CHART["primary_soft"],
        )
    )
    last_d = df_tail["date"].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last_d, last_d],
            y=[live, pred],
            mode="lines+markers",
            name=t("chart_live_to_pred"),
            line=dict(color=CHART["secondary"], width=2, dash="dot"),
            marker=dict(size=10, color=[CHART["up"], CHART["warn"]]),
        )
    )
    fig.add_hline(
        y=live,
        line_dash="dash",
        line_color=CHART["up"],
        line_width=2,
        annotation_text=t("chart_live_ann", val=live),
    )
    fig.add_hline(
        y=pred,
        line_dash="dot",
        line_color=CHART["warn"],
        line_width=2,
        annotation_text=t("chart_pred_ann", val=pred, dir=arah),
    )
    return apply_layout(
        fig,
        t("chart_pred_title", iso=iso, dir=arah),
        y_title=t("chart_y_rate"),
    )


def evaluation_chart(actual, predicted, iso: str, name: str):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(t("chart_eval_subplot1"), t("chart_eval_subplot2")),
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Scatter(
            y=actual,
            mode="lines",
            name=t("chart_actual"),
            line=dict(color=CHART["primary"], width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=predicted,
            mode="lines",
            name=t("chart_forecast"),
            line=dict(color=CHART["secondary"], width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            name=t("chart_point"),
            marker=dict(color=CHART["primary"], size=5, opacity=0.55),
        ),
        row=1,
        col=2,
    )
    mn, mx = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
    fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            name=t("chart_ideal"),
            line=dict(color=CHART["muted"], dash="dot"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        font=dict(family="DM Sans, sans-serif", color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART["bg"],
        height=CHART_HEIGHT,
        margin=dict(l=48, r=32, t=80, b=48),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(gridcolor=CHART["grid"], row=1, col=2, title_text=t("chart_actual"))
    fig.update_yaxes(gridcolor=CHART["grid"], row=1, col=1, title_text=t("chart_value_axis"))
    fig.update_yaxes(gridcolor=CHART["grid"], row=1, col=2, title_text=t("chart_forecast"))
    if len(fig.layout.annotations) >= 2:
        fig.layout.annotations[0].update(font=dict(size=13, color=CHART["text_soft"]))
        fig.layout.annotations[1].update(font=dict(size=13, color=CHART["text_soft"]))
    return fig


def residual_chart(actual, predicted):
    residual = actual - predicted
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=residual,
            mode="lines",
            name="Residual",
            line=dict(color=CHART["secondary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.12)",
        )
    )
    fig.add_hline(y=0, line_color=CHART["muted"], line_width=1)
    return apply_layout(fig, t("chart_residual_title"), y_title=t("chart_gap_axis"))


def evaluation_with_baseline(actual, lstm_pred, ma_pred, iso: str):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(t("chart_model_sub1"), t("chart_model_sub2")),
    )
    fig.add_trace(
        go.Scatter(y=actual, name=t("chart_actual"), line=dict(color=CHART["primary"], width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=lstm_pred, name="LSTM", line=dict(color=CHART["secondary"], dash="dash")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=ma_pred,
            name=t("chart_moving_avg"),
            line=dict(color=CHART["warn"], dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ma_pred,
            y=lstm_pred,
            mode="markers",
            marker=dict(size=4, color=CHART["primary"], opacity=0.5),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        font=dict(family="DM Sans", color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART["bg"],
        height=CHART_HEIGHT,
        margin=dict(l=48, r=32, t=72, b=48),
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_xaxes(gridcolor=CHART["grid"], row=1, col=2, title_text=t("chart_ma_axis"))
    fig.update_yaxes(gridcolor=CHART["grid"], row=1, col=2, title_text=t("chart_lstm_axis"))
    return fig


def dual_currency_chart(df1, df2, iso1, iso2, name1, name2):
    """Normalized index base 100."""
    b1, b2 = df1["rate"].iloc[0], df2["rate"].iloc[0]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df1["date"],
            y=df1["rate"] / b1 * 100,
            name=f"{iso1} ({name1})",
            line=dict(color=CHART["primary"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df2["date"],
            y=df2["rate"] / b2 * 100,
            name=f"{iso2} ({name2})",
            line=dict(color=CHART["secondary"], width=2),
        )
    )
    return apply_layout(
        fig,
        t("chart_index_title", iso1=iso1, iso2=iso2),
        y_title=t("chart_index_axis"),
    )


def trend_heatmap(df_rt):
    """df_rt needs Kode, Perubahan (%) numeric."""
    sub = df_rt[df_rt["Perubahan (%)"].notna()].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("Perubahan (%)", ascending=False)
    fig = go.Figure(
        data=go.Heatmap(
            z=[sub["Perubahan (%)"].tolist()],
            x=sub["Kode"].tolist(),
            y=[t("chart_heatmap_row")],
            colorscale=[[0, CHART["down"]], [0.5, "#f8fafc"], [1, CHART["up"]]],
            zmid=0,
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=40, r=20, t=40, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART["bg"],
        font=dict(family="DM Sans", size=11),
    )
    return fig


def sparkline_figure(values, color=None):
    fig = go.Figure(
        go.Scatter(
            y=list(values),
            mode="lines",
            line=dict(color=color or CHART["primary"], width=1.5),
            fill="tozeroy",
            fillcolor=CHART["primary_soft"],
        )
    )
    fig.update_layout(
        height=48,
        width=140,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
