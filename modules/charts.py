import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.chart_theme import CHART, CHART_HEIGHT, CHART_LEGEND, apply_layout
from modules.i18n import format_direction, t


_DATASET_SERIES_COLORS = (
    CHART["primary"],
    CHART["secondary"],
    CHART["up"],
    CHART["down"],
    CHART["warn"],
    "#8b5cf6",
    "#ec4899",
    "#0ea5e9",
)


def _hline(fig, y, color, dash="dash", width=1.5):
    fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=width)


def _last_point(fig, df: pd.DataFrame, color):
    if df.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=[df["date"].iloc[-1]],
            y=[df["rate"].iloc[-1]],
            mode="markers",
            name=t("chart_last_point"),
            marker=dict(size=9, color=color, line=dict(width=2, color="white")),
            hovertemplate=f"{t('chart_last_marker')}: %{{y:,.6f}}<extra></extra>",
            showlegend=False,
        )
    )


def _dataset_preview_single(df: pd.DataFrame, iso: str, name: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rate"],
            mode="lines",
            name=f"{iso} — {name}",
            line=dict(color=CHART["primary"], width=2.5),
            fill="tozeroy",
            fillcolor=CHART["primary_soft"],
        )
    )
    if len(df):
        avg = df["rate"].mean()
        _hline(fig, avg, CHART["warn"])
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=t("chart_avg_period", avg=avg),
                line=dict(color=CHART["warn"], dash="dash", width=1.5),
            )
        )
        _last_point(fig, df, CHART["down"])
    return apply_layout(
        fig,
        f"{iso} — {name}",
        y_title=t("chart_y_raw_rate", iso=iso),
        bottom_margin=96,
    )


def dataset_preview_chart(series_list: list[tuple[pd.DataFrame, str, str]]):
    """series_list: [(df date+rate, iso, name), ...]. Multi → indeks basis 100."""
    if not series_list:
        return apply_layout(go.Figure(), t("no_data"), y_title="")
    if len(series_list) == 1:
        df, iso, name = series_list[0]
        return _dataset_preview_single(df, iso, name)

    fig = go.Figure()
    for i, (df, iso, name) in enumerate(series_list):
        if df.empty:
            continue
        base = df["rate"].iloc[0]
        if not base:
            continue
        color = _DATASET_SERIES_COLORS[i % len(_DATASET_SERIES_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rate"] / base * 100,
                mode="lines",
                name=f"{iso} — {name}",
                line=dict(color=color, width=2.5),
            )
        )
    return apply_layout(
        fig,
        t("chart_dataset_multi_title"),
        y_title=t("chart_index_axis"),
        bottom_margin=104,
    )


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
    _hline(fig, avg_5y, CHART["warn"])
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=t("chart_avg_5y", avg=avg_5y),
            line=dict(color=CHART["warn"], dash="dash", width=1.5),
        )
    )
    _last_point(fig, df, CHART["down"])
    return apply_layout(
        fig,
        f"USD/{iso} — {name}",
        y_title=t("chart_y_rate"),
        bottom_margin=96,
    )


def prediction_chart(df_tail: pd.DataFrame, iso: str, live: float, pred: float, arah: str, lower=None, upper=None):
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
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=[last_d, last_d],
                y=[lower, upper],
                mode="lines+markers",
                name="Prediction range",
                line=dict(color=CHART["muted"], width=5),
                marker=dict(size=7, color=CHART["muted"]),
                hovertemplate="Range: %{y:,.4f}<extra></extra>",
            )
        )
    _hline(fig, live, CHART["up"])
    _hline(fig, pred, CHART["warn"], dash="dot")
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=t("chart_live_ann", val=live),
            line=dict(color=CHART["up"], dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=t("chart_pred_ann", val=pred, dir=arah),
            line=dict(color=CHART["warn"], dash="dot", width=1.5),
        )
    )
    return apply_layout(
        fig,
        t("chart_pred_title", iso=iso, dir=format_direction(arah)),
        y_title=t("chart_y_rate"),
        bottom_margin=104,
    )


def evaluation_chart(actual, predicted, iso: str, name: str):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(t("chart_eval_subplot1"), t("chart_eval_subplot2")),
        horizontal_spacing=0.14,
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
        margin=dict(l=56, r=28, t=64, b=88),
        legend=dict(CHART_LEGEND),
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
        margin=dict(l=56, r=28, t=64, b=88),
        legend=dict(CHART_LEGEND),
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
        bottom_margin=96,
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
