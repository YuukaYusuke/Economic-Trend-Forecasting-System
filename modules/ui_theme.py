import streamlit as st
from datetime import datetime

LIGHT = {
    "ink": "#0f172a",
    "ink_soft": "#475569",
    "ink_muted": "#94a3b8",
    "surface": "#ffffff",
    "surface_alt": "#f1f5f9",
    "border": "#e2e8f0",
    "brand": "#0d9488",
    "up": "#059669",
    "down": "#dc2626",
    "bg": "#f8fafc",
    "hero_from": "#0f172a",
    "hero_mid": "#134e4a",
}

DARK = {
    "ink": "#f1f5f9",
    "ink_soft": "#cbd5e1",
    "ink_muted": "#94a3b8",
    "surface": "#1e293b",
    "surface_alt": "#0f172a",
    "border": "#334155",
    "brand": "#2dd4bf",
    "up": "#34d399",
    "down": "#f87171",
    "bg": "#0f172a",
    "hero_from": "#020617",
    "hero_mid": "#134e4a",
}


def _p(dark: bool):
    return DARK if dark else LIGHT


def inject_theme(dark: bool = False, rtl: bool = False):
    p = _p(dark)
    dir_css = "direction: rtl;" if rtl else "direction: ltr;"
    align = "right" if rtl else "left"
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: {'"Noto Sans Arabic", ' if rtl else ''}'Plus Jakarta Sans', 'Noto Sans JP', system-ui, sans-serif !important;
        {dir_css}
        text-align: {align};
    }}
    .stApp {{ background: {p['bg']}; }}
    .block-container {{
        {dir_css}
        padding-top: 1.5rem;
        max-width: 1180px;
    }}

    .fx-hero {{
        background: linear-gradient(125deg, {p['hero_from']} 0%, {p['hero_mid']} 50%, {p['brand']} 100%);
        border-radius: 20px; padding: 2rem 2.25rem; color: #f8fafc;
        margin-bottom: 1.75rem; box-shadow: 0 20px 50px -12px rgba(0,0,0,0.35);
    }}
    .fx-hero h1 {{ color: #fff !important; font-weight: 700; font-size: 1.85rem; margin: 0 0 0.5rem 0; }}
    .fx-hero p {{ color: #cbd5e1; margin: 0; line-height: 1.6; }}
    .fx-badge {{
        display: inline-block; background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2); border-radius: 999px;
        padding: 0.25rem 0.75rem; font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.75rem;
    }}
    .fx-section {{
        font-size: 1.1rem; font-weight: 700; color: {p['ink']};
        margin: 1.75rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid {p['brand']}; display: inline-block;
    }}
    .fx-card {{
        background: {p['surface']}; border: 1px solid {p['border']};
        border-radius: 14px; padding: 1.15rem 1.35rem; text-align: center;
        min-height: 96px; box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }}
    .fx-card-label {{ font-size: 0.72rem; color: {p['ink_muted']}; text-transform: uppercase; font-weight: 600; }}
    .fx-card-value {{ font-size: 1.55rem; font-weight: 700; color: {p['ink']}; margin-top: 0.35rem; }}
    .fx-card-up .fx-card-value {{ color: {p['up']}; }}
    .fx-card-down .fx-card-value {{ color: {p['down']}; }}
    .fx-note {{
        background: {p['surface_alt']}; border-left: 4px solid {p['brand']};
        padding: 0.9rem 1.15rem; border-radius: 0 12px 12px 0;
        color: {p['ink_soft']}; font-size: 0.9rem; margin-bottom: 1.25rem;
    }}
    .fx-panel {{
        background: {p['surface']}; border: 1px solid {p['border']};
        border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
    }}
    .fx-footer {{
        margin-top: 2.5rem; padding: 1rem 1.25rem; border-top: 1px solid {p['border']};
        font-size: 0.78rem; color: {p['ink_muted']}; line-height: 1.6;
    }}
    .fx-skeleton {{
        background: linear-gradient(90deg, {p['surface_alt']} 25%, {p['border']} 50%, {p['surface_alt']} 75%);
        background-size: 200% 100%; animation: shimmer 1.2s infinite;
        border-radius: 12px; height: 120px; margin-bottom: 1rem;
    }}
    @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
    div[data-testid="stMetric"] {{
        background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 12px; padding: 0.75rem 1rem;
    }}
    div[data-testid="stMetric"] label {{ color: {p['ink_soft']} !important; }}
    div[data-testid="stMetricValue"] {{ color: {p['ink']} !important; }}
    .stPlotlyChart {{
        border-radius: 14px; border: 1px solid {p['border']}; background: {p['surface']}; padding: 0.5rem;
    }}

    .lang-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {p['ink_muted']};
        margin: 0 0 0.35rem 0;
    }}
    [data-testid="stSidebar"] .stButton button {{
        font-weight: 600;
        border-radius: 8px;
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badge: str = "FX"):
    st.markdown(
        f'<div class="fx-hero"><div class="fx-badge">{badge}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(f'<div class="fx-section">{title}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="fx-note">{text}</div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, variant: str = "neutral"):
    st.markdown(
        f'<div class="fx-card fx-card-{variant}"><div class="fx-card-label">{label}</div>'
        f'<div class="fx-card-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def skeleton_block():
    st.markdown('<div class="fx-skeleton"></div>', unsafe_allow_html=True)


def footer(source: str = "", model_info: str = ""):
    from modules.i18n import t

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
<div class="fx-footer">
<strong>{t('disclaimer')}</strong><br>
{t('footer_source')}: {source or 'dataset.csv · API'} ·
{t('footer_updated')}: {ts} · {t('footer_model')}: {model_info or 'LSTM'}
</div>
        """,
        unsafe_allow_html=True,
    )
