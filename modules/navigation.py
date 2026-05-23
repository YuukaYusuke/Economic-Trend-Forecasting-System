import streamlit as st

from modules.i18n import tn


def build_navigation():
    """Sidebar menu — judul mengikuti bahasa aktif."""
    return st.navigation(
        [
            st.Page("pages/0_Home.py", title=tn("home"), icon="🏠", default=True),
            st.Page("pages/1_Dataset.py", title=tn("dataset"), icon="📂"),
            st.Page("pages/2_Preprocessing.py", title=tn("preprocessing"), icon="⚙️"),
            st.Page("pages/3_Visualisasi.py", title=tn("visualisasi"), icon="📊"),
            st.Page("pages/4_Model_LSTM.py", title=tn("lstm"), icon="🤖"),
            st.Page("pages/5_Prediksi.py", title=tn("prediksi"), icon="🔮"),
            st.Page("pages/6_Evaluasi.py", title=tn("evaluasi"), icon="📉"),
            st.Page("pages/7_Trend.py", title=tn("trend"), icon="📈"),
            st.Page("pages/8_Insight.py", title=tn("insight"), icon="💡"),
            st.Page("pages/9_Realtime.py", title=tn("realtime"), icon="🌐"),
            st.Page("pages/11_Bandingkan.py", title=tn("bandingkan"), icon="⚖️"),
            st.Page("pages/12_Simulasi.py", title=tn("simulasi"), icon="🧪"),
        ]
    )
