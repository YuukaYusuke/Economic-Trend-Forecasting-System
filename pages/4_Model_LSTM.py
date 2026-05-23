import streamlit as st

from modules.config import TRAIN_EPOCHS, WINDOW
from modules.i18n import tp
from modules.model_storage import _paths
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import ModelsNotDeployedError, get_trained_model
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("lstm")

iso, column, name = currency_selector()
saved = _paths(column)[0].exists()

st.markdown(f'<div class="fx-panel">{tp("lstm", "explain")}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
stat_card(tp("lstm", "window"), f"{WINDOW}", "neutral")
stat_card(tp("lstm", "epoch"), str(TRAIN_EPOCHS), "neutral")
stat_card(tp("lstm", "file"), tp("lstm", "file_ok") if saved else tp("lstm", "file_no"), "up" if saved else "down")

with st.spinner(tp("lstm", "loading")):
    try:
        model, *_ = get_trained_model(column)
    except ModelsNotDeployedError as exc:
        st.error(str(exc))
        st.stop()

note(tp("lstm", "ready", iso=iso, params=model.count_params()))
page_footer()
