"""Streamlit UI for the Masking System app."""
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run() -> None:
    icon_path = _PROJECT_ROOT / "domixi.ico"
    page_icon = str(icon_path) if icon_path.is_file() else "🎭"

    st.set_page_config(
        page_title="Masking System",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Masking System")

    st.markdown(
        """
<style>
    [data-testid="stFileUploader"] {
        border: 2px dashed #c0c0c0;
        border-radius: 12px;
        padding: 0;
        background-color: #fafafa;
        position: relative;
        min-height: 250px;
    }
    [data-testid="stFileUploader"] > section {
        min-height: 250px;
        padding: 20px;
        border-radius: 12px;
        background-color: #fafafa;
        display: flex;
        flex-direction: row-reverse;
        flex-wrap: wrap;
        align-items: flex-end;
        align-content: flex-end;
        justify-content: flex-start;
        gap: 10px;
    }
</style>
""",
        unsafe_allow_html=True,
    )

    with st.container():
        uploaded_file = st.file_uploader(
            "Drop file here or click to upload",
            label_visibility="collapsed",
        )
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
