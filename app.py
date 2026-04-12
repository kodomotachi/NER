import streamlit as st
from pathlib import Path

_base = Path(__file__).resolve().parent
_icon_path = _base / "domixi.ico"
_page_icon = str(_icon_path) if _icon_path.is_file() else "🎭"

st.set_page_config(
    page_title="Masking System",
    page_icon=_page_icon,
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
