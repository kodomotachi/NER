"""Streamlit UI for the Masking System app."""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def _get_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


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
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.image(uploaded_file, use_container_width=True)
        with st.spinner("Đang nhận dạng văn bản..."):
            image = np.array(Image.open(uploaded_file).convert("RGB"))
            ocr_result = _get_ocr().predict(image)
            results = ocr_result[0]["rec_texts"] if ocr_result else []
        st.text_area(
            "Văn bản trích xuất",
            value="\n".join(results),
            height=200,
        )
