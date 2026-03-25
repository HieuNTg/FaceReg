import base64
import streamlit as st
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def set_background(gif_file):
    base64_str = get_base64(ASSETS_DIR / gif_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/gif;base64,{base64_str}");
            background-size: cover;
            background-attachment: fixed;
            margin: 0; padding: 0;
            height: 100vh;
            overflow: hidden;
            opacity: 0.9;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_background_sidebar(img_file):
    base64_str = get_base64(ASSETS_DIR / img_file)
    st.markdown(
        f"""
        <style>
        .st-emotion-cache-1cypcdb {{
            background-image: url("data:image/jpg;base64,{base64_str}");
            background-size: cover;
            margin: 0; padding: 0;
            height: 100vh;
            overflow: hidden;
            opacity: 0.9;
            object-fit: contain;
            max-width: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_global_styles():
    st.markdown(
        """
        <style>
        .st-emotion-cache-1avcm0n { height: 0rem; }
        .st-emotion-cache-10trblm {
            font-family: "Georgia", sans-serif;
            text-align: center;
            color: blue;
        }
        .st-emotion-cache-q8sbsg {
            font-family: "Georgia", sans-serif;
            color: blue;
        }
        .st-emotion-cache-q8sbsg p {
            font-family: "Georgia", sans-serif;
            font-size: 16px;
        }
        .st-emotion-cache-1erivf3 {
            font-family: "Georgia", sans-serif;
            background-color: transparent;
            color: red;
        }
        .st-emotion-cache-noeb3a { color: red; }
        .st-emotion-cache-7oyrr6 {
            font-family: "Georgia", sans-serif;
            color: red;
        }
        .st-emotion-cache-19rxjzo { border: 2px solid blue; }
        .st-emotion-cache-1uixxvy { color: red; }
        .st-emotion-cache-vskyf7 { color: red; }
        img { border: 2px solid blue; }
        .st-emotion-cache-x78sv8 {
            font-family: "Georgia", sans-serif;
            color: red;
        }
        .st-emotion-cache-1xw8zd0 {
            border: 2px solid blue;
            border-radius: 0.5rem;
        }
        .st-f2 { color: yellow; }
        .st-f8 { color: yellow; }
        .st-bn { color: red; }
        .st-az { font-family: "Georgia", sans-serif; }
        .st-bb {
            background-color: transparent;
            border: 2px solid blue;
        }
        .st-emotion-cache-15wihvi { background: #7CFC00; }
        .st-emotion-cache-fblp2m { color: red; }
        .st-b6 { color: red; }
        </style>
        """,
        unsafe_allow_html=True,
    )
