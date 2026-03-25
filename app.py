import cv2
import streamlit as st

from src.core import FaceEngine
from src.ui.styles import set_background_sidebar, apply_global_styles
from src.ui.pages import (
    image_recognition,
    video_recognition,
    realtime_recognition,
    video_train,
    realtime_train,
)

st.set_page_config(page_title="FaceReg")


def get_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    engine = FaceEngine()

    add_selectbox = st.sidebar.selectbox(
        "What do you want to do?",
        ("Recognition", "Train a new face"),
    )

    set_background_sidebar("download.jpg")
    apply_global_styles()

    if add_selectbox == "Recognition":
        with st.sidebar:
            add_radio = st.radio(
                "Choose mode:", ("Image", "Video", "Real-Time")
            )
        if add_radio == "Image":
            image_recognition(engine)
        elif add_radio == "Video":
            video_recognition(engine)
        else:
            realtime_recognition(engine, get_camera())
    else:
        with st.sidebar:
            add_radio = st.radio("Choose mode:", ("Real-time", "Video"))
        if add_radio == "Video":
            video_train(engine)
        else:
            realtime_train(engine, get_camera())


if __name__ == "__main__":
    main()
