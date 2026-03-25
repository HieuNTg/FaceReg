import cv2
import io
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

from src.ui.styles import set_background

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


def image_recognition(engine):
    st.title("Demo FaceRecognition")
    set_background("img_reg.gif")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpeg", "jpg"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file))
        img = engine.process(img)
        st.image(img)


def video_recognition(engine):
    st.title("Demo FaceRecognition")
    set_background("img_reg.gif")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            text = st.empty()
        text.write("Processing video...")

        g = io.BytesIO(uploaded_file.read())
        temporary_location = "testout_simple.mp4"
        with open(temporary_location, "wb") as out:
            out.write(g.read())

        output_video_path = "result.mp4"
        sf = st.empty()
        video_stream = cv2.VideoCapture(temporary_location)

        frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_stream.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = video_stream.read()
            if not ret:
                break
            frame = engine.process(frame)
            sf.image(frame, channels="BGR")
            out.write(frame)

        video_stream.release()
        out.release()

        text.write("Download the processed video below:")
        with open("result.mp4", "rb") as file:
            with col2:
                st.download_button(
                    label="Download",
                    data=file,
                    file_name="result.mp4",
                    mime="video/mp4",
                )


def realtime_recognition(engine, cap):
    st.title("Demo FaceRecognition RealTime")
    set_background("img_reg.gif")

    stframe = st.empty()
    stframe.image(str(ASSETS_DIR / "default_frame.jpg"), width=700, channels="BGR")

    col1, col2 = st.columns([1.8, 0.2])
    start = col1.button("Start")
    stop = col2.button("Stop")

    if start:
        while True:
            ret, frame = cap.read()
            frame = engine.process(frame)
            stframe.image(frame, width=700, channels="BGR")
            if stop:
                break

    cap.release()


def video_train(engine):
    st.title("Train by Upload Video")
    set_background("img_reg.gif")

    with st.form("form1"):
        username = st.text_input("User name:", max_chars=100)
        st.form_submit_button("Submit")

    if username:
        st.info("Please upload a video showing different angles of the face.")
        video_file = st.file_uploader("Choose a video", type=["mp4"])

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            vidcap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            frames = []
            t = 0
            while vidcap.isOpened() and t < 100:
                ret, frame = vidcap.read()
                if not ret:
                    break
                frames.append(frame)
                t += 1

            success, annotated = engine.train_from_frames(frames, username)
            for f in annotated:
                stframe.image(f, channels="BGR")

            if success:
                st.success("Training completed successfully!")
            else:
                st.error("No face detected in the video.")


def realtime_train(engine, cap):
    st.title("Train by Realtime")
    set_background("img_reg.gif")

    with st.form("form1"):
        username = st.text_input("User name:", max_chars=100)
        st.form_submit_button("Submit")

    if username:
        st.info("After pressing Start, please turn your head left, right, up, and down.")
        stframe = st.empty()
        stframe.image(str(ASSETS_DIR / "default_frame.jpg"), width=700, channels="BGR")
        train = st.button("Start")

        if train:
            frames = []
            t = 0
            while t < 100:
                ret, frame = cap.read()
                if not ret:
                    break
                stframe.image(frame, width=700, channels="BGR")
                frames.append(frame)
                t += 1

            success, _ = engine.train_from_frames(frames, username)
            if success:
                st.success("Training completed successfully!")
            else:
                st.error("No face detected.")

    cap.release()
