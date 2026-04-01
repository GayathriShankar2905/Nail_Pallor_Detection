import streamlit as st
import numpy as np

# SAFE IMPORT
try:
    import cv2
except ImportError:
    cv2 = None

from nai import analyze_nails

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Nail Pallor Detection",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.markdown("""
    <div style='text-align: center; padding: 10px'>
        <h2 style='color:#0B3D91;'>Government Health Screening System</h2>
        <h3 style='margin-top:-10px;'>Nail Pallor Anemia Detection</h3>
        <p style='color:gray;'>Early screening tool for anemia risk</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# CHECK CV2
# =========================
if cv2 is None:
    st.error("⚠️ OpenCV failed to load.")
    st.stop()

st.warning("⚠️ Allow camera access to start detection")

# =========================
# VIDEO PROCESSOR
# =========================
class NailProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            result_img, final_result, pale_count = analyze_nails(img)
        except:
            result_img = img

        return av.VideoFrame.from_ndarray(result_img, format="bgr24")


# =========================
# WEBRTC STREAM
# =========================
webrtc_streamer(
    key="nail-detection",
    video_processor_factory=NailProcessor,
    media_stream_constraints={"video": True, "audio": False},
)