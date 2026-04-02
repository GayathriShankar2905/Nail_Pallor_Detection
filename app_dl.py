import streamlit as st
import numpy as np
from PIL import Image
import cv2
from hand import analyze_palm  # Fixed the import to match your filename

# Page Configuration
st.set_page_config(page_title="Palm Pallor Screening", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0B3D91;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 10px'>
        <h1 style='color:#0B3D91;'>VitalsCheck: Anemia Screening</h1>
        <h3 style='margin-top:-10px; color:#555;'>Palm Pallor Detection System</h3>
    </div>
""", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.header("📋 Setup & Tips")
    st.info("1. Use bright, natural light.\n2. Keep palm flat and parallel to camera.\n3. Remove henna or ink.")
    st.warning("⚠️ This is a screening tool, not a medical diagnosis.")

camera_image = st.camera_input("📷 Position your palm in the center")

if camera_image is not None:
    image = Image.open(camera_image)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Analyzing Palmar Tissue..."):
        result_img, final_result, pale_status = analyze_palm(image_bgr)

    st.subheader("Analysis View")
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="ROI Analysis", use_container_width=True)

    st.divider()
    
    st.markdown("### 🧾 Clinical Summary")
    
    if "High" in final_result:
        st.error(f"### {final_result}")
    elif "Moderate" in final_result:
        st.warning(f"### {final_result}")
    else:
        st.success(f"### {final_result}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Hand Detected", value="Yes" if "not detected" not in final_result else "No")
    with col2:
        val = "N/A" if "not detected" in final_result else ("Positive" if pale_status == 1 else "Negative")
        st.metric(label="Pallor Finding", value=val)

st.divider()
st.caption("Powered by MediaPipe & OpenCV LAB Analysis.")