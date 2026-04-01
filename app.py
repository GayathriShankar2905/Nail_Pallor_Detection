import streamlit as st
import numpy as np
from PIL import Image
import cv2
from nai import analyze_nails

st.set_page_config(page_title="Nail Pallor Detection", layout="centered")

st.markdown("""
    <div style='text-align: center; padding: 10px'>
        <h2 style='color:#0B3D91;'>Government Health Screening System</h2>
        <h3 style='margin-top:-10px;'>Nail Pallor Anemia Detection</h3>
        <p style='color:gray;'>Early screening tool for anemia risk</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
### 📌 Instructions:
- Place your hand clearly in front of camera  
- Keep fingers slightly spread  
- Ensure good lighting  
- Avoid blur  
""")

camera_image = st.camera_input("📷 Capture your hand")

if camera_image is not None:
    image = Image.open(camera_image)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Captured Image", use_container_width=True)

    with st.spinner("Analyzing nails... (first run downloads model, ~5s)"):
        result_img, final_result, pale_count = analyze_nails(image_bgr)

    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Analyzed Image", use_container_width=True)

    st.divider()
    st.markdown("## 🧾 Result Summary")

    if "High" in final_result:
        st.error("🔴 High Anemia Risk")
    elif "Moderate" in final_result:
        st.warning("🟠 Moderate Risk")
    else:
        st.success("🟢 Normal")

    st.markdown(f"**Pale Nails Detected:** {pale_count}")
    st.divider()

    st.markdown("""
    <small>⚠️ This is a preliminary screening tool and not a medical diagnosis.  
    Please consult a doctor for confirmation.</small>
    """, unsafe_allow_html=True)