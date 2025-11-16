import streamlit as st
import requests
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------
API_URL = "http://localhost:8000"   # change if deployed

st.set_page_config(page_title="Deepfake Detection UI", page_icon="🕵️‍♂️", layout="wide")
st.title("🕵️‍♂️ Deepfake Detection System")
st.markdown("Upload **Images**, **Audio**, or **Videos** to analyze using your FastAPI backend.")

media_type = st.selectbox("Select media type:", ["Image", "Audio", "Video"])

# ------------------------------------------
# FILE UPLOAD COMPONENT
# ------------------------------------------
if media_type == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif media_type == "Audio":
    uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])
else:
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])


# ------------------------------------------
# SEND FILE TO FASTAPI BACKEND
# ------------------------------------------
if uploaded is not None:
    st.info("✅ File uploaded successfully!")

    if media_type == "Image":
        st.image(
            Image.open(uploaded),
            caption="Uploaded Image",
            use_container_width=True  # ✅ FIXED
        )

        endpoint = "/analyze/image"

    elif media_type == "Audio":
        st.audio(uploaded)
        endpoint = "/analyze/audio"

    else:  # Video
        st.video(uploaded)
        endpoint = "/analyze/video"

    # ------------------------------------------
    # ANALYZE BUTTON
    # ------------------------------------------
    if st.button("Analyze"):
        with st.spinner("Analyzing using FastAPI backend…"):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}

            try:
                response = requests.post(API_URL + endpoint, files=files)

                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    data = response.json()

                    # ---------------------------
                    # DISPLAY RESULTS
                    # ---------------------------
                    st.success("✅ Analysis Complete!")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.subheader("Prediction")
                        st.write(f"**Type:** {data['type']}")
                        st.write(f"**Prediction:** `{data['prediction']}`")
                        st.write(f"**Confidence:** `{data['confidence']:.2f}%`")

                    with col2:
                        st.subheader("Feature Analysis")
                        details = data.get("details", {})
                        for k, v in details.items():
                            st.write(f"**{k}:** {v}")

            except Exception as e:
                st.error(f"Request failed: {str(e)}")
